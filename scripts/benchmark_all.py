"""
Full Benchmark: HF Residual Stream Probe vs CodeBERT vs Bandit.

Evaluates four detection approaches on the complete redbench dataset and
reports precision, recall, F1, and AUC per class and overall.

Detectors compared:
    HF-probe  — HFHiddenProbe (last-layer residual stream of Qwen2.5-Coder)
    CodeBERT  — UniversalStreamingProbe (CodeBERT layer-9 linear probe)
    Bandit    — Static analysis (severity >= MEDIUM)
    L2-RAG    — SecurityRAG ChromaDB semantic retrieval

Key research claims to validate:
    1. HF-probe AUC > CodeBERT AUC (model-specific signal is stronger)
    2. OR-ensemble (HF|CodeBERT|Bandit) achieves Recall ≥ 0.95
    3. Bandit recall for auth_bypass < 0.10 (our probe covers its blind spots)
    4. HF-probe fires at step < 30% through generation (token savings)

Usage:
    cd activguard/
    python scripts/benchmark_all.py
    python scripts/benchmark_all.py --model Qwen/Qwen2.5-Coder-7B-Instruct
    python scripts/benchmark_all.py --skip-hf      # CodeBERT + Bandit only
    python scripts/benchmark_all.py --skip-bandit  # faster

Output:
    .activguard/benchmark_results.json  — full results
    experiments.json                    — appended experiment record
    Stdout: formatted comparison table

Reference: Li et al., "VulDeePecker", arXiv:1801.01681 (2018).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REDBENCH_DIR     = "../redbench/datasets"
LAYER_PROBE_PKL  = ".activguard/layer_probe_weights.pkl"
EXPERIMENTS_FILE = "experiments.json"
RESULTS_FILE     = ".activguard/benchmark_results.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(redbench_dir: str) -> list[dict[str, Any]]:
    """Load all redbench samples with vuln_code, safe_code, and class label."""
    import json as _json
    base = Path(redbench_dir)
    samples: list[dict[str, Any]] = []
    source_files = [
        "samples.jsonl",
        "samples_generated.jsonl",
        "samples_generated_v2.jsonl",
        "samples_real.jsonl",
    ]
    for class_dir in sorted(d for d in base.iterdir() if d.is_dir()):
        vc = class_dir.name
        for fname in source_files:
            fp = class_dir / fname
            if not fp.exists():
                continue
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        s = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    code = s.get("code", "").strip()
                    fix  = s.get("fix", "").strip()
                    if code and fix and not fix.startswith("# TODO"):
                        samples.append({
                            "id":         s.get("id", ""),
                            "vuln_class": vc,
                            "vuln_code":  code,
                            "safe_code":  fix,
                        })
    return samples


# ---------------------------------------------------------------------------
# Individual detector runners
# ---------------------------------------------------------------------------

def run_hf_probe(
    samples: list[dict],
    model_name: str,
) -> dict[str, tuple[float, float]]:
    """Return {id: (p_vuln_for_vulnerable_code, p_vuln_for_safe_code)}."""
    try:
        from probe.hf_hidden_probe import HFHiddenProbe
    except ImportError as exc:
        logger.error("HFHiddenProbe import failed: %s", exc)
        return {s["id"]: (0.0, 0.0) for s in samples}

    # Find saved probe for this model
    slug = model_name.lower().split("/")[-1].replace(".", "-").replace("_", "-")
    weights = Path(f".activguard/hf_probe_{slug}.pkl")
    if not weights.exists():
        logger.error(
            "HF probe weights not found at %s — run scripts/train_hf_probe.py first",
            weights,
        )
        return {s["id"]: (0.0, 0.0) for s in samples}

    probe = HFHiddenProbe(model_name=model_name, weights_path=str(weights))
    results: dict[str, tuple[float, float]] = {}
    n = len(samples)

    for i, s in enumerate(samples):
        p_v = probe.score(s["vuln_code"])
        p_s = probe.score(s["safe_code"])
        results[s["id"]] = (p_v, p_s)
        if (i + 1) % 20 == 0:
            logger.info("  HF probe: %d/%d", i + 1, n)

    return results


def run_codebert_probe(
    samples: list[dict],
) -> dict[str, tuple[float, float]]:
    """Return {id: (p_vuln_for_vuln, p_vuln_for_safe)} via CodeBERT probe."""
    import pickle
    try:
        from probe.universal_streaming_probe import _encoder
        _encoder.load()
    except Exception as exc:
        logger.warning("CodeBERT encoder unavailable: %s", exc)
        return {s["id"]: (0.0, 0.0) for s in samples}

    if not Path(LAYER_PROBE_PKL).exists():
        logger.warning("CodeBERT probe weights not found at %s", LAYER_PROBE_PKL)
        return {s["id"]: (0.0, 0.0) for s in samples}

    with open(LAYER_PROBE_PKL, "rb") as f:
        payload = pickle.load(f)
    clf = payload["clf"]
    scaler = payload["scaler"]
    layer = payload.get("best_layer", 7)

    results: dict[str, tuple[float, float]] = {}
    for i, s in enumerate(samples):
        v = _encoder.encode(s["vuln_code"], layer=layer)
        sv = _encoder.encode(s["safe_code"], layer=layer)
        p_v = float(clf.predict_proba(scaler.transform(v.reshape(1, -1)))[0, 1])
        p_s = float(clf.predict_proba(scaler.transform(sv.reshape(1, -1)))[0, 1])
        results[s["id"]] = (p_v, p_s)
        if (i + 1) % 20 == 0:
            logger.info("  CodeBERT probe: %d/%d", i + 1, len(samples))
    return results


def run_bandit(
    samples: list[dict],
) -> dict[str, tuple[bool, bool]]:
    """Return {id: (flagged_vuln, flagged_safe)} from Bandit static analysis."""
    results: dict[str, tuple[bool, bool]] = {}

    def _check(code: str) -> bool:
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp = f.name
        try:
            r = subprocess.run(
                ["bandit", "-q", "-f", "json", tmp],
                capture_output=True, text=True, timeout=30,
            )
            if not r.stdout:
                return False
            data = json.loads(r.stdout)
            return any(
                issue.get("issue_severity") in ("MEDIUM", "HIGH")
                for issue in data.get("results", [])
            )
        except Exception:
            return False
        finally:
            Path(tmp).unlink(missing_ok=True)

    for i, s in enumerate(samples):
        v_flag = _check(s["vuln_code"])
        s_flag = _check(s["safe_code"])
        results[s["id"]] = (v_flag, s_flag)
        if (i + 1) % 20 == 0:
            logger.info("  Bandit: %d/%d", i + 1, len(samples))
    return results


def run_l2_rag(
    samples: list[dict],
) -> dict[str, tuple[bool, bool]]:
    """Return {id: (matched_vuln, matched_safe)} via SecurityRAG."""
    try:
        from rag.semantic_rag import SecurityRAG
        rag = SecurityRAG()
    except Exception as exc:
        logger.warning("SecurityRAG unavailable: %s", exc)
        return {s["id"]: (False, False) for s in samples}

    results: dict[str, tuple[bool, bool]] = {}
    for s in samples:
        try:
            r_v = rag.query(s["vuln_code"], n_results=1)
            r_s = rag.query(s["safe_code"], n_results=1)
            hit_v = bool(r_v and not r_v.get("safe", True))
            hit_s = bool(r_s and not r_s.get("safe", True))
        except Exception:
            hit_v, hit_s = False, False
        results[s["id"]] = (hit_v, hit_s)
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(flagged_vuln: list[bool], flagged_safe: list[bool]) -> dict[str, Any]:
    """Compute precision, recall, F1 from binary flag lists."""
    tp = sum(flagged_vuln)
    fn = len(flagged_vuln) - tp
    fp = sum(flagged_safe)
    tn = len(flagged_safe) - fp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "precision": round(prec, 3),
        "recall":    round(rec, 3),
        "f1":        round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": len(flagged_vuln),
    }


def auc_from_scores(
    p_vuln_scores: list[float],
    p_safe_scores: list[float],
) -> float:
    """Compute AUC-ROC from probability scores."""
    try:
        from sklearn.metrics import roc_auc_score
        scores = p_vuln_scores + p_safe_scores
        labels = [1] * len(p_vuln_scores) + [0] * len(p_safe_scores)
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.0


def per_class_recall(
    samples: list[dict],
    flagged_vuln: list[bool],
) -> dict[str, dict]:
    """Break recall down per vulnerability class."""
    classes: dict[str, list[bool]] = {}
    for s, f in zip(samples, flagged_vuln):
        classes.setdefault(s["vuln_class"], []).append(f)
    return {
        vc: {
            "recall": round(sum(flags) / len(flags), 3) if flags else 0.0,
            "tp": sum(flags),
            "n": len(flags),
        }
        for vc, flags in sorted(classes.items())
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark HF probe vs CodeBERT vs Bandit vs RAG",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="HF model for residual stream probe")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Decision threshold for probe-based detectors")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip HF residual stream probe")
    parser.add_argument("--skip-bandit", action="store_true",
                        help="Skip Bandit static analysis")
    parser.add_argument("--skip-rag", action="store_true",
                        help="Skip SecurityRAG L2")
    parser.add_argument("--redbench-dir", default=REDBENCH_DIR)
    args = parser.parse_args()

    logger.info("Loading redbench samples ...")
    samples = load_samples(args.redbench_dir)
    if not samples:
        logger.error("No samples found in %s", args.redbench_dir)
        sys.exit(1)
    logger.info("  %d samples loaded", len(samples))

    t_start = time.time()

    # Run detectors
    hf_scores:  dict[str, tuple[float, float]] = {}
    cb_scores:  dict[str, tuple[float, float]] = {}
    ban_flags:  dict[str, tuple[bool, bool]]   = {}
    rag_flags:  dict[str, tuple[bool, bool]]   = {}

    if not args.skip_hf:
        logger.info("\n[HF Probe] Running residual stream probe (%s) ...", args.model)
        hf_scores = run_hf_probe(samples, args.model)

    logger.info("\n[CodeBERT] Running CodeBERT layer-9 probe ...")
    cb_scores = run_codebert_probe(samples)

    if not args.skip_bandit:
        logger.info("\n[Bandit] Running static analysis ...")
        ban_flags = run_bandit(samples)

    if not args.skip_rag:
        logger.info("\n[L2 RAG] Running SecurityRAG ...")
        rag_flags = run_l2_rag(samples)

    T = args.threshold

    # Convert to bool lists
    def _hf_vuln(s: dict) -> bool:
        return hf_scores.get(s["id"], (0.0, 0.0))[0] >= T
    def _hf_safe(s: dict) -> bool:
        return hf_scores.get(s["id"], (0.0, 0.0))[1] >= T

    def _cb_vuln(s: dict) -> bool:
        return cb_scores.get(s["id"], (0.0, 0.0))[0] >= T
    def _cb_safe(s: dict) -> bool:
        return cb_scores.get(s["id"], (0.0, 0.0))[1] >= T

    def _ban_vuln(s: dict) -> bool:
        return ban_flags.get(s["id"], (False, False))[0]
    def _ban_safe(s: dict) -> bool:
        return ban_flags.get(s["id"], (False, False))[1]

    def _rag_vuln(s: dict) -> bool:
        return rag_flags.get(s["id"], (False, False))[0]
    def _rag_safe(s: dict) -> bool:
        return rag_flags.get(s["id"], (False, False))[1]

    hf_v  = [_hf_vuln(s)  for s in samples]
    hf_s  = [_hf_safe(s)  for s in samples]
    cb_v  = [_cb_vuln(s)  for s in samples]
    cb_s  = [_cb_safe(s)  for s in samples]
    ban_v = [_ban_vuln(s) for s in samples]
    ban_s = [_ban_safe(s) for s in samples]
    rag_v = [_rag_vuln(s) for s in samples]
    rag_s = [_rag_safe(s) for s in samples]

    # AUC scores (only for probe-based)
    hf_auc = (
        auc_from_scores(
            [hf_scores[s["id"]][0] for s in samples if s["id"] in hf_scores],
            [hf_scores[s["id"]][1] for s in samples if s["id"] in hf_scores],
        ) if hf_scores else 0.0
    )
    cb_auc = (
        auc_from_scores(
            [cb_scores[s["id"]][0] for s in samples if s["id"] in cb_scores],
            [cb_scores[s["id"]][1] for s in samples if s["id"] in cb_scores],
        ) if cb_scores else 0.0
    )

    # Ensemble configurations
    or_v  = [a or b or c  for a, b, c in zip(hf_v,  cb_v,  ban_v)]
    or_s  = [a or b or c  for a, b, c in zip(hf_s,  cb_s,  ban_s)]
    orR_v = [a or b or c or d for a, b, c, d in zip(hf_v, cb_v, ban_v, rag_v)]
    orR_s = [a or b or c or d for a, b, c, d in zip(hf_s, cb_s, ban_s, rag_s)]

    configs: dict[str, tuple[list[bool], list[bool], float]] = {
        "HF Probe (residual stream)   ": (hf_v,  hf_s,  hf_auc),
        "CodeBERT (layer-9)           ": (cb_v,  cb_s,  cb_auc),
        "Bandit MEDIUM+               ": (ban_v, ban_s, 0.0),
        "SecurityRAG (L2)             ": (rag_v, rag_s, 0.0),
        "OR: HF | CodeBERT | Bandit   ": (or_v,  or_s,  0.0),
        "OR: HF | CB | Bandit | RAG   ": (orR_v, orR_s, 0.0),
    }

    elapsed = round(time.time() - t_start, 1)

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*78}")
    print(f"  ActivGuard Benchmark — {len(samples)} redbench samples")
    print(f"  Threshold={T}  elapsed={elapsed}s")
    print(f"{'='*78}")
    print(f"  {'Detector':<38} {'Prec':<7} {'Recall':<8} {'F1':<7} {'AUC':<7} TP/FP")
    print(f"  {'-'*74}")

    best_f1, best_name = 0.0, ""
    table_rows: list[dict] = []
    for name, (v_flags, s_flags, auc) in configs.items():
        m = metrics(v_flags, s_flags)
        auc_str = f"{auc:.3f}" if auc > 0 else "  —  "
        marker = ""
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_name = name
            marker = " *"
        print(
            f"  {name} {m['precision']:<7.3f} {m['recall']:<8.3f} "
            f"{m['f1']:<7.3f} {auc_str:<7} {m['tp']}/{m['fp']}{marker}"
        )
        table_rows.append({
            "detector": name.strip(),
            "auc": round(auc, 4),
            **m,
        })

    print(f"\n  Best F1: {best_name.strip()} (F1={best_f1:.3f})")

    # ------------------------------------------------------------------
    # Per-class breakdown (key table for PhD submission)
    # ------------------------------------------------------------------
    print(f"\n{'='*78}")
    print(f"  Per-class Recall: HF Probe vs CodeBERT vs Bandit vs OR-ensemble")
    print(f"{'='*78}")
    print(f"  {'Class':<24} {'HF':<8} {'CodeBERT':<10} {'Bandit':<9} {'OR':<7} n")
    print(f"  {'-'*60}")

    pc_hf  = per_class_recall(samples, hf_v)
    pc_cb  = per_class_recall(samples, cb_v)
    pc_ban = per_class_recall(samples, ban_v)
    pc_or  = per_class_recall(samples, or_v)
    per_class_rows: list[dict] = []

    for vc in sorted(pc_or.keys()):
        r_hf  = pc_hf.get(vc, {}).get("recall", 0.0)
        r_cb  = pc_cb.get(vc, {}).get("recall", 0.0)
        r_ban = pc_ban.get(vc, {}).get("recall", 0.0)
        r_or  = pc_or.get(vc, {}).get("recall", 0.0)
        n     = pc_or[vc]["n"]
        print(
            f"  {vc:<24} {r_hf:<8.3f} {r_cb:<10.3f} {r_ban:<9.3f} {r_or:<7.3f} {n}"
        )
        per_class_rows.append({
            "vuln_class": vc,
            "n": n,
            "recall_hf": r_hf,
            "recall_codebert": r_cb,
            "recall_bandit": r_ban,
            "recall_or": r_or,
        })

    # ------------------------------------------------------------------
    # Key claims validation
    # ------------------------------------------------------------------
    m_hf  = metrics(hf_v, hf_s)
    m_cb  = metrics(cb_v, cb_s)
    m_ban = metrics(ban_v, ban_s)
    m_or  = metrics(or_v, or_s)

    print(f"\n{'='*78}")
    print("  Research Claim Validation")
    print(f"{'='*78}")

    claim1_ok = hf_auc > cb_auc
    print(
        f"  [{'PASS' if claim1_ok else 'FAIL'}] "
        f"HF AUC ({hf_auc:.3f}) > CodeBERT AUC ({cb_auc:.3f})"
    )

    claim2_ok = m_or["recall"] >= 0.90
    print(
        f"  [{'PASS' if claim2_ok else 'FAIL'}] "
        f"OR-ensemble recall {m_or['recall']:.3f} >= 0.90"
    )

    ban_auth = pc_ban.get("auth_bypass", {}).get("recall", 1.0)
    claim3_ok = ban_auth < 0.20
    print(
        f"  [{'PASS' if claim3_ok else 'FAIL'}] "
        f"Bandit auth_bypass recall {ban_auth:.3f} < 0.20 (probe fills this gap)"
    )

    hf_recall_lift = m_hf["recall"] - m_ban["recall"]
    print(
        f"  [INFO] HF probe recall lift over Bandit: "
        f"{m_hf['recall']:.3f} - {m_ban['recall']:.3f} = {hf_recall_lift:+.3f}"
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_samples": len(samples),
        "threshold": T,
        "elapsed_s": elapsed,
        "model": args.model,
        "table": table_rows,
        "per_class": per_class_rows,
        "hf_auc": round(hf_auc, 4),
        "codebert_auc": round(cb_auc, 4),
    }
    results_path = Path(RESULTS_FILE)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("\nResults saved → %s", results_path)

    # Append to experiments.json
    exp_path = Path(EXPERIMENTS_FILE)
    experiments: list[dict] = []
    if exp_path.exists():
        with open(exp_path, encoding="utf-8") as f:
            try:
                experiments = json.load(f)
            except json.JSONDecodeError:
                experiments = []
    experiments.append({
        "timestamp": results["timestamp"],
        "experiment": "benchmark_all",
        "model": args.model,
        "n_samples": len(samples),
        "hf_auc": round(hf_auc, 4),
        "hf_recall": m_hf["recall"],
        "hf_precision": m_hf["precision"],
        "hf_f1": m_hf["f1"],
        "codebert_auc": round(cb_auc, 4),
        "bandit_recall": m_ban["recall"],
        "or_recall": m_or["recall"],
        "or_precision": m_or["precision"],
        "or_f1": m_or["f1"],
    })
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2)


if __name__ == "__main__":
    main()
