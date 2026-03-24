"""
Combined Layer Detection Rate Analysis.

Evaluates precision/recall for each layer individually and all ensemble
combinations.  Answers the question: does combining L1 + L2 + L3 improve
detection rate over any single layer?

Layers:
    L1 — Activation probe (CodeBERT layer 7, P(vuln) threshold)
    L2 — Semantic RAG (ChromaDB anti-pattern match)
    L3 — Static analysis (Bandit, severity >= MEDIUM)

Ensemble strategies:
    OR  — flag if ANY layer fires (maximises recall, trades precision)
    AND — flag only if ALL layers agree (maximises precision, trades recall)
    VOTE — flag if majority (2+/3) layers fire
    L1_GATE — flag if L1 fires AND (L2 OR L3) confirms (reduces L1 false positives)

Reference: Ensemble methods for vulnerability detection reviewed in
Li et al., "VulDeePecker", arXiv:1801.01681 (2018).
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

REDBENCH_DIR = "../redbench/datasets"
LAYER_PROBE_WEIGHTS = ".activguard/layer_probe_weights.pkl"
BANDIT_RESULTS = ".activguard/bandit_results.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples() -> list[dict]:
    base = Path(REDBENCH_DIR)
    samples = []
    for vc in sorted(d.name for d in base.iterdir() if d.is_dir()):
        for fname in ("samples.jsonl", "samples_generated.jsonl"):
            jsonl = base / vc / fname
            if not jsonl.exists():
                continue
            with open(jsonl, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    s = json.loads(line)
                    samples.append({
                        "id": s["id"],
                        "vuln_class": vc,
                        "vuln_code": s["code"],
                        "safe_code": s["fix"],
                    })
    return samples


# ---------------------------------------------------------------------------
# L1: CodeBERT activation probe
# ---------------------------------------------------------------------------

def run_l1_probe(samples: list[dict]) -> dict[str, tuple[float, float]]:
    """Return {id: (p_vuln_for_code, p_vuln_for_fix)} for all samples."""
    from probe.universal_streaming_probe import _encoder
    _encoder.load()

    with open(LAYER_PROBE_WEIGHTS, "rb") as f:
        payload = pickle.load(f)
    clf = payload["clf"]
    scaler = payload["scaler"]
    layer = payload.get("best_layer", 7)

    results = {}
    n = len(samples)
    for i, s in enumerate(samples):
        if (i + 1) % 20 == 0:
            print(f"  L1: {i+1}/{n}")
        v_vec = _encoder.encode(s["vuln_code"], layer=layer)
        s_vec = _encoder.encode(s["safe_code"], layer=layer)
        p_vuln = float(clf.predict_proba(scaler.transform(v_vec.reshape(1, -1)))[0, 1])
        p_safe = float(clf.predict_proba(scaler.transform(s_vec.reshape(1, -1)))[0, 1])
        results[s["id"]] = (p_vuln, p_safe)
    return results


# ---------------------------------------------------------------------------
# L2: Semantic RAG
# ---------------------------------------------------------------------------

def run_l2_rag(samples: list[dict]) -> dict[str, tuple[bool, bool]]:
    """Return {id: (matched_vuln, matched_safe)}."""
    try:
        from rag.semantic_rag import SecurityRAG
        rag = SecurityRAG()
    except Exception as exc:
        print(f"  L2: ChromaDB unavailable ({exc}), skipping")
        return {s["id"]: (False, False) for s in samples}

    results = {}
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
# L3: Bandit (load from cached results or re-run)
# ---------------------------------------------------------------------------

def run_l3_bandit(samples: list[dict]) -> dict[str, tuple[bool, bool]]:
    """Return {id: (flagged_vuln, flagged_safe)} from bandit."""
    # Try loading cached results first
    cache = Path(BANDIT_RESULTS)
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            raw = json.load(f)
        bandit_raw = raw.get("bandit", [])
        # Build lookup: id -> flagged
        vuln_map: dict[str, bool] = {}
        safe_map: dict[str, bool] = {}
        for r in bandit_raw:
            sid = r["id"]
            if sid.endswith("_safe"):
                safe_map[sid[:-5]] = r["flagged"]
            else:
                vuln_map[sid] = r["flagged"]
        results = {}
        for s in samples:
            results[s["id"]] = (
                vuln_map.get(s["id"], False),
                safe_map.get(s["id"], False),
            )
        return results

    # Re-run if no cache
    print("  L3: No bandit cache, running now...")
    results = {}
    for s in samples:
        def _bandit(code: str) -> bool:
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w",
                                             delete=False, encoding="utf-8") as f:
                f.write(code)
                tmp = f.name
            try:
                r = subprocess.run(["bandit", "-q", "-f", "json", tmp],
                                   capture_output=True, text=True, timeout=30)
                data = json.loads(r.stdout) if r.stdout else {"results": []}
                return any(
                    r.get("issue_severity") in ("MEDIUM", "HIGH")
                    for r in data.get("results", [])
                )
            except Exception:
                return False
            finally:
                Path(tmp).unlink(missing_ok=True)
        results[s["id"]] = (_bandit(s["vuln_code"]), _bandit(s["safe_code"]))
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(flagged_vuln: list[bool], flagged_safe: list[bool]) -> dict:
    tp = sum(flagged_vuln)
    fn = len(flagged_vuln) - tp
    fp = sum(flagged_safe)
    tn = len(flagged_safe) - fp
    n = len(flagged_vuln)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n": n}


def per_class_recall(
    samples: list[dict],
    flagged_vuln: list[bool],
) -> dict[str, dict]:
    classes: dict[str, list[bool]] = {}
    for s, f in zip(samples, flagged_vuln):
        classes.setdefault(s["vuln_class"], []).append(f)
    return {
        vc: {"recall": round(sum(flags)/len(flags), 3), "n": len(flags), "tp": sum(flags)}
        for vc, flags in sorted(classes.items())
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading redbench samples...")
    samples = load_samples()
    n = len(samples)
    print(f"  {n} samples ({n} vulnerable + {n} safe = {2*n} snippets)\n")

    print("[L1] Running CodeBERT activation probe...")
    l1 = run_l1_probe(samples)
    L1_THRESHOLD = 0.55
    l1_vuln = [l1[s["id"]][0] >= L1_THRESHOLD for s in samples]
    l1_safe = [l1[s["id"]][1] >= L1_THRESHOLD for s in samples]

    print("\n[L2] Running semantic RAG (ChromaDB)...")
    l2 = run_l2_rag(samples)
    l2_vuln = [l2[s["id"]][0] for s in samples]
    l2_safe = [l2[s["id"]][1] for s in samples]

    print("\n[L3] Loading Bandit results...")
    l3 = run_l3_bandit(samples)
    l3_vuln = [l3[s["id"]][0] for s in samples]
    l3_safe = [l3[s["id"]][1] for s in samples]

    # ------------------------------------------------------------------
    # Individual layers
    # ------------------------------------------------------------------
    configs = {
        "L1 only  (activation probe)  ": (l1_vuln, l1_safe),
        "L2 only  (semantic RAG)       ": (l2_vuln, l2_safe),
        "L3 only  (Bandit MEDIUM+)     ": (l3_vuln, l3_safe),
        # Ensembles
        "OR   L1|L2|L3                 ": (
            [a or b or c  for a,b,c in zip(l1_vuln,l2_vuln,l3_vuln)],
            [a or b or c  for a,b,c in zip(l1_safe,l2_safe,l3_safe)],
        ),
        "AND  L1&L2&L3                 ": (
            [a and b and c for a,b,c in zip(l1_vuln,l2_vuln,l3_vuln)],
            [a and b and c for a,b,c in zip(l1_safe,l2_safe,l3_safe)],
        ),
        "VOTE 2+/3 agree               ": (
            [(a+b+c)>=2    for a,b,c in zip(l1_vuln,l2_vuln,l3_vuln)],
            [(a+b+c)>=2    for a,b,c in zip(l1_safe,l2_safe,l3_safe)],
        ),
        "L1_GATE L1 AND (L2 OR L3)     ": (
            [a and (b or c) for a,b,c in zip(l1_vuln,l2_vuln,l3_vuln)],
            [a and (b or c) for a,b,c in zip(l1_safe,l2_safe,l3_safe)],
        ),
        "L3_GATE L3 OR (L1 AND L2)     ": (
            [c or (a and b) for a,b,c in zip(l1_vuln,l2_vuln,l3_vuln)],
            [c or (a and b) for a,b,c in zip(l1_safe,l2_safe,l3_safe)],
        ),
    }

    print(f"\n{'='*72}")
    print(f"  Combined Layer Detection Rate — {n} redbench samples")
    print(f"{'='*72}")
    print(f"  {'Method':<38} {'Prec':<7} {'Recall':<8} {'F1':<7} {'TP/FP'}")
    print(f"  {'-'*68}")
    best_f1, best_name = 0.0, ""
    for name, (v_flags, s_flags) in configs.items():
        m = metrics(v_flags, s_flags)
        marker = ""
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_name = name
            marker = " *"
        print(f"  {name} {m['precision']:<7.3f} {m['recall']:<8.3f} {m['f1']:<7.3f} "
              f"{m['tp']}/{m['fp']}{marker}")

    print(f"\n  Best: {best_name.strip()} (F1={best_f1:.3f})")

    # ------------------------------------------------------------------
    # Per-class recall for best ensemble vs individual layers
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"  Per-class recall: L1 vs Bandit vs OR-ensemble")
    print(f"{'='*72}")
    print(f"  {'Class':<22} {'L1':<8} {'Bandit':<8} {'L1|L2|L3':<8} {'n'}")
    print(f"  {'-'*50}")
    or_vuln = configs["OR   L1|L2|L3                 "][0]
    pc_l1     = per_class_recall(samples, l1_vuln)
    pc_bandit = per_class_recall(samples, l3_vuln)
    pc_or     = per_class_recall(samples, or_vuln)
    for vc in sorted(pc_l1):
        print(f"  {vc:<22} {pc_l1[vc]['recall']:<8.2f} "
              f"{pc_bandit[vc]['recall']:<8.2f} "
              f"{pc_or[vc]['recall']:<8.2f} "
              f"{pc_l1[vc]['n']}")

    # ------------------------------------------------------------------
    # Key finding summary
    # ------------------------------------------------------------------
    m_or   = metrics(configs["OR   L1|L2|L3                 "][0],
                     configs["OR   L1|L2|L3                 "][1])
    m_l1   = metrics(l1_vuln, l1_safe)
    m_ban  = metrics(l3_vuln, l3_safe)
    print(f"\n  Recall lift:  L1={m_l1['recall']:.3f}  Bandit={m_ban['recall']:.3f}  "
          f"OR-ensemble={m_or['recall']:.3f}  "
          f"(+{m_or['recall']-max(m_l1['recall'],m_ban['recall']):.3f} vs best single layer)")
    print(f"  Precision:    OR={m_or['precision']:.3f}  "
          f"(trade-off: {m_or['fp']} extra FP from combining)")


if __name__ == "__main__":
    main()
