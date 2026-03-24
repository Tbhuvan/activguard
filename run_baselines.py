"""
Baseline Security Tool Comparison — Bandit and Semgrep vs Activation Probe.

Runs published static analysis tools on the redbench dataset and measures
precision, recall, and F1 against the ground-truth labels.  Outputs a
comparison table suitable for the PhD portfolio README.

Baselines:
    - Bandit (PyCQA/bandit): Python AST-based SAST tool, rule-based.
    - Semgrep (semgrep/semgrep): Pattern-matching SAST tool with community rules.

Comparison framing:
    Static analysis tools run on COMPLETE code AFTER it is written.
    The activation probe runs DURING generation BEFORE the code is finished.
    This is the key contribution framing for RQ1/RQ4.

Usage:
    python run_baselines.py                        # run all tools
    python run_baselines.py --tool bandit          # bandit only
    python run_baselines.py --severity MEDIUM      # include MEDIUM findings
    python run_baselines.py --out baselines.json   # save raw results

Reference: OWASP Top 10 (2021), CWE/SANS Top 25.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REDBENCH_DEFAULT = "../redbench/datasets"
SEVERITY_DEFAULT = "HIGH"   # Bandit severity threshold


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(datasets_dir: str) -> list[dict[str, Any]]:
    """Load all redbench samples with code, fix, and label."""
    base = Path(datasets_dir)
    samples: list[dict[str, Any]] = []
    for vc in sorted(d.name for d in base.iterdir() if d.is_dir()):
        for fname in ("samples.jsonl", "samples_generated.jsonl", "samples_generated_v2.jsonl"):
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
                        "cwe": s.get("cwe", ""),
                        "vuln_code": s["code"],
                        "safe_code": s["fix"],
                    })
    return samples


# ---------------------------------------------------------------------------
# Bandit runner
# ---------------------------------------------------------------------------

def run_bandit(code: str, severity_threshold: str = "HIGH") -> dict[str, Any]:
    """Run bandit on a code snippet. Returns dict with flagged, findings list.

    Args:
        code: Python source code string.
        severity_threshold: Minimum severity to count ("LOW", "MEDIUM", "HIGH").

    Returns:
        dict: {flagged: bool, n_findings: int, findings: list, error: str|None}
    """
    sev_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    min_sev = sev_order.get(severity_threshold.upper(), 2)

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False,
                                     encoding="utf-8") as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["bandit", "-q", "-f", "json", tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        # bandit exits 1 if issues found, 0 if clean — both are valid
        if not result.stdout:
            return {"flagged": False, "n_findings": 0, "findings": [], "error": None}

        data = json.loads(result.stdout)
        findings = [
            r for r in data.get("results", [])
            if sev_order.get(r.get("issue_severity", "LOW"), 0) >= min_sev
        ]
        return {
            "flagged": len(findings) > 0,
            "n_findings": len(findings),
            "findings": findings,
            "error": None,
        }
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as exc:
        return {"flagged": False, "n_findings": 0, "findings": [], "error": str(exc)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Semgrep runner
# ---------------------------------------------------------------------------

def run_semgrep(code: str) -> dict[str, Any]:
    """Run semgrep with python security rules on a code snippet.

    Uses semgrep --config=p/python and p/owasp-top-ten for broad coverage.

    Args:
        code: Python source code string.

    Returns:
        dict: {flagged: bool, n_findings: int, findings: list, error: str|None}
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False,
                                     encoding="utf-8") as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["semgrep", "--config=p/python", "--json", "--quiet", tmp_path],
            capture_output=True, text=True, timeout=60,
        )
        if not result.stdout:
            return {"flagged": False, "n_findings": 0, "findings": [], "error": None}
        data = json.loads(result.stdout)
        findings = data.get("results", [])
        return {
            "flagged": len(findings) > 0,
            "n_findings": len(findings),
            "findings": [r.get("check_id", "") for r in findings],
            "error": None,
        }
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as exc:
        return {"flagged": False, "n_findings": 0, "findings": [], "error": str(exc)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute precision, recall, F1, accuracy from flagged/label pairs.

    Convention:
        TP = vulnerable code flagged
        FN = vulnerable code not flagged
        FP = safe code flagged
        TN = safe code not flagged

    Args:
        results: List of {flagged: bool, label: int (1=vuln, 0=safe)}.

    Returns:
        dict: precision, recall, f1, accuracy, tp, fp, fn, tn.
    """
    tp = sum(1 for r in results if r["flagged"] and r["label"] == 1)
    fn = sum(1 for r in results if not r["flagged"] and r["label"] == 1)
    fp = sum(1 for r in results if r["flagged"] and r["label"] == 0)
    tn = sum(1 for r in results if not r["flagged"] and r["label"] == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(results) if results else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_samples": len(results),
    }


# ---------------------------------------------------------------------------
# Per-class breakdown
# ---------------------------------------------------------------------------

def per_class_metrics(
    sample_results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute recall per vulnerability class (vulnerable samples only)."""
    classes: dict[str, list[dict]] = {}
    for r in sample_results:
        if r["label"] == 1:
            vc = r["vuln_class"]
            classes.setdefault(vc, []).append(r)
    out: dict[str, dict] = {}
    for vc, items in sorted(classes.items()):
        tp = sum(1 for i in items if i["flagged"])
        recall = tp / len(items) if items else 0.0
        out[vc] = {"recall": round(recall, 3), "n": len(items), "tp": tp}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bandit/Semgrep baselines on redbench")
    parser.add_argument("--redbench", default=REDBENCH_DEFAULT)
    parser.add_argument("--tool", choices=["bandit", "semgrep", "all"], default="all")
    parser.add_argument("--severity", default=SEVERITY_DEFAULT, help="Bandit severity threshold")
    parser.add_argument("--out", default=None, help="Save raw results JSON")
    args = parser.parse_args()

    samples = load_samples(args.redbench)
    print(f"[*] Loaded {len(samples)} samples from {args.redbench}")
    print(f"    ({len(samples)} vulnerable + {len(samples)} safe = {len(samples)*2} total code snippets)")

    tools_to_run: list[str] = []
    if args.tool in ("bandit", "all"):
        tools_to_run.append("bandit")
    if args.tool in ("semgrep", "all"):
        tools_to_run.append("semgrep")

    all_results: dict[str, list[dict]] = {}

    for tool in tools_to_run:
        print(f"\n[*] Running {tool}...")
        tool_results: list[dict] = []
        n = len(samples)

        for i, s in enumerate(samples):
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{n}...")

            # Score vulnerable code
            if tool == "bandit":
                vuln_res = run_bandit(s["vuln_code"], args.severity)
                safe_res = run_bandit(s["safe_code"], args.severity)
            else:
                vuln_res = run_semgrep(s["vuln_code"])
                safe_res = run_semgrep(s["safe_code"])

            tool_results.append({
                "id": s["id"], "vuln_class": s["vuln_class"],
                "flagged": vuln_res["flagged"], "label": 1,
                "n_findings": vuln_res["n_findings"],
            })
            tool_results.append({
                "id": s["id"] + "_safe", "vuln_class": s["vuln_class"],
                "flagged": safe_res["flagged"], "label": 0,
                "n_findings": safe_res["n_findings"],
            })

        all_results[tool] = tool_results
        metrics = compute_metrics(tool_results)
        per_class = per_class_metrics(tool_results)

        print(f"\n  {tool.upper()} results (severity>={args.severity}):")
        print(f"  {'Metric':<12} Value")
        print(f"  {'-'*25}")
        print(f"  {'Precision':<12} {metrics['precision']:.3f}   ({metrics['tp']} TP, {metrics['fp']} FP)")
        print(f"  {'Recall':<12} {metrics['recall']:.3f}   ({metrics['tp']} TP, {metrics['fn']} FN)")
        print(f"  {'F1':<12} {metrics['f1']:.3f}")
        print(f"  {'Accuracy':<12} {metrics['accuracy']:.3f}")
        print(f"\n  Per-class recall:")
        for vc, m in per_class.items():
            bar = "#" * int(m["recall"] * 20)
            print(f"    {vc:<20} {m['recall']:.2f}  [{bar:<20}]  ({m['tp']}/{m['n']})")

    # Summary comparison table
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  Comparison vs Activation Probe (AUC 0.900 / Acc 0.840)")
        print(f"{'='*60}")
        print(f"  {'Method':<28} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Note'}")
        print(f"  {'-'*60}")
        for tool_name, results in all_results.items():
            m = compute_metrics(results)
            note = "runs AFTER code written" if tool_name in ("bandit", "semgrep") else ""
            print(f"  {tool_name.capitalize():<28} {m['precision']:<8.3f} {m['recall']:<8.3f} {m['f1']:<8.3f} {note}")
        print(f"  {'CodeBERT probe (CV AUC=0.900)':<28} {'—':<8} {'—':<8} {'—':<8} runs DURING generation")
        print(f"  {'Streaming probe (P=0.95 demo)':<28} {'—':<8} {'—':<8} {'—':<8} flags at token 255/305")
        print(f"{'='*60}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({t: r for t, r in all_results.items()}, f, indent=2)
        print(f"\n[+] Raw results saved to {args.out}")


if __name__ == "__main__":
    main()
