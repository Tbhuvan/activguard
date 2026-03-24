"""
Fix Vuln Samples — Safe Counterpart Generator.

For every redbench sample that has a vulnerable code block but no safe fix,
calls Ollama to generate a secure counterpart and writes it back to the JSONL.
This is faster than generating fresh matched pairs from scratch because:
  - No meta-model prompt generation step (saves ~60-90s per class)
  - Produces semantically matched pairs (same function, safe implementation)
  - Works directly on the existing 71 GHSA samples lacking fix code

Usage:
    cd activguard/
    python scripts/fix_vuln_samples.py --dry-run        # preview only
    python scripts/fix_vuln_samples.py                   # write fixes
    python scripts/fix_vuln_samples.py --source-file samples_generated_v3.jsonl

After running, retrain with:
    python scripts/train_hf_probe.py --layer 12

Research context:
    Addresses the training data imbalance (vuln=N, safe=0) that causes
    100% false-positive rate.  Matched pairs are theoretically superior
    to independently generated safe code because the probe must learn the
    *difference* between two similar representations, not arbitrary safe code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REDBENCH_DIR = Path("../redbench/datasets")
OLLAMA_BASE  = "http://localhost:11434"
FIX_MODEL    = "qwen2.5:3b"   # safety-aware, generates secure fixes
OUTPUT_FILE  = "samples_fixed.jsonl"

_SYSTEM = (
    "You are a security-conscious Python developer. "
    "You will be given vulnerable Python code. Rewrite it to fix the security issue. "
    "Return ONLY the fixed Python code — no markdown fences, no explanation, no comments about what you changed."
)


def _ollama_fix(vuln_code: str, cwe: str, timeout: int = 180, fix_model: str = FIX_MODEL) -> str:
    """Call Ollama to generate a secure fix for a vulnerable code snippet.

    Args:
        vuln_code: The vulnerable Python code to fix.
        cwe: CWE identifier for context (e.g. 'CWE-089').
        timeout: Request timeout in seconds.

    Returns:
        Fixed code string, or empty string on failure.
    """
    prompt = (
        f"Fix the following vulnerable Python code ({cwe}).\n"
        f"Return ONLY the corrected code, nothing else.\n\n"
        f"```python\n{vuln_code}\n```"
    )
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": fix_model,
                "prompt": prompt,
                "system": _SYSTEM,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 512},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        # Strip markdown fences if model included them
        if raw.startswith("```"):
            lines = raw.splitlines()
            inner = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(inner).strip()
        return raw
    except requests.RequestException as exc:
        logger.warning("Ollama fix failed: %s", exc)
        return ""


def _md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def fix_class(
    class_dir: Path,
    source_file: str,
    dry_run: bool = False,
    fix_model: str = FIX_MODEL,
) -> dict[str, int]:
    """Generate safe fixes for all unfixed samples in a class directory.

    Args:
        class_dir: Path to a redbench class directory (e.g. datasets/sqli/).
        source_file: JSONL filename to read from.
        dry_run: If True, log what would be written without writing.

    Returns:
        Dict with counts: total, already_fixed, new_fixed, failed.
    """
    src = class_dir / source_file
    if not src.exists():
        return {"total": 0, "already_fixed": 0, "new_fixed": 0, "failed": 0}

    out_path = class_dir / OUTPUT_FILE

    # Load existing fixes to avoid duplicates
    existing_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                existing_ids.add(obj.get("id", ""))
            except json.JSONDecodeError:
                pass

    counts = {"total": 0, "already_fixed": 0, "new_fixed": 0, "failed": 0}
    new_entries: list[dict[str, Any]] = []

    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            continue

        code = sample.get("code", "").strip()
        fix  = sample.get("fix", "").strip()
        cwe  = sample.get("cwe", "CWE-???")
        sid  = sample.get("id", _md5(code))

        counts["total"] += 1

        # Skip if already has a real fix
        if fix and not fix.startswith("# TODO"):
            counts["already_fixed"] += 1
            continue

        # Skip if we already generated a fix for this sample
        fixed_id = f"fix_{sid}"
        if fixed_id in existing_ids:
            counts["already_fixed"] += 1
            continue

        if not code:
            counts["failed"] += 1
            continue

        if dry_run:
            logger.info("  [DRY RUN] would fix %s (%s)", sid[:12], cwe)
            counts["new_fixed"] += 1
            continue

        logger.info("  Fixing %s (%s) ...", sid[:12], cwe)
        t0 = time.time()
        fixed_code = _ollama_fix(code, cwe, fix_model=fix_model)
        elapsed = time.time() - t0

        if not fixed_code or len(fixed_code) < 20:
            logger.warning("    Fix too short or empty (%.1fs)", elapsed)
            counts["failed"] += 1
            continue

        entry: dict[str, Any] = {
            "id":         fixed_id,
            "cwe":        cwe,
            "vuln_class": sample.get("vuln_class", class_dir.name),
            "severity":   "none",
            "label":      "safe",
            "language":   "python",
            "code":       fixed_code,
            "fix":        fixed_code,   # fix == code for safe samples
            "description": sample.get("description", ""),
            "source":     f"ollama:{FIX_MODEL}:fix",
            "original_id": sid,
        }
        new_entries.append(entry)
        existing_ids.add(fixed_id)
        counts["new_fixed"] += 1
        logger.info("    Fixed in %.1fs (%d chars)", elapsed, len(fixed_code))

    if new_entries and not dry_run:
        with open(out_path, "a", encoding="utf-8") as fh:
            for e in new_entries:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        logger.info("  Wrote %d new fixes → %s", len(new_entries), out_path)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate safe fixes for unfixed redbench vuln samples via Ollama",
    )
    parser.add_argument(
        "--redbench-dir", default=str(REDBENCH_DIR),
        help="Path to redbench/datasets/ directory",
    )
    parser.add_argument(
        "--source-file", default=None,
        help="Only process this JSONL file per class (default: all files without fixes)",
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Limit to specific vulnerability classes",
    )
    parser.add_argument(
        "--model", default=FIX_MODEL,
        help=f"Ollama model to use for fixes (default: {FIX_MODEL})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be written without calling Ollama",
    )
    parser.add_argument(
        "--audit", action="store_true",
        help="Show count of samples needing fixes per class and exit",
    )
    args = parser.parse_args()

    fix_model = args.model
    base = Path(args.redbench_dir)
    if not base.exists():
        logger.error("redbench dir not found: %s", base)
        sys.exit(1)

    # Check Ollama
    if not args.dry_run and not args.audit:
        try:
            requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5).raise_for_status()
        except requests.RequestException as exc:
            logger.error("Ollama not reachable at %s: %s", OLLAMA_BASE, exc)
            sys.exit(1)

    # Collect source files to process
    SOURCE_FILES = [
        "samples.jsonl",
        "samples_generated.jsonl",
        "samples_generated_v2.jsonl",
        "samples_generated_v3.jsonl",
        "samples_real.jsonl",
    ]
    if args.source_file:
        SOURCE_FILES = [args.source_file]

    class_dirs = sorted(
        d for d in base.iterdir()
        if d.is_dir() and (args.classes is None or d.name in args.classes)
    )

    grand_total    = 0
    grand_fixed    = 0
    grand_skipped  = 0
    grand_failed   = 0

    for class_dir in class_dirs:
        class_counts = {"total": 0, "already_fixed": 0, "new_fixed": 0, "failed": 0}
        for src_file in SOURCE_FILES:
            c = fix_class(class_dir, src_file, dry_run=(args.dry_run or args.audit), fix_model=fix_model)
            for k in class_counts:
                class_counts[k] += c[k]

        if class_counts["total"] == 0:
            continue

        needs_fix = class_counts["total"] - class_counts["already_fixed"]
        if args.audit:
            logger.info(
                "%-22s  total=%-3d  has_fix=%-3d  needs_fix=%-3d",
                class_dir.name,
                class_counts["total"],
                class_counts["already_fixed"],
                needs_fix,
            )
        else:
            logger.info(
                "%s: new=%d  skipped=%d  failed=%d",
                class_dir.name,
                class_counts["new_fixed"],
                class_counts["already_fixed"],
                class_counts["failed"],
            )

        grand_total   += class_counts["total"]
        grand_fixed   += class_counts["new_fixed"]
        grand_skipped += class_counts["already_fixed"]
        grand_failed  += class_counts["failed"]

    print(f"\n{'=' * 50}")
    print(f"  fix_vuln_samples summary")
    print(f"  Total samples scanned: {grand_total}")
    print(f"  Already had fix:       {grand_skipped}")
    print(f"  New fixes written:     {grand_fixed}")
    print(f"  Failed:                {grand_failed}")
    print(f"{'=' * 50}")
    if grand_fixed > 0 and not args.dry_run:
        print(f"\n  Retrain now:")
        print(f"    python scripts/train_hf_probe.py --layer 12")


if __name__ == "__main__":
    main()
