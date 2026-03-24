"""
Real Vulnerability Dataset Aggregator for ActivGuard.

Pulls from three public sources and writes to redbench JSONL format:

  1. SecurityEval (SOAP 2022)
     121 LLM-generated insecure code samples across 68 CWE types.
     Source: github.com/s2e-lab/SecurityEval
     Reference: Siddiq & Santos, "SecurityEval Dataset", SOAP 2022, arXiv:2208.09583.

  2. GitHub Advisory Database (live API)
     Filterable by ecosystem=pip, returns advisories with CWE IDs and summaries.
     We generate Python anti-pattern stubs from the advisory text.
     Source: api.github.com/advisories?ecosystem=pip

  3. CVEfixes (Perl et al., arXiv:2107.08760)
     Automated dataset of CVE patches from GitHub.  We pull the metadata and
     attempt to download the diff for Python files.

Output format (per line in each *.jsonl):
    {
        "id":          "seceval-089-001",
        "cwe":         "CWE-089",
        "severity":    "high",
        "label":       "vulnerable",
        "language":    "python",
        "code":        "<vulnerable code>",
        "description": "<why it is vulnerable>",
        "fix":         "<safe version or placeholder>",
        "attack_scenario": "<how to exploit>",
        "source":      "securityeval|github_advisory|cvefixes"
    }

Usage:
    python scripts/pull_real_data.py [--out-dir ../redbench/datasets]
                                     [--sources securityeval,github_advisory]
                                     [--limit 50]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# CWE → redbench vuln_class mapping
_CWE_MAP: dict[str, str] = {
    "CWE-089": "sqli",
    "CWE-078": "command_injection",
    "CWE-022": "path_traversal",
    "CWE-918": "ssrf",
    "CWE-639": "idor",
    "CWE-287": "auth_bypass",
    "CWE-079": "xss",
    "CWE-502": "deserialization",
    "CWE-601": "ssrf",         # Open Redirect → SSRF class
    "CWE-094": "command_injection",  # Code injection → command_injection class
    "CWE-798": "auth_bypass",  # Hard-coded creds
    "CWE-306": "auth_bypass",  # Missing auth
}

_SEVERITY_MAP: dict[str, str] = {
    "CRITICAL": "critical",
    "HIGH": "high",
    "MODERATE": "medium",
    "LOW": "low",
}


def _cwe_to_class(cwe_id: str) -> str | None:
    """Map a CWE ID string to a redbench vuln_class or None if unknown."""
    key = cwe_id.upper().replace(" ", "")
    return _CWE_MAP.get(key)


def _make_id(source: str, cwe: str, n: int) -> str:
    cwe_slug = cwe.lower().replace("cwe-", "")
    return f"{source}-{cwe_slug}-{n:03d}"


# ---------------------------------------------------------------------------
# Source 1: SecurityEval
# ---------------------------------------------------------------------------

SECURITYEVAL_URL = (
    "https://raw.githubusercontent.com/s2e-lab/SecurityEval/master/dataset.jsonl"
)

def pull_securityeval(limit: int = 200) -> list[dict[str, Any]]:
    """Pull SecurityEval dataset and convert to redbench format.

    SecurityEval provides (prompt, insecure_code) pairs.  We treat the
    insecure code as the vulnerable sample and generate a stub fix comment
    since full safe rewrites are not in the dataset.

    Args:
        limit: Maximum number of samples to return.

    Returns:
        List of redbench-format dicts.
    """
    logger.info("Pulling SecurityEval from GitHub ...")
    try:
        resp = requests.get(SECURITYEVAL_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to fetch SecurityEval: %s", exc)
        return []

    raw_lines = [l for l in resp.text.splitlines() if l.strip()]
    logger.info("  %d raw SecurityEval entries", len(raw_lines))

    samples: list[dict[str, Any]] = []
    counters: dict[str, int] = {}

    for line in raw_lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Extract CWE from ID field: "CWE-089_author_1.py" → "CWE-089"
        raw_id: str = entry.get("ID", "")
        cwe_match = re.match(r"(CWE-\d+)", raw_id)
        if not cwe_match:
            continue
        cwe = cwe_match.group(1)
        vuln_class = _cwe_to_class(cwe)
        if vuln_class is None:
            continue   # CWE not in our taxonomy

        code = entry.get("Insecure_code", "").strip()
        prompt = entry.get("Prompt", "").strip()
        if not code:
            continue

        counters[cwe] = counters.get(cwe, 0) + 1
        sample_id = _make_id("seceval", cwe, counters[cwe])

        # Build a minimal fix stub — SecurityEval only provides insecure code
        fix_stub = (
            "# TODO: Refactor to use parameterised inputs and avoid direct "
            "string interpolation.  See NVD entry for " + cwe + "."
        )

        samples.append({
            "id":          sample_id,
            "cwe":         cwe,
            "severity":    "high",
            "label":       "vulnerable",
            "language":    "python",
            "code":        code,
            "description": prompt,
            "fix":         fix_stub,
            "attack_scenario": f"LLM-generated vulnerable code (SecurityEval prompt: {prompt[:120]})",
            "source":      "securityeval",
        })

        if len(samples) >= limit:
            break

    logger.info("  Extracted %d SecurityEval samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Source 2: GitHub Advisory Database
# ---------------------------------------------------------------------------

GHSA_URL = "https://api.github.com/advisories"

def pull_github_advisories(
    limit: int = 100,
    github_token: str | None = None,
) -> list[dict[str, Any]]:
    """Pull Python security advisories from the GitHub Advisory Database.

    Each advisory contains: summary, description, severity, cwe_ids, references.
    We extract Python code patterns from the description when present, or build
    a stub from the advisory metadata.

    Args:
        limit: Maximum advisories to return.
        github_token: Optional GitHub PAT for higher rate limits.

    Returns:
        List of redbench-format dicts.
    """
    logger.info("Pulling GitHub Advisory Database (ecosystem=pip) ...")
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    params = {
        "ecosystem": "pip",
        "per_page": min(limit, 100),
        "sort": "updated",
        "direction": "desc",
    }

    samples: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    page = 1

    while len(samples) < limit:
        params["page"] = page
        try:
            resp = requests.get(GHSA_URL, params=params, headers=headers, timeout=30)
            if resp.status_code == 403:
                logger.warning("GitHub rate limit hit — stopping advisory pull")
                break
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("GitHub Advisory fetch failed: %s", exc)
            break

        batch = resp.json()
        if not batch:
            break

        for adv in batch:
            cwes: list[str] = [c.get("cwe_id", "") for c in adv.get("cwes", [])]
            vuln_class = None
            cwe = ""
            for c in cwes:
                vuln_class = _cwe_to_class(c)
                if vuln_class:
                    cwe = c
                    break

            if not vuln_class:
                continue  # Skip if not in our taxonomy

            ghsa_id: str = adv.get("ghsa_id", "GHSA-????-????-????")
            summary: str = adv.get("summary", "").strip()
            description: str = adv.get("description", "").strip()
            severity: str = _SEVERITY_MAP.get(
                adv.get("severity", "").upper(), "medium"
            )
            package_names = [
                p.get("package", {}).get("name", "")
                for p in adv.get("vulnerabilities", [])
                if p.get("package", {}).get("ecosystem", "") == "pip"
            ]
            pkg = package_names[0] if package_names else "unknown"

            # Extract any Python code blocks from description
            code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", description)
            code_block = code_blocks[0].strip() if code_blocks else ""

            # If no code block, synthesise a minimal stub from advisory metadata
            if not code_block:
                code_block = _synthesise_vuln_stub(vuln_class, pkg, summary)

            counters[cwe] = counters.get(cwe, 0) + 1
            sample_id = _make_id("ghsa", cwe, counters[cwe])

            samples.append({
                "id":          sample_id,
                "cwe":         cwe,
                "severity":    severity,
                "label":       "vulnerable",
                "language":    "python",
                "code":        code_block,
                "description": f"{ghsa_id}: {summary}",
                "fix":         f"# Upgrade {pkg} to a patched version.  See {ghsa_id}.",
                "attack_scenario": description[:300] if description else summary,
                "source":      f"github_advisory:{ghsa_id}",
            })

            if len(samples) >= limit:
                break

        logger.info("  Page %d: %d samples so far", page, len(samples))
        page += 1
        time.sleep(0.5)   # respectful rate limiting

    logger.info("  Extracted %d GitHub Advisory samples", len(samples))
    return samples


def _synthesise_vuln_stub(vuln_class: str, pkg: str, summary: str) -> str:
    """Generate a minimal vulnerable code stub from advisory metadata.

    This is used when no code block is present in the advisory description.
    The stub illustrates the vulnerability pattern in idiomatic Python.

    Args:
        vuln_class: Redbench class name (e.g. 'sqli').
        pkg: Python package name that has the vulnerability.
        summary: Advisory summary text.

    Returns:
        A minimal Python code string illustrating the vulnerability.
    """
    stubs: dict[str, str] = {
        "sqli": (
            f"# {pkg}: {summary[:80]}\n"
            "import sqlite3\ndef query(conn, user_input):\n"
            "    return conn.execute(f\"SELECT * FROM data WHERE val='{user_input}'\")"
        ),
        "command_injection": (
            f"# {pkg}: {summary[:80]}\n"
            "import subprocess\ndef run(user_cmd):\n"
            "    subprocess.run(user_cmd, shell=True)"
        ),
        "path_traversal": (
            f"# {pkg}: {summary[:80]}\n"
            "import os\ndef read_file(user_path, base='/data'):\n"
            "    return open(os.path.join(base, user_path)).read()"
        ),
        "ssrf": (
            f"# {pkg}: {summary[:80]}\n"
            "import requests\ndef fetch(user_url):\n"
            "    return requests.get(user_url).text"
        ),
        "deserialization": (
            f"# {pkg}: {summary[:80]}\n"
            "import pickle\ndef load_data(user_bytes):\n"
            "    return pickle.loads(user_bytes)"
        ),
        "xss": (
            f"# {pkg}: {summary[:80]}\n"
            "from flask import request, make_response\n"
            "def echo():\n"
            "    val = request.args.get('q', '')\n"
            "    return make_response(f'<div>{val}</div>')"
        ),
        "auth_bypass": (
            f"# {pkg}: {summary[:80]}\n"
            "def check_auth(token, expected):\n"
            "    return token == expected   # timing-attack vulnerable\n"
        ),
        "idor": (
            f"# {pkg}: {summary[:80]}\n"
            "def get_record(user_id, record_id, db):\n"
            "    return db.query(f'SELECT * FROM records WHERE id={record_id}')"
        ),
    }
    return stubs.get(vuln_class, f"# {pkg}: {summary[:80]}\n# No code example available.")


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_to_redbench(
    samples: list[dict[str, Any]],
    out_dir: Path,
    filename: str = "samples_real.jsonl",
) -> dict[str, int]:
    """Write samples to per-class redbench JSONL files.

    Deduplicates by code hash to avoid duplicate entries.

    Args:
        samples: List of redbench-format dicts.
        out_dir: Path to redbench/datasets/ directory.
        filename: Output file name within each class directory.

    Returns:
        Dict mapping vuln_class → count of new samples written.
    """
    # Group by vuln class
    by_class: dict[str, list[dict]] = {}
    for s in samples:
        cwe = s["cwe"]
        vc = _cwe_to_class(cwe)
        if vc:
            by_class.setdefault(vc, []).append(s)

    counts: dict[str, int] = {}
    for vc, entries in by_class.items():
        class_dir = out_dir / vc
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / filename

        # Load existing hashes to deduplicate
        existing_hashes: set[str] = set()
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            ex = json.loads(line)
                            h = hashlib.md5(ex.get("code", "").encode()).hexdigest()
                            existing_hashes.add(h)
                        except json.JSONDecodeError:
                            continue

        new_entries = [
            e for e in entries
            if hashlib.md5(e.get("code", "").encode()).hexdigest() not in existing_hashes
        ]

        if not new_entries:
            logger.info("  %s: 0 new samples (all already present)", vc)
            counts[vc] = 0
            continue

        with open(out_path, "a", encoding="utf-8") as f:
            for e in new_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        logger.info("  %s: wrote %d new samples to %s", vc, len(new_entries), out_path)
        counts[vc] = len(new_entries)

    return counts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull real vulnerability datasets into redbench format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        default="../redbench/datasets",
        help="Path to redbench/datasets/ directory (default: ../redbench/datasets)",
    )
    parser.add_argument(
        "--sources",
        default="securityeval,github_advisory",
        help="Comma-separated sources: securityeval, github_advisory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max samples per source (default: 200)",
    )
    parser.add_argument(
        "--github-token",
        default=None,
        help="GitHub PAT for higher advisory rate limits (optional)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    sources = [s.strip() for s in args.sources.split(",")]

    all_samples: list[dict[str, Any]] = []

    if "securityeval" in sources:
        se_samples = pull_securityeval(limit=args.limit)
        all_samples.extend(se_samples)

    if "github_advisory" in sources:
        gh_samples = pull_github_advisories(
            limit=args.limit,
            github_token=args.github_token,
        )
        all_samples.extend(gh_samples)

    if not all_samples:
        logger.error("No samples collected — check network access and source flags.")
        sys.exit(1)

    logger.info("\nTotal samples collected: %d", len(all_samples))
    counts = write_to_redbench(all_samples, out_dir)

    total_new = sum(counts.values())
    logger.info("\nDone.  %d new samples written across %d classes.", total_new, len(counts))
    logger.info("Re-run scripts/train_hf_probe.py to retrain with the new data.")


if __name__ == "__main__":
    main()
