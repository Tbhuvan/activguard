"""
Live CVE → Anti-Pattern → ChromaDB Seeding Pipeline.

Pulls fresh vulnerability advisories from NVD, OSV, and the GitHub Advisory
Database, converts each into a structured anti-pattern description, and upserts
them into the SecurityRAG ChromaDB collection.

This pipeline implements RQ5 of the ActivGuard research programme:
  "What is the latency from CVE publication to detection-capability update?"

By running this pipeline after new CVEs are published, the L2 RAG layer gains
detection coverage within the time it takes to embed one document (~50ms) —
orders of magnitude faster than Snyk (48–72h) or commercial SAST tools.

Architecture:
    NVD / OSV / GitHub Advisory API
        ↓  pull_fresh_cves()
    ThreatIndicator (raw CVE metadata)
        ↓  indicator_to_antipattern()
    anti-pattern description (natural language + code sketch)
        ↓  SecurityRAG.add_pattern()
    ChromaDB `antipatterns` collection

Usage:
    python -m rag.cve_pipeline                      # pull all sources
    python -m rag.cve_pipeline --source nvd --limit 25
    python -m rag.cve_pipeline --source osv --ecosystem PyPI
    python -m rag.cve_pipeline --source github --limit 50

Reference: NVD REST API 2.0 — https://nvd.nist.gov/developers/vulnerabilities
Reference: OSV.dev API — https://api.osv.dev/v1/query
Reference: GitHub Advisory DB API — api.github.com/advisories
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anti-pattern record
# ---------------------------------------------------------------------------

@dataclass
class AntiPattern:
    """A structured anti-pattern derived from a CVE or security advisory.

    Attributes:
        pattern_id: Unique identifier (e.g. 'nvd-cve-2024-1234').
        cwe:        CWE identifier (e.g. 'CWE-89').
        vuln_class: Redbench class name (e.g. 'sqli').
        source:     Origin system (nvd / osv / github_advisory).
        summary:    One-line description of the vulnerability pattern.
        description: Detailed anti-pattern description for RAG embedding.
        code_sketch: Minimal Python code illustrating the pattern (optional).
        severity:   critical / high / medium / low.
    """

    pattern_id: str
    cwe: str
    vuln_class: str
    source: str
    summary: str
    description: str
    code_sketch: str
    severity: str


# CWE → redbench class
_CWE_CLASS_MAP: dict[str, str] = {
    "CWE-89":  "sqli",
    "CWE-78":  "command_injection",
    "CWE-22":  "path_traversal",
    "CWE-918": "ssrf",
    "CWE-639": "idor",
    "CWE-287": "auth_bypass",
    "CWE-79":  "xss",
    "CWE-502": "deserialization",
    "CWE-601": "ssrf",
    "CWE-94":  "command_injection",
    "CWE-798": "auth_bypass",
    "CWE-306": "auth_bypass",
}

_CODE_SKETCHES: dict[str, str] = {
    "sqli": (
        "# SQL Injection pattern\n"
        "query = f\"SELECT * FROM users WHERE id='{user_input}'\"  # UNSAFE\n"
        "# Safe: cursor.execute('SELECT * FROM users WHERE id=%s', (user_input,))"
    ),
    "command_injection": (
        "# Command Injection pattern\n"
        "os.system(f'ping {host}')          # UNSAFE — shell=True equivalent\n"
        "# Safe: subprocess.run(['ping', host], shell=False)"
    ),
    "path_traversal": (
        "# Path Traversal pattern\n"
        "open(os.path.join('/var/data', user_path))  # UNSAFE — no realpath check\n"
        "# Safe: assert os.path.realpath(full).startswith('/var/data')"
    ),
    "ssrf": (
        "# SSRF pattern\n"
        "requests.get(user_provided_url)    # UNSAFE — no allowlist\n"
        "# Safe: validate scheme and hostname against allowlist"
    ),
    "idor": (
        "# IDOR pattern\n"
        "db.query(f'SELECT * FROM docs WHERE id={doc_id}')  # no ownership check\n"
        "# Safe: AND owner_id = ? with authenticated user's ID"
    ),
    "auth_bypass": (
        "# Auth Bypass pattern\n"
        "if token == expected_token:  # timing-oracle vulnerable\n"
        "# Safe: hmac.compare_digest(token, expected_token)"
    ),
    "xss": (
        "# XSS pattern\n"
        "return f'<div>{user_input}</div>'   # UNSAFE — no escaping\n"
        "# Safe: return f'<div>{html.escape(user_input)}</div>'"
    ),
    "deserialization": (
        "# Deserialization pattern\n"
        "pickle.loads(user_bytes)            # UNSAFE — arbitrary code execution\n"
        "# Safe: use json.loads() or cryptographically sign the payload"
    ),
}


def _normalise_cwe(raw: str) -> str:
    """Normalise raw CWE string to 'CWE-N' format."""
    m = re.search(r"\d+", raw)
    return f"CWE-{m.group()}" if m else ""


def _cwe_to_class(cwe: str) -> str | None:
    return _CWE_CLASS_MAP.get(cwe.upper())


# ---------------------------------------------------------------------------
# Source 1: NVD
# ---------------------------------------------------------------------------

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

def pull_nvd(
    keywords: list[str] | None = None,
    limit: int = 25,
    days_back: int = 90,
) -> list[AntiPattern]:
    """Pull CVEs from NVD filtered by Python-relevant keywords.

    Args:
        keywords: Keyword list (default: Python vulnerability keywords).
        limit: Maximum number of patterns to return.
        days_back: Only CVEs published in the last N days.

    Returns:
        List of AntiPattern objects.
    """
    if keywords is None:
        keywords = [
            "python injection", "python path traversal",
            "python deserialization", "python SSRF", "flask", "django",
        ]

    patterns: list[AntiPattern] = []
    seen_ids: set[str] = set()

    for kw in keywords:
        if len(patterns) >= limit:
            break
        params: dict[str, Any] = {
            "keywordSearch": kw,
            "resultsPerPage": min(20, limit),
            "startIndex": 0,
        }
        try:
            resp = requests.get(NVD_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning("NVD pull failed for keyword=%s: %s", kw, exc)
            time.sleep(1)
            continue

        for item in data.get("vulnerabilities", []):
            cve = item.get("cve", {})
            cve_id: str = cve.get("id", "")
            if cve_id in seen_ids:
                continue
            seen_ids.add(cve_id)

            # Extract CWE
            cwes: list[str] = []
            for wk in cve.get("weaknesses", []):
                for desc in wk.get("description", []):
                    val = desc.get("value", "")
                    if val.startswith("CWE-"):
                        cwes.append(_normalise_cwe(val))

            vuln_class = None
            cwe = ""
            for c in cwes:
                vc = _cwe_to_class(c)
                if vc:
                    cwe = c
                    vuln_class = vc
                    break
            if not vuln_class:
                continue

            # Description text
            descs = cve.get("descriptions", [])
            en_desc = next(
                (d["value"] for d in descs if d.get("lang") == "en"), ""
            )

            # Severity from CVSS
            severity = "medium"
            metrics = cve.get("metrics", {})
            for metric_ver in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                if metric_ver in metrics:
                    base = metrics[metric_ver][0].get("cvssData", {})
                    sev = base.get("baseSeverity", "").upper()
                    severity = sev.lower() if sev else "medium"
                    break

            pattern = AntiPattern(
                pattern_id=f"nvd-{cve_id.lower()}",
                cwe=cwe,
                vuln_class=vuln_class,
                source="nvd",
                summary=f"{cve_id} ({cwe}): {en_desc[:120]}",
                description=(
                    f"Vulnerability: {cve_id}\n"
                    f"CWE: {cwe} — {vuln_class}\n"
                    f"Severity: {severity}\n"
                    f"Description: {en_desc}\n"
                    f"Detection: Flag Python code using unsafe {vuln_class} patterns."
                ),
                code_sketch=_CODE_SKETCHES.get(vuln_class, ""),
                severity=severity,
            )
            patterns.append(pattern)

            if len(patterns) >= limit:
                break

        time.sleep(0.7)   # NVD rate limit: 5 req/30s without API key

    logger.info("NVD: %d anti-patterns extracted", len(patterns))
    return patterns


# ---------------------------------------------------------------------------
# Source 2: OSV
# ---------------------------------------------------------------------------

OSV_URL      = "https://api.osv.dev/v1/query"
OSV_BULK_URL = "https://api.osv.dev/v1/vulns/{osv_id}"

# High-profile PyPI packages with known vulnerability histories
_KNOWN_VULN_PACKAGES: list[str] = [
    "django", "flask", "requests", "pillow", "cryptography",
    "pyyaml", "sqlalchemy", "werkzeug", "paramiko", "urllib3",
    "aiohttp", "fastapi", "celery", "redis", "jinja2",
    "numpy", "scipy", "tensorflow", "torch", "langchain",
]


def pull_osv(
    ecosystem: str = "PyPI",
    packages: list[str] | None = None,
    limit: int = 50,
) -> list[AntiPattern]:
    """Pull vulnerability records from OSV for known PyPI packages.

    The OSV /v1/query endpoint requires a package name — it does not support
    ecosystem-only queries.  We iterate over a curated list of high-profile
    packages with known vulnerability histories.

    Args:
        ecosystem: Package ecosystem (default: 'PyPI').
        packages: Package names to query (default: _KNOWN_VULN_PACKAGES).
        limit: Maximum patterns to return.

    Returns:
        List of AntiPattern objects.
    """
    pkg_list = packages or _KNOWN_VULN_PACKAGES
    patterns: list[AntiPattern] = []
    seen: set[str] = set()

    for pkg in pkg_list:
        if len(patterns) >= limit:
            break
        payload = {"package": {"ecosystem": ecosystem, "name": pkg}}
        try:
            resp = requests.post(OSV_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.debug("OSV query for %s failed: %s", pkg, exc)
            time.sleep(0.3)
            continue

        for vuln in data.get("vulns", []):
            if len(patterns) >= limit:
                break

            osv_id: str = vuln.get("id", "")
            if osv_id in seen:
                continue
            seen.add(osv_id)

            # Extract CWE — OSV stores them in database_specific.cwe_ids
            cwes: list[str] = []
            db_specific = vuln.get("database_specific", {})
            for cid in db_specific.get("cwe_ids", []):
                cwes.append(_normalise_cwe(str(cid)))
            # Fallback: scan aliases and reference URLs
            for alias in vuln.get("aliases", []):
                m = re.search(r"CWE-(\d+)", alias)
                if m:
                    cwes.append(f"CWE-{m.group(1)}")
            for ref in vuln.get("references", []):
                m = re.search(r"CWE-(\d+)", ref.get("url", ""))
                if m:
                    cwes.append(f"CWE-{m.group(1)}")

            vuln_class = None
            cwe = ""
            for c in cwes:
                vc = _cwe_to_class(c)
                if vc:
                    cwe = c
                    vuln_class = vc
                    break
            if not vuln_class:
                continue

            details = vuln.get("details", "") or vuln.get("summary", "")
            summary = vuln.get("summary", details[:120])
            # OSV severity.score is a CVSS vector string like "CVSS:3.1/AV:N/.../7.5"
            # Fall back to database_specific.severity for a plain label
            severity = (db_specific.get("severity", "MEDIUM") or "MEDIUM").lower()
            if severity not in ("critical", "high", "medium", "low"):
                severity = "medium"

            patterns.append(AntiPattern(
                pattern_id=f"osv-{osv_id.lower()}",
                cwe=cwe,
                vuln_class=vuln_class,
                source="osv",
                summary=f"{osv_id} ({cwe}): {summary[:120]}",
                description=(
                    f"Vulnerability: {osv_id}\n"
                    f"CWE: {cwe} — {vuln_class}\n"
                    f"Severity: {severity}\n"
                    f"Details: {details[:400]}\n"
                    f"Detection: Flag Python code using unsafe {vuln_class} patterns."
                ),
                code_sketch=_CODE_SKETCHES.get(vuln_class, ""),
                severity=severity,
            ))

        time.sleep(0.2)

    logger.info("OSV: %d anti-patterns extracted", len(patterns))
    return patterns


# ---------------------------------------------------------------------------
# Source 3: GitHub Advisory Database
# ---------------------------------------------------------------------------

GHSA_URL = "https://api.github.com/advisories"

def pull_github_advisories(
    limit: int = 50,
    github_token: str | None = None,
) -> list[AntiPattern]:
    """Pull Python advisories from GitHub Advisory Database.

    Args:
        limit: Maximum patterns to return.
        github_token: Optional GitHub PAT for higher rate limits.

    Returns:
        List of AntiPattern objects.
    """
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    params = {"ecosystem": "pip", "per_page": min(limit, 100), "page": 1}
    patterns: list[AntiPattern] = []
    seen: set[str] = set()

    while len(patterns) < limit:
        try:
            resp = requests.get(GHSA_URL, params=params, headers=headers, timeout=30)
            if resp.status_code == 403:
                logger.warning("GitHub rate limit reached")
                break
            resp.raise_for_status()
            batch = resp.json()
        except requests.RequestException as exc:
            logger.warning("GitHub Advisory pull failed: %s", exc)
            break

        if not batch:
            break

        for adv in batch:
            ghsa_id: str = adv.get("ghsa_id", "")
            if ghsa_id in seen:
                continue
            seen.add(ghsa_id)

            cwes: list[str] = [
                _normalise_cwe(c.get("cwe_id", ""))
                for c in adv.get("cwes", [])
            ]
            vuln_class = None
            cwe = ""
            for c in cwes:
                vc = _cwe_to_class(c)
                if vc:
                    cwe = c
                    vuln_class = vc
                    break
            if not vuln_class:
                continue

            summary: str = adv.get("summary", "")[:200]
            description: str = adv.get("description", summary)[:600]
            severity = _SEVERITY_MAP.get(
                adv.get("severity", "").upper(), "medium"
            )

            pattern = AntiPattern(
                pattern_id=f"ghsa-{ghsa_id.lower()}",
                cwe=cwe,
                vuln_class=vuln_class,
                source="github_advisory",
                summary=f"{ghsa_id} ({cwe}): {summary}",
                description=(
                    f"Advisory: {ghsa_id}\n"
                    f"CWE: {cwe} — {vuln_class}\n"
                    f"Severity: {severity}\n"
                    f"Details: {description}\n"
                    f"Detection: Flag Python code using unsafe {vuln_class} patterns."
                ),
                code_sketch=_CODE_SKETCHES.get(vuln_class, ""),
                severity=severity,
            )
            patterns.append(pattern)

            if len(patterns) >= limit:
                break

        params["page"] += 1
        time.sleep(0.5)

    logger.info("GitHub Advisory: %d anti-patterns extracted", len(patterns))
    return patterns


_SEVERITY_MAP: dict[str, str] = {
    "CRITICAL": "critical",
    "HIGH":     "high",
    "MODERATE": "medium",
    "LOW":      "low",
}


# ---------------------------------------------------------------------------
# ChromaDB upsert
# ---------------------------------------------------------------------------

def seed_chromadb(
    patterns: list[AntiPattern],
    chroma_path: str = ".activguard/chromadb",
    collection_name: str = "antipatterns",
    dry_run: bool = False,
) -> dict[str, int]:
    """Upsert anti-patterns into ChromaDB for L2 RAG retrieval.

    Each anti-pattern becomes one ChromaDB document.  The embedding is
    computed from (summary + description + code_sketch) so that the
    retrieval picks up both natural-language descriptions and code tokens.

    Args:
        patterns: List of AntiPattern objects to upsert.
        chroma_path: Path to ChromaDB persistence directory.
        collection_name: Name of the ChromaDB collection.
        dry_run: If True, print patterns but do not write to ChromaDB.

    Returns:
        Dict with keys: 'upserted', 'already_present', 'failed'.
    """
    if dry_run:
        for p in patterns:
            print(f"  [DRY RUN] {p.pattern_id}: {p.summary[:80]}")
        return {"upserted": len(patterns), "already_present": 0, "failed": 0}

    try:
        import chromadb as _chroma
    except ImportError:
        logger.error("chromadb not installed — pip install chromadb")
        return {"upserted": 0, "already_present": 0, "failed": len(patterns)}

    client = _chroma.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    stats = {"upserted": 0, "already_present": 0, "failed": 0}

    for p in patterns:
        # Check if already present (by pattern_id as ChromaDB document ID)
        doc_id = hashlib.md5(p.pattern_id.encode()).hexdigest()
        try:
            existing = collection.get(ids=[doc_id])
            if existing["ids"]:
                stats["already_present"] += 1
                continue
        except Exception:
            pass   # get() may raise if collection is empty

        # Build the document text — optimised for semantic similarity search
        doc_text = (
            f"Vulnerability pattern: {p.summary}\n\n"
            f"{p.description}\n\n"
            f"Code example:\n{p.code_sketch}"
        ).strip()

        metadata = {
            "pattern_id": p.pattern_id,
            "cwe":        p.cwe,
            "vuln_class": p.vuln_class,
            "source":     p.source,
            "severity":   p.severity,
            "safe":       "false",   # All anti-patterns are flagged as unsafe
        }

        try:
            collection.upsert(
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[metadata],
            )
            stats["upserted"] += 1
            logger.info("  Upserted: %s (%s)", p.pattern_id, p.vuln_class)
        except Exception as exc:
            logger.warning("  Failed to upsert %s: %s", p.pattern_id, exc)
            stats["failed"] += 1

    logger.info(
        "ChromaDB seed complete: %d upserted, %d already present, %d failed",
        stats["upserted"], stats["already_present"], stats["failed"],
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull live CVEs and seed ChromaDB anti-pattern collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default="nvd,osv,github",
        help="Comma-separated sources: nvd, osv, github (default: all)",
    )
    parser.add_argument("--limit", type=int, default=50,
                        help="Max patterns per source (default: 50)")
    parser.add_argument("--chroma-path", default=".activguard/chromadb",
                        help="ChromaDB persistence path")
    parser.add_argument("--github-token", default=None,
                        help="GitHub PAT for higher advisory rate limits")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print patterns without writing to ChromaDB")
    args = parser.parse_args()

    sources = [s.strip().lower() for s in args.source.split(",")]
    all_patterns: list[AntiPattern] = []

    if "nvd" in sources:
        all_patterns.extend(pull_nvd(limit=args.limit))

    if "osv" in sources:
        all_patterns.extend(pull_osv(limit=args.limit))

    if "github" in sources:
        all_patterns.extend(
            pull_github_advisories(
                limit=args.limit,
                github_token=args.github_token,
            )
        )

    if not all_patterns:
        logger.error("No patterns collected — check network access.")
        sys.exit(1)

    logger.info("\nTotal patterns collected: %d", len(all_patterns))
    stats = seed_chromadb(
        all_patterns,
        chroma_path=args.chroma_path,
        dry_run=args.dry_run,
    )
    logger.info("Seed stats: %s", stats)


if __name__ == "__main__":
    main()
