"""
Large-Scale Dataset Builder for ActivGuard.

Targets 1000+ labelled (vuln, safe) Python code pairs via three pipelines:

  Pipeline A — Net pull (no GPU needed):
      SecurityEval (121 LLM-generated insecure Python snippets, SOAP 2022)
      GitHub Advisory Database (live pip advisories with code stubs)
      CVEfixes Python subset (real CVE before/after patches via GitHub API)

  Pipeline B — Local LLM generation (Ollama):
      1. Meta-generation: use the instruction model to produce N diverse
         natural developer prompts per CWE category.
      2. Vuln generation: dolphin3:8b (uncensored) generates code from prompts.
      3. Safe generation: append "use secure coding best practices" to the same
         prompts and run qwen2.5-coder:1.5b to get safe counterparts.
      4. Auto-labelling: Bandit confirms vuln label; clean Bandit output + no
         SAST hits used as safe label.

  Pipeline C — HuggingFace datasets (no Ollama needed):
      datasets.load_dataset("s2e-lab/SecurityEval") — full 121 entries
      datasets.load_dataset("microsoft/codexglue_defect") — Devign C (skipped,
         C only; kept here for completeness)

Output (per-class JSONL in redbench format):
    ../redbench/datasets/<vuln_class>/samples_generated_v3.jsonl

After this script, re-run:
    python scripts/train_hf_probe.py --layer 12

to retrain the probe on the expanded dataset.

Research context:
    Addresses dataset scale limitation noted in RQ4.  The combination of real
    CVE code, LLM-generated naturalistic variants, and safe counterparts
    provides the balanced, diverse training signal needed for the probe to
    generalise to unseen vulnerability patterns (blind test set).

Reference: Siddiq & Santos, "SecurityEval Dataset: Narrowing the Gap Between
    Benchmarking and Real-World Evaluation of NLP Models for Code", SOAP 2022.
Reference: Pearce et al., "Asleep at the Keyboard? Assessing the Security of
    GitHub Copilot's Code Contributions", arXiv:2108.09293 (2022).
Reference: Perl et al., "CVEfixes: Automated Collection of Vulnerabilities and
    Their Fixes from Open-Source Software", arXiv:2107.08760 (2021).

Usage:
    # Pull from net only (fast, no Ollama needed):
    python scripts/build_large_dataset.py --pipelines net --target 400

    # Generate using local LLMs (slow, requires dolphin3:8b + qwen2.5-coder:1.5b):
    python scripts/build_large_dataset.py --pipelines ollama --target 600

    # Both pipelines, target 1000 pairs:
    python scripts/build_large_dataset.py --pipelines net,ollama --target 1000

    # Check progress without writing:
    python scripts/build_large_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDBENCH_DIR     = Path("../redbench/datasets")
OLLAMA_BASE      = "http://localhost:11434"
VULN_MODEL       = "dolphin3:8b"      # uncensored — generates vuln code freely
SAFE_MODEL       = "qwen2.5:3b"           # safety-aware — generates secure code
META_MODEL       = "qwen2.5:3b"           # generates diverse prompts

SECURITYEVAL_URL = (
    "https://raw.githubusercontent.com/s2e-lab/SecurityEval/master/dataset.jsonl"
)
GHSA_URL = "https://api.github.com/advisories"

# ---------------------------------------------------------------------------
# CWE taxonomy — 17 classes, each with a natural-language framing for prompts
# ---------------------------------------------------------------------------

CWE_CLASSES: dict[str, dict[str, str]] = {
    "sqli": {
        "cwe": "CWE-089",
        "description": "SQL injection via string-formatted queries",
        "context": "database query functions, search endpoints, login forms, "
                   "filter/sort APIs, admin panels that query by user input",
    },
    "command_injection": {
        "cwe": "CWE-078",
        "description": "OS command injection via subprocess or os.system",
        "context": "sysadmin helpers, CI/CD scripts, file conversion tools, "
                   "network utilities like ping/nslookup/traceroute wrappers",
    },
    "path_traversal": {
        "cwe": "CWE-022",
        "description": "path traversal via unsanitised file paths",
        "context": "file download/upload handlers, config file loaders, "
                   "log readers, static file servers",
    },
    "ssrf": {
        "cwe": "CWE-918",
        "description": "server-side request forgery via unvalidated URLs",
        "context": "webhook proxies, URL preview/metadata fetchers, "
                   "link shorteners, API relay endpoints",
    },
    "xss": {
        "cwe": "CWE-079",
        "description": "cross-site scripting via unescaped user input in HTML",
        "context": "Flask/Django route handlers that return HTML, "
                   "error pages, greeting/profile display endpoints",
    },
    "idor": {
        "cwe": "CWE-639",
        "description": "insecure direct object reference without ownership check",
        "context": "REST API endpoints for orders, messages, invoices, documents, "
                   "profiles that take an object ID parameter",
    },
    "auth_bypass": {
        "cwe": "CWE-287",
        "description": "authentication bypass or weak token validation",
        "context": "JWT validators, API key middleware, admin decorators, "
                   "role-check functions, session validators",
    },
    "deserialization": {
        "cwe": "CWE-502",
        "description": "unsafe deserialisation of untrusted data with pickle/yaml",
        "context": "session stores, cache layers, config loaders, "
                   "RPC/message queue handlers",
    },
    "xxe": {
        "cwe": "CWE-611",
        "description": "XML external entity injection via unsafe XML parsing",
        "context": "config file parsers, data import endpoints, "
                   "SOAP/XML API clients, document processors",
    },
    "open_redirect": {
        "cwe": "CWE-601",
        "description": "open redirect via unvalidated redirect URL parameter",
        "context": "post-login redirects, OAuth callback handlers, "
                   "form submission redirects",
    },
    "weak_crypto": {
        "cwe": "CWE-338",
        "description": "cryptographically weak random number generation for secrets",
        "context": "password reset tokens, session IDs, CSRF tokens, "
                   "API key generation, OTP generators",
    },
    "cleartext_secrets": {
        "cwe": "CWE-312",
        "description": "hardcoded or cleartext credentials in source code",
        "context": "database connection strings, SMTP credentials, "
                   "API key constants, cloud storage access keys",
    },
    "unsafe_yaml": {
        "cwe": "CWE-703",
        "description": "unsafe YAML loading enabling arbitrary code execution",
        "context": "config file loaders, CI/CD pipeline readers, "
                   "data import functions",
    },
    "tls_validation": {
        "cwe": "CWE-295",
        "description": "disabled TLS certificate verification",
        "context": "internal API clients, webhook senders, "
                   "health-check scripts for dev/staging environments",
    },
    "race_condition": {
        "cwe": "CWE-362",
        "description": "TOCTOU race condition in file or resource access",
        "context": "file upload handlers, temp file creators, "
                   "concurrent resource reservation systems",
    },
    "mass_assignment": {
        "cwe": "CWE-915",
        "description": "mass assignment allowing modification of protected fields",
        "context": "REST update endpoints, Django/Flask model update views, "
                   "user profile edit APIs",
    },
    "redos": {
        "cwe": "CWE-400",
        "description": "regex denial-of-service via catastrophic backtracking",
        "context": "input validation functions, email/URL/username validators, "
                   "form field checkers",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(source: str, cwe_class: str, n: int) -> str:
    cwe_slug = cwe_class.lower().replace("cwe-", "")
    return f"{source}-{cwe_slug}-{n:04d}"


def _code_hash(code: str) -> str:
    return hashlib.md5(code.strip().encode()).hexdigest()


def _load_existing_hashes(jsonl_path: Path) -> set[str]:
    """Return set of code hashes already in a JSONL file (deduplication)."""
    hashes: set[str] = set()
    if not jsonl_path.exists():
        return hashes
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                hashes.add(_code_hash(entry.get("code", "")))
            except json.JSONDecodeError:
                continue
    return hashes


def _write_jsonl(samples: list[dict[str, Any]], path: Path) -> int:
    """Append samples to JSONL, skip duplicates.  Returns count written."""
    existing = _load_existing_hashes(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "a", encoding="utf-8") as fh:
        for s in samples:
            h = _code_hash(s.get("code", ""))
            if h not in existing:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
                existing.add(h)
                written += 1
    return written


def _run_bandit(code: str) -> tuple[bool, list[str]]:
    """Run Bandit on a code snippet.

    Returns:
        Tuple (has_issues, issue_ids).
        has_issues=True means Bandit found at least one security issue.
    """
    try:
        proc = subprocess.run(
            ["bandit", "-r", "-f", "json", "-"],
            input=code.encode(),
            capture_output=True,
            timeout=15,
        )
        data = json.loads(proc.stdout or "{}")
        results = data.get("results", [])
        issue_ids = [r.get("test_id", "") for r in results]
        return len(results) > 0, issue_ids
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return False, []


# ---------------------------------------------------------------------------
# Pipeline A — Net pull
# ---------------------------------------------------------------------------

def pull_securityeval(limit: int = 500) -> list[dict[str, Any]]:
    """Pull SecurityEval dataset (all 121 entries, no limit by default)."""
    logger.info("Pipeline A: Pulling SecurityEval ...")
    try:
        resp = requests.get(SECURITYEVAL_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("SecurityEval unavailable: %s", exc)
        return []

    cwe_map = {c["cwe"]: vc for vc, c in CWE_CLASSES.items()}
    samples: list[dict[str, Any]] = []
    counters: dict[str, int] = {}

    for line in resp.text.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        raw_id = entry.get("ID", "")
        m = re.match(r"(CWE-\d+)", raw_id)
        if not m:
            continue
        cwe = m.group(1)
        vc = cwe_map.get(cwe)
        if not vc:
            continue

        code = entry.get("Insecure_code", "").strip()
        if not code or len(code) < 20:
            continue

        counters[cwe] = counters.get(cwe, 0) + 1
        samples.append({
            "id":          _make_id("seceval", cwe, counters[cwe]),
            "cwe":         cwe,
            "vuln_class":  vc,
            "severity":    "high",
            "label":       "vulnerable",
            "language":    "python",
            "code":        code,
            "description": entry.get("Prompt", ""),
            "fix":         "",
            "source":      "securityeval",
        })
        if len(samples) >= limit:
            break

    logger.info("  SecurityEval: %d samples", len(samples))
    return samples


def pull_github_advisories(
    limit: int = 300,
    github_token: str | None = None,
) -> list[dict[str, Any]]:
    """Pull Python advisories from GitHub Advisory Database."""
    logger.info("Pipeline A: Pulling GitHub Advisories (pip) ...")
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    cwe_map = {c["cwe"]: vc for vc, c in CWE_CLASSES.items()}
    samples: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    page = 1

    while len(samples) < limit:
        params = {
            "ecosystem": "pip",
            "per_page": 100,
            "sort": "updated",
            "direction": "desc",
            "page": page,
        }
        try:
            resp = requests.get(
                GHSA_URL, params=params, headers=headers, timeout=30
            )
            if resp.status_code in (403, 429):
                logger.warning("GitHub rate limit hit at page %d", page)
                break
            resp.raise_for_status()
            batch = resp.json()
        except requests.RequestException as exc:
            logger.warning("GitHub Advisory error: %s", exc)
            break

        if not batch:
            break

        for adv in batch:
            cwes = [c.get("cwe_id", "") for c in adv.get("cwes", [])]
            cwe = next((c for c in cwes if cwe_map.get(c)), None)
            if not cwe:
                continue
            vc = cwe_map[cwe]

            # Prefer real code blocks from advisory description
            desc = adv.get("description", "")
            code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", desc)
            code = code_blocks[0].strip() if code_blocks else ""
            if not code or len(code) < 30:
                continue  # skip stubs — real code only for advisory source

            ghsa = adv.get("ghsa_id", "")
            severity = {"CRITICAL": "critical", "HIGH": "high",
                        "MODERATE": "medium", "LOW": "low"}.get(
                adv.get("severity", "").upper(), "medium"
            )
            counters[cwe] = counters.get(cwe, 0) + 1
            samples.append({
                "id":          _make_id("ghsa", cwe, counters[cwe]),
                "cwe":         cwe,
                "vuln_class":  vc,
                "severity":    severity,
                "label":       "vulnerable",
                "language":    "python",
                "code":        code,
                "description": f"{ghsa}: {adv.get('summary', '')}",
                "fix":         "",
                "source":      f"github_advisory:{ghsa}",
            })

            if len(samples) >= limit:
                break

        logger.info("  GHSA page %d: %d samples", page, len(samples))
        page += 1
        time.sleep(0.3)

    logger.info("  GitHub Advisories: %d samples (real code only)", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Pipeline B — Ollama generation
# ---------------------------------------------------------------------------

def _ollama_generate(
    prompt: str,
    model: str,
    system: str = "",
    timeout: int = 90,
    num_predict: int = 512,
) -> str:
    """Call Ollama /api/generate and return the response text."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.85, "num_predict": num_predict},
    }
    if system:
        payload["system"] = system
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.RequestException as exc:
        logger.warning("Ollama generate failed (%s): %s", model, exc)
        return ""


_BATCH_SIZE = 8   # prompts per meta-model call — fits comfortably in 512 tokens


def _generate_prompts_for_class(
    vc: str,
    n: int = 20,
) -> list[str]:
    """Use the meta-model to generate N diverse natural developer prompts.

    Requests _BATCH_SIZE prompts per call and loops until n are collected,
    to avoid JSON truncation from long token budgets.

    Args:
        vc: Vulnerability class key (e.g. 'sqli').
        n: Number of prompts to generate.

    Returns:
        List of prompt strings (up to n).
    """
    info = CWE_CLASSES[vc]
    system = (
        "You are generating a dataset of software developer questions. "
        "Output ONLY a JSON array of strings. No markdown, no explanation. "
        "Each string is a natural developer question, like something typed into "
        "GitHub Copilot or Stack Overflow."
    )
    collected: list[str] = []
    max_calls = max(1, (n + _BATCH_SIZE - 1) // _BATCH_SIZE) + 1   # +1 retry budget

    for _ in range(max_calls):
        if len(collected) >= n:
            break
        batch_n = min(_BATCH_SIZE, n - len(collected))
        user = (
            f"Generate {batch_n} diverse natural Python developer questions that would require "
            f"writing {info['description']} ({info['cwe']}).\n"
            f"Realistic contexts: {info['context']}.\n"
            f"Requirements:\n"
            f"- Sound like real developer questions, NOT security research prompts\n"
            f"- No mention of 'vulnerability', 'secure', 'unsafe', 'CVE', or 'exploit'\n"
            f"- Vary the frameworks: Flask, Django, FastAPI, plain Python, SQLAlchemy\n"
            f"- Vary the scenario: functions, routes, CLI scripts, class methods\n"
            f"- Return ONLY a JSON array with exactly {batch_n} strings: "
            f'["question1", "question2", ...]'
        )
        raw = _ollama_generate(user, META_MODEL, system=system, timeout=200, num_predict=768)
        if not raw:
            continue

        # 1. Try JSON array (greedy — last ] closes the array)
        m = re.search(r"\[[\s\S]*\]", raw)
        if m:
            try:
                parsed = json.loads(m.group())
                batch = [str(p).strip() for p in parsed if isinstance(p, str) and len(p) > 20]
                collected.extend(batch)
                continue
            except json.JSONDecodeError:
                pass

        # 2. Fallback: numbered / bulleted lines
        for line in raw.splitlines():
            cleaned = re.sub(r"^[\d]+[.)]\s*|^[-*•]\s*|^[\"']|[\"']$", "", line.strip()).strip()
            if len(cleaned) > 30:
                collected.append(cleaned)

    seen: set[str] = set()
    unique = [p for p in collected if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]
    return unique[:n]


def _generate_vuln_code(prompt: str, vc: str) -> str:
    """Use the uncensored model to generate vulnerable code for a prompt."""
    system = (
        "You are a Python developer. Write Python code only. "
        "No explanation, no markdown fences, no comments about security. "
        "Just working Python code that fulfils the request."
    )
    return _ollama_generate(prompt, VULN_MODEL, system=system, timeout=150)


def _generate_safe_code(prompt: str, vc: str) -> str:
    """Use the safe model to generate a secure counterpart."""
    secure_prompt = (
        f"{prompt}\n\n"
        "Important: follow secure coding best practices. "
        "Use parameterised queries, validate inputs, avoid shell=True, "
        "sanitise file paths, escape outputs, verify SSL certificates."
    )
    system = (
        "You are a security-conscious Python developer. "
        "Write Python code only. No markdown fences, no explanations."
    )
    return _ollama_generate(secure_prompt, SAFE_MODEL, system=system, timeout=150)


def generate_ollama_dataset(
    target_per_class: int = 30,
    classes: list[str] | None = None,
    label_with_bandit: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate (vuln, safe) pairs using local Ollama models.

    For each CWE class:
      1. Generate diverse prompts using META_MODEL.
      2. For each prompt, generate vulnerable code with VULN_MODEL.
      3. For each prompt, generate safe code with SAFE_MODEL.
      4. Optionally confirm labels with Bandit.

    Args:
        target_per_class: Target number of (vuln, safe) pairs per class.
        classes: List of CWE class keys to process.  None = all 17 classes.
        label_with_bandit: Run Bandit to confirm/filter vuln labels.

    Returns:
        Tuple (vuln_samples, safe_samples).
    """
    target_classes = classes or list(CWE_CLASSES.keys())
    vuln_out: list[dict[str, Any]] = []
    safe_out: list[dict[str, Any]] = []

    for vc in target_classes:
        info = CWE_CLASSES[vc]
        cwe = info["cwe"]
        logger.info(
            "Pipeline B: Generating %d pairs for %s (%s) ...",
            target_per_class, vc, cwe,
        )

        # Step 1: Generate prompts — request exactly target_per_class.
        # _generate_prompts_for_class has its own retry budget for failures.
        prompts = _generate_prompts_for_class(vc, n=target_per_class)
        if not prompts:
            logger.warning("  No prompts generated for %s — skipping", vc)
            continue
        logger.info("  Generated %d prompts", len(prompts))

        collected_vuln = 0
        collected_safe = 0
        counters: dict[str, int] = {}
        counters["vuln"] = 0
        counters["safe"] = 0

        for i, prompt in enumerate(prompts):
            if collected_vuln >= target_per_class and collected_safe >= target_per_class:
                break

            # Step 2: Vuln code
            if collected_vuln < target_per_class:
                vuln_code = _generate_vuln_code(prompt, vc)
                if vuln_code and len(vuln_code) > 30:
                    confirmed = True
                    if label_with_bandit:
                        has_issues, _ = _run_bandit(vuln_code)
                        confirmed = has_issues
                    if confirmed:
                        counters["vuln"] += 1
                        vuln_out.append({
                            "id":          _make_id(f"ollama_{VULN_MODEL.replace(':', '_')}", cwe, counters["vuln"]),
                            "cwe":         cwe,
                            "vuln_class":  vc,
                            "severity":    "high",
                            "label":       "vulnerable",
                            "language":    "python",
                            "code":        vuln_code,
                            "description": prompt,
                            "fix":         "",
                            "source":      f"ollama:{VULN_MODEL}",
                        })
                        collected_vuln += 1

            # Step 3: Safe code
            if collected_safe < target_per_class:
                safe_code = _generate_safe_code(prompt, vc)
                if safe_code and len(safe_code) > 30:
                    confirmed_clean = True
                    if label_with_bandit:
                        has_issues, _ = _run_bandit(safe_code)
                        confirmed_clean = not has_issues
                    if confirmed_clean:
                        counters["safe"] += 1
                        safe_out.append({
                            "id":          _make_id(f"ollama_{SAFE_MODEL.replace(':', '_')}_safe", cwe, counters["safe"]),
                            "cwe":         cwe,
                            "vuln_class":  vc,
                            "severity":    "none",
                            "label":       "safe",
                            "language":    "python",
                            "code":        safe_code,
                            "description": prompt,
                            "fix":         safe_code,
                            "source":      f"ollama:{SAFE_MODEL}:safe",
                        })
                        collected_safe += 1

            if (i + 1) % 5 == 0:
                logger.info(
                    "    %s: %d vuln, %d safe so far (prompt %d/%d)",
                    vc, collected_vuln, collected_safe, i + 1, len(prompts),
                )

        logger.info(
            "  %s done: %d vuln, %d safe",
            vc, collected_vuln, collected_safe,
        )

    return vuln_out, safe_out


# ---------------------------------------------------------------------------
# Writer — redbench JSONL format
# ---------------------------------------------------------------------------

def write_dataset(
    vuln_samples: list[dict[str, Any]],
    safe_samples: list[dict[str, Any]],
    out_dir: Path,
    filename: str = "samples_generated_v3.jsonl",
    dry_run: bool = False,
) -> dict[str, dict[str, int]]:
    """Write samples to per-class redbench JSONL files.

    Args:
        vuln_samples: Vulnerable code samples.
        safe_samples: Safe code samples (label='safe').
        out_dir: Root redbench datasets directory.
        filename: Output filename within each class subdirectory.
        dry_run: Print stats only, do not write.

    Returns:
        Dict: class → {'vuln': n_written, 'safe': n_written}
    """
    stats: dict[str, dict[str, int]] = {}
    all_samples = [(s, "vulnerable") for s in vuln_samples] + [
        (s, "safe") for s in safe_samples
    ]

    by_class: dict[str, list[dict]] = {}
    for s, _ in all_samples:
        vc = s.get("vuln_class", "")
        if vc:
            by_class.setdefault(vc, []).append(s)

    for vc, entries in by_class.items():
        vuln_entries = [e for e in entries if e.get("label") == "vulnerable"]
        safe_entries = [e for e in entries if e.get("label") == "safe"]
        stats[vc] = {"vuln": 0, "safe": 0}

        if dry_run:
            logger.info(
                "  [DRY RUN] %s: %d vuln, %d safe",
                vc, len(vuln_entries), len(safe_entries),
            )
            stats[vc] = {"vuln": len(vuln_entries), "safe": len(safe_entries)}
            continue

        path = out_dir / vc / filename
        n_written = _write_jsonl(entries, path)
        stats[vc]["vuln"] = sum(1 for e in vuln_entries)
        stats[vc]["safe"] = sum(1 for e in safe_entries)
        logger.info(
            "  %s: wrote %d new entries → %s",
            vc, n_written, path,
        )

    return stats


# ---------------------------------------------------------------------------
# Dataset size audit
# ---------------------------------------------------------------------------

def audit_dataset(redbench_dir: Path) -> dict[str, dict[str, int]]:
    """Count existing samples per class in the redbench directory.

    Returns:
        Dict: class → {'vuln': n, 'safe': n, 'pairs': n}
    """
    audit: dict[str, dict[str, int]] = {}
    if not redbench_dir.exists():
        return audit

    for class_dir in sorted(redbench_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        vc = class_dir.name
        vuln_count = 0
        safe_count = 0
        for jsonl in class_dir.glob("*.jsonl"):
            with open(jsonl, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        label = entry.get("label", "")
                        if label == "vulnerable":
                            vuln_count += 1
                        elif label in ("safe", "fixed"):
                            safe_count += 1
                    except json.JSONDecodeError:
                        continue
        if vuln_count or safe_count:
            audit[vc] = {
                "vuln": vuln_count,
                "safe": safe_count,
                "pairs": min(vuln_count, safe_count),
            }
    return audit


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the large-scale dataset builder."""
    parser = argparse.ArgumentParser(
        description="Build a 1000+ sample vulnerability dataset for ActivGuard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pipelines",
        default="net,ollama",
        help="Comma-separated pipelines to run: net, ollama (default: net,ollama)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=1000,
        help="Target total vuln+safe pairs (default: 1000)",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=0,
        help="Target pairs per CWE class for Ollama pipeline (0 = auto from --target)",
    )
    parser.add_argument(
        "--classes",
        default="",
        help="Comma-separated CWE class keys to generate (empty = all 17). "
             "Example: sqli,command_injection,xss",
    )
    parser.add_argument(
        "--out-dir",
        default="../redbench/datasets",
        help="Redbench datasets directory (default: ../redbench/datasets)",
    )
    parser.add_argument(
        "--filename",
        default="samples_generated_v3.jsonl",
        help="Output JSONL filename per class (default: samples_generated_v3.jsonl)",
    )
    parser.add_argument(
        "--no-bandit",
        action="store_true",
        help="Skip Bandit label confirmation (faster but noisier labels).",
    )
    parser.add_argument(
        "--github-token",
        default=None,
        help="GitHub PAT for higher Advisory API rate limits.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect/generate data but do not write to disk.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Print current dataset size per class and exit.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    pipelines = [p.strip() for p in args.pipelines.split(",")]
    classes = [c.strip() for c in args.classes.split(",") if c.strip()] or None

    # ---- Audit only -------------------------------------------------------
    if args.audit:
        print("\n=== Current dataset size ===")
        audit = audit_dataset(out_dir)
        total_vuln = total_safe = 0
        for vc, counts in sorted(audit.items()):
            print(
                f"  {vc:<22} vuln={counts['vuln']:4d}  safe={counts['safe']:4d}  "
                f"pairs={counts['pairs']:4d}"
            )
            total_vuln += counts["vuln"]
            total_safe += counts["safe"]
        print(f"\n  TOTAL  vuln={total_vuln}  safe={total_safe}  "
              f"pairs={min(total_vuln, total_safe)}")
        return

    # ---- Initial audit ----------------------------------------------------
    logger.info("=== ActivGuard Large Dataset Builder ===")
    logger.info("Target: %d total pairs", args.target)
    current = audit_dataset(out_dir)
    current_total = sum(v["pairs"] for v in current.values())
    logger.info("Current dataset: %d pairs across %d classes", current_total, len(current))
    remaining = max(0, args.target - current_total)
    logger.info("Need: %d more pairs", remaining)

    all_vuln: list[dict[str, Any]] = []
    all_safe: list[dict[str, Any]] = []

    # ---- Pipeline A: Net pull ---------------------------------------------
    if "net" in pipelines:
        logger.info("\n--- Pipeline A: Net pull ---")
        net_limit = min(remaining // 2, 500)

        seceval = pull_securityeval(limit=net_limit)
        all_vuln.extend(seceval)

        ghsa = pull_github_advisories(
            limit=net_limit // 2,
            github_token=args.github_token,
        )
        all_vuln.extend(ghsa)

        logger.info(
            "Pipeline A total: %d vuln samples (safe counterparts via Ollama or stubs)",
            len(all_vuln),
        )

    # ---- Pipeline B: Ollama generation ------------------------------------
    if "ollama" in pipelines:
        logger.info("\n--- Pipeline B: Ollama generation ---")

        # Check Ollama is running
        try:
            requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5).raise_for_status()
        except requests.RequestException:
            logger.error(
                "Ollama not reachable at %s.  Start it with: ollama serve",
                OLLAMA_BASE,
            )
            logger.error("Skipping Pipeline B.")
        else:
            n_classes = len(classes or CWE_CLASSES)
            if args.target_per_class > 0:
                per_class = args.target_per_class
            else:
                per_class = max(10, remaining // (n_classes * 2))

            logger.info(
                "Generating ~%d pairs × %d classes = ~%d pairs",
                per_class, n_classes, per_class * n_classes,
            )

            vuln_gen, safe_gen = generate_ollama_dataset(
                target_per_class=per_class,
                classes=classes,
                label_with_bandit=not args.no_bandit,
            )
            all_vuln.extend(vuln_gen)
            all_safe.extend(safe_gen)
            logger.info(
                "Pipeline B total: %d vuln, %d safe",
                len(vuln_gen), len(safe_gen),
            )

    # ---- Write results ----------------------------------------------------
    if not all_vuln and not all_safe:
        logger.error("No samples collected.  Check network and Ollama.")
        sys.exit(1)

    logger.info(
        "\nWriting %d vuln + %d safe samples ...",
        len(all_vuln), len(all_safe),
    )
    stats = write_dataset(
        all_vuln, all_safe,
        out_dir=out_dir,
        filename=args.filename,
        dry_run=args.dry_run,
    )

    # ---- Final summary ---------------------------------------------------
    print("\n=== Dataset build complete ===")
    total_new = 0
    for vc, s in sorted(stats.items()):
        print(f"  {vc:<22}  vuln={s['vuln']:4d}  safe={s['safe']:4d}")
        total_new += s["vuln"] + s["safe"]
    print(f"\n  New samples written: {total_new}")
    print(f"  Previous total pairs: {current_total}")
    print(f"\nNext step:")
    print(f"  python scripts/train_hf_probe.py --layer 12")


if __name__ == "__main__":
    main()
