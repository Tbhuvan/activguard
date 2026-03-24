"""
ActivGuard Pipeline — End-to-end scan of LLM-generated code.

Connects:
  Ollama (local LLM)  →  Layer 3 (AST formal checker)  →  Layer 4 (live NVD CVEs)

Usage:
    python pipeline.py "write a user profile endpoint that fetches by user ID"
    python pipeline.py "write a login function" --model qwen3:8b
    python pipeline.py --interactive
    python pipeline.py --scan-file mycode.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Layer imports (graceful if run from project root)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from verifier.formal_check import FormalChecker
from verifier.property_templates import PROPERTY_TEMPLATES

try:
    from rag.semantic_rag import SecurityRAG
    from rag.antipattern_library import AntiPatternLibrary
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False

try:
    from probe.residual_stream_probe import ResidualStreamProbe
    _PROBE_AVAILABLE = True
except ImportError:
    _PROBE_AVAILABLE = False

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen3-coder:30b"

VULN_CLASSES = list(PROPERTY_TEMPLATES.keys())


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

class OllamaClient:
    """Thin wrapper around the Ollama /api/chat endpoint."""

    def __init__(self, base_url: str = OLLAMA_BASE, model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a response and return the full text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def is_available(self) -> bool:
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=3).raise_for_status()
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]


# ---------------------------------------------------------------------------
# Code extractor
# ---------------------------------------------------------------------------

def extract_code_blocks(text: str) -> list[str]:
    """Pull all ```python ... ``` fenced blocks, or the whole text if none found."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    # Fallback: look for lines that look like Python code
    lines = text.splitlines()
    code_lines = [l for l in lines if l.startswith(("def ", "class ", "import ", "from ", "    "))]
    if len(code_lines) > 3:
        return ["\n".join(code_lines)]
    return [text.strip()]


# ---------------------------------------------------------------------------
# Security scan
# ---------------------------------------------------------------------------

def scan_code_l2(code: str, rag: "SecurityRAG") -> dict:
    """Run code through Layer 2 semantic RAG, querying per top-level function.

    Querying the whole file at once dilutes the signal — a 5-function file
    embeds as the centroid of all functions.  We split by function boundary
    and return the worst (lowest-confidence) result across all snippets.
    """
    import ast as _ast

    snippets: list[str] = []
    try:
        tree = _ast.parse(code)
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                segment = _ast.get_source_segment(code, node)
                if segment and len(segment.strip()) > 20:
                    snippets.append(segment.strip())
    except SyntaxError:
        pass

    if not snippets:
        snippets = [code]

    all_evidence: list[str] = []
    all_patterns: list[str] = []
    max_confidence = 0.0

    for snippet in snippets:
        r = rag.query(snippet, n_results=3)
        if not r["safe"]:
            all_evidence.extend(r["evidence"])
            all_patterns.extend(r["patterns_matched"])
            max_confidence = max(max_confidence, r["confidence"])

    return {
        "safe": len(all_patterns) == 0,
        "evidence": all_evidence[:6],
        "patterns_matched": list(dict.fromkeys(all_patterns)),  # deduplicate, preserve order
        "collections_searched": ["antipatterns", "auth_model", "threat_intel"],
        "confidence": round(max_confidence, 4),
    }


def scan_code(code: str, checker: FormalChecker) -> list[dict]:
    """Run code through all Layer 3 checks. Returns list of findings."""
    findings = []
    for vuln_class in VULN_CLASSES:
        result = checker.verify(code, vuln_class)
        if result["result"] == "VIOLATION":
            findings.append({
                "vuln_class": vuln_class,
                "result": result["result"],
                "evidence": result["evidence"],
                "confidence": result["confidence"],
                "property": result["property"],
            })
    return findings


def fetch_related_cves(vuln_classes: list[str], max_per_class: int = 2) -> list[dict]:
    """Pull recent real CVEs from NVD for the detected vulnerability classes."""
    keyword_map = {
        "IDOR":            "insecure direct object reference authorization bypass",
        "SQLi":            "SQL injection",
        "SSRF":            "server-side request forgery",
        "auth_bypass":     "authentication bypass",
        "path_traversal":  "path traversal directory traversal",
        "XSS":             "cross-site scripting XSS",
        "deserialization": "deserialization unsafe",
        "command_injection": "command injection OS injection",
    }
    cves = []
    try:
        from connectors.nvd_connector import NVDConnector
        for vc in vuln_classes:
            kw = keyword_map.get(vc, vc)
            conn = NVDConnector(keyword_filters=[kw], results_per_page=max_per_class)
            indicators = conn.pull()
            for ind in indicators[:max_per_class]:
                cves.append({
                    "id": ind.id,
                    "severity": ind.severity,
                    "cvss": ind.cvss_score,
                    "vuln_class": vc,
                    "description": ind.description[:120] + "..." if len(ind.description) > 120 else ind.description,
                })
            time.sleep(0.5)  # NVD rate limit
    except Exception as e:
        cves.append({"error": str(e)})
    return cves


# ---------------------------------------------------------------------------
# Report renderer
# ---------------------------------------------------------------------------

SEVERITY_ICON = {"critical": "[CRIT]", "high": "[HIGH]", "medium": "[MED]", "low": "[LOW]", "unknown": "[?]"}

def render_report(
    prompt: str,
    model: str,
    generated_code: str,
    findings: list[dict],
    cves: list[dict],
    elapsed: float,
    l2_result: dict | None = None,
    l1_result: dict | None = None,
) -> str:
    lines = []
    sep = "=" * 72
    thin = "-" * 72

    lines.append(sep)
    lines.append("  ACTIVGUARD SCAN REPORT")
    lines.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  model: {model}")
    lines.append(sep)

    lines.append("\nPROMPT")
    lines.append(thin)
    lines.append(textwrap.fill(prompt, width=70))

    lines.append("\nGENERATED CODE")
    lines.append(thin)
    for line in generated_code.splitlines():
        lines.append("  " + line)

    if l1_result is not None:
        lines.append("\nACTIVATION PROBE  (Layer 1 — Residual Stream)")
        lines.append(thin)
        result = l1_result.get("result", "UNKNOWN")
        conf = l1_result.get("confidence", 0.0)
        probe_model = l1_result.get("model", "?")
        if result == "VIOLATION":
            lines.append(f"  [!] VIOLATION  P(vulnerable)={conf:.3f}  model={probe_model}")
        elif result == "UNTRAINED":
            lines.append(f"  [--] UNTRAINED  Run train_probe.py to enable Layer 1")
        elif result == "ERROR":
            lines.append(f"  [ERR] {l1_result.get('evidence', 'error')}")
        else:
            lines.append(f"  [OK] VERIFIED  P(vulnerable)={conf:.3f}  model={probe_model}")
        per_fn = l1_result.get("per_function", [])
        if per_fn:
            for fn in per_fn:
                flag = "!" if fn.get("result") == "VIOLATION" else "."
                lines.append(f"    [{flag}] {fn.get('function','?')}  p={fn.get('confidence',0):.3f}")
        lines.append("")

    if l2_result is not None:
        lines.append("\nSEMATIC RAG  (Layer 2 — Anti-Pattern Retrieval)")
        lines.append(thin)
        if not l2_result.get("safe", True):
            lines.append(f"  [!] Matched {len(l2_result['patterns_matched'])} known pattern(s)  "
                         f"(confidence: {l2_result['confidence']:.2%})")
            for i, ev in enumerate(l2_result["evidence"][:3], 1):
                lines.append(f"  Match #{i}: {ev[:120]}...")
        else:
            lines.append(f"  [OK]  No anti-pattern matches  (confidence: {l2_result['confidence']:.2%})")
        lines.append("")

    lines.append(f"SECURITY SCAN  (Layer 3 — AST Formal Checker)  [{elapsed:.1f}s]")
    lines.append(thin)

    if not findings:
        lines.append("  [CLEAN]  No vulnerabilities detected across all check classes.")
        lines.append(f"  Checked: {', '.join(VULN_CLASSES)}")
    else:
        lines.append(f"  [!] {len(findings)} FINDING(S) DETECTED\n")
        for i, f in enumerate(findings, 1):
            lines.append(f"  Finding #{i}: {f['vuln_class']}  (confidence: {f['confidence']:.0%})")
            lines.append(f"  Property:  {f['property']}")
            lines.append(f"  Evidence:  {f['evidence']}")
            lines.append("")

    if cves and not any("error" in c for c in cves):
        lines.append("RELATED LIVE CVEs  (Layer 4 — NVD Feed)")
        lines.append(thin)
        for c in cves:
            icon = SEVERITY_ICON.get(c.get("severity", "unknown"), "[?]")
            lines.append(f"  {icon}  {c['id']}  CVSS={c.get('cvss', '?'):.1f}  [{c['vuln_class']}]")
            lines.append(f"       {c['description']}")
        lines.append("")

    lines.append(sep)
    verdict = "VULNERABLE" if findings else "CLEAN"
    lines.append(f"  VERDICT: {verdict}  |  {len(findings)} finding(s)  |  {len(cves)} related CVEs")
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main scan function
# ---------------------------------------------------------------------------

def scan_prompt(
    prompt: str,
    model: str = DEFAULT_MODEL,
    fetch_cves: bool = True,
    system_prompt: str | None = None,
) -> dict:
    """Full pipeline: prompt → generate → scan → report."""

    if system_prompt is None:
        system_prompt = (
            "You are an expert Python developer. Write clean, idiomatic Python code. "
            "Return ONLY the code, no explanation. Use realistic variable names and include "
            "a docstring. Do not add ```python fences unless asked."
        )

    client = OllamaClient(model=model)

    # Step 1: Generate
    t0 = time.time()
    raw = client.generate(prompt, system=system_prompt)
    blocks = extract_code_blocks(raw)
    code = blocks[0] if blocks else raw
    gen_time = time.time() - t0

    # Step 2a: Layer 1 — residual stream activation probe
    l1_result: dict | None = None
    if _PROBE_AVAILABLE:
        try:
            probe = ResidualStreamProbe()
            l1_result = probe.score_functions(code)
        except Exception as exc:
            l1_result = {"result": "ERROR", "confidence": 0.0,
                         "evidence": str(exc), "model": "?"}

    # Step 2b: Layer 2 — semantic RAG
    l2_result: dict | None = None
    if _RAG_AVAILABLE:
        try:
            rag = SecurityRAG(persist_dir=".activguard/chroma")
            l2_result = scan_code_l2(code, rag)
        except Exception as exc:
            l2_result = {"safe": True, "evidence": [], "patterns_matched": [],
                         "confidence": 0.0, "error": str(exc)}

    # Step 2b: Layer 3 — AST formal checker
    checker = FormalChecker()
    t1 = time.time()
    findings = scan_code(code, checker)
    scan_time = time.time() - t1

    # Step 3: CVE lookup for detected vuln classes
    vuln_classes_found = [f["vuln_class"] for f in findings]
    cves = fetch_related_cves(vuln_classes_found) if fetch_cves and vuln_classes_found else []

    report = render_report(prompt, model, code, findings, cves, gen_time + scan_time,
                           l2_result, l1_result)

    return {
        "prompt": prompt,
        "model": model,
        "code": code,
        "findings": findings,
        "cves": cves,
        "report": report,
        "gen_time_s": round(gen_time, 2),
        "scan_time_s": round(scan_time, 3),
        "verdict": "VULNERABLE" if findings else "CLEAN",
        "l1_result": l1_result,
        "l2_result": l2_result,
    }


def scan_file(path: str, model: str = DEFAULT_MODEL) -> dict:
    """Scan an existing code file (no LLM generation)."""
    code = Path(path).read_text(encoding="utf-8")

    l1_result: dict | None = None
    if _PROBE_AVAILABLE:
        try:
            probe = ResidualStreamProbe()
            l1_result = probe.score_functions(code)
        except Exception as exc:
            l1_result = {"result": "ERROR", "confidence": 0.0,
                         "evidence": str(exc), "model": "?"}

    l2_result: dict | None = None
    if _RAG_AVAILABLE:
        try:
            rag = SecurityRAG(persist_dir=".activguard/chroma")
            l2_result = scan_code_l2(code, rag)
        except Exception as exc:
            l2_result = {"safe": True, "evidence": [], "patterns_matched": [],
                         "confidence": 0.0, "error": str(exc)}

    checker = FormalChecker()
    findings = scan_code(code, checker)
    vuln_classes_found = [f["vuln_class"] for f in findings]
    cves = fetch_related_cves(vuln_classes_found) if vuln_classes_found else []
    report = render_report(f"[file: {path}]", "N/A", code, findings, cves, 0.0,
                           l2_result, l1_result)
    return {"path": path, "code": code, "findings": findings, "cves": cves, "report": report,
            "l1_result": l1_result, "l2_result": l2_result}


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

DEMO_PROMPTS = [
    "Write a Django REST API endpoint that fetches a user's order by order ID",
    "Write a function that searches users in a database by name",
    "Write a Python function that fetches a URL provided by the user",
    "Write a file download endpoint that serves files by filename from a uploads folder",
    "Write a login function that checks username and password",
]

def interactive_mode(model: str) -> None:
    print("\nActivGuard Interactive Scanner")
    print("Type a coding prompt, 'demo' for demo prompts, or 'quit' to exit.\n")

    client = OllamaClient(model=model)
    if not client.is_available():
        print("[ERROR] Ollama is not running at", OLLAMA_BASE)
        return

    while True:
        try:
            prompt = input("prompt> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if prompt.lower() == "demo":
            print("\nDemo prompts:")
            for i, p in enumerate(DEMO_PROMPTS, 1):
                print(f"  {i}. {p}")
            choice = input("Pick a number (or press Enter to skip): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(DEMO_PROMPTS):
                prompt = DEMO_PROMPTS[int(choice) - 1]
            else:
                continue

        print(f"\n[*] Generating with {model}...")
        result = scan_prompt(prompt, model=model, fetch_cves=bool(result.get("findings")) if False else True)
        print(result["report"])
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ActivGuard — scan LLM-generated code for vulnerabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python pipeline.py "write a user profile endpoint"
          python pipeline.py "write a login function" --model qwen3:8b
          python pipeline.py --interactive
          python pipeline.py --scan-file mycode.py
          python pipeline.py "write a search function" --json
          python pipeline.py --list-models
        """),
    )
    parser.add_argument("prompt", nargs="?", help="Coding prompt to scan")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--scan-file", "-f", metavar="FILE", help="Scan an existing file instead of generating")
    parser.add_argument("--json", "-j", action="store_true", help="Output raw JSON instead of report")
    parser.add_argument("--no-cves", action="store_true", help="Skip NVD CVE lookup")
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models")

    args = parser.parse_args()

    if args.list_models:
        client = OllamaClient()
        try:
            for m in client.list_models():
                print(m)
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
        return

    if args.interactive:
        interactive_mode(args.model)
        return

    if args.scan_file:
        result = scan_file(args.scan_file, model=args.model)
        if args.json:
            print(json.dumps({k: v for k, v in result.items() if k != "report"}, indent=2))
        else:
            print(result["report"])
        return

    if not args.prompt:
        parser.print_help()
        return

    client = OllamaClient(model=args.model)
    if not client.is_available():
        print(f"[ERROR] Ollama not running at {OLLAMA_BASE}. Start it with: ollama serve", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Generating with {args.model}...")
    result = scan_prompt(args.prompt, model=args.model, fetch_cves=not args.no_cves)

    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "report"}, indent=2))
    else:
        print(result["report"])

    sys.exit(1 if result["findings"] else 0)


if __name__ == "__main__":
    main()
