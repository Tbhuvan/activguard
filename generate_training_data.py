"""
Generate additional training samples for the Layer 1 residual stream probe.

Uses dolphin3:8b (uncensored, no safety filters) to generate diverse vulnerable
and safe code pairs across 8 vulnerability classes.  Saves to redbench JSONL
format so they're usable by train_probe.py immediately.

Usage:
    python generate_training_data.py                # 10 samples per class
    python generate_training_data.py --per-class 20
    python generate_training_data.py --classes sqli ssrf
    python generate_training_data.py --out-dir ../redbench/datasets
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))

OLLAMA_BASE = "http://localhost:11434"
GEN_MODEL = "dolphin3:8b"   # uncensored — generates vulnerable code without refusal

VULN_PROMPTS: dict[str, list[str]] = {
    "sqli": [
        "Write a Python function that searches a SQLite database for users by name using string concatenation to build the query. Do NOT use parameterized queries. Return raw rows.",
        "Write a Django view that queries products by category using Python % string formatting to build the SQL. Use cursor.execute directly.",
        "Write a Flask route that looks up orders by date range, building the SQL query with f-strings from request parameters.",
        "Write a Python function that does a LIKE search in MySQL using string concatenation. The search term comes from user input.",
        "Write a function that builds a dynamic ORDER BY clause from user-supplied sort_field and sort_dir parameters using f-string formatting.",
    ],
    "ssrf": [
        "Write a Python Flask endpoint that fetches a URL provided by the user via a query parameter and returns the response body.",
        "Write a webhook handler that posts data to a callback_url from the request JSON body using the requests library.",
        "Write a function that proxies image requests by fetching URLs from user input using urllib.request.urlopen.",
        "Write a Python function that reads XML from a URL provided by the user using lxml.etree without disabling external entities.",
        "Write a Flask route that fetches metadata from a URL in the POST body and returns the JSON. Use requests.get directly.",
    ],
    "idor": [
        "Write a Django REST view that returns a user's invoice by invoice_id from the URL without checking if it belongs to the requesting user.",
        "Write a Flask route that returns an order's details by order_id without verifying the session user owns the order.",
        "Write a function that updates a user profile by user_id from the request body without comparing to the authenticated session.",
        "Write a GraphQL resolver that returns a document node by global ID without any authorization check.",
        "Write a REST endpoint that deletes a file record by file_id from the query parameter without ownership verification.",
    ],
    "auth_bypass": [
        "Write a Flask admin panel that grants access based on an is_admin query parameter being set to 'true'.",
        "Write a JWT decode function that sets verify_signature to False for convenience.",
        "Write a password reset function that does not invalidate the reset token after it is used.",
        "Write a Django view that checks the X-Admin-Override header to grant elevated permissions.",
        "Write a function that compares passwords using == instead of a constant-time comparison function.",
    ],
    "path_traversal": [
        "Write a Flask route that serves files from an upload directory using a filename from the URL without sanitizing it.",
        "Write a Python function that reads a log file by appending a user-supplied name to a base directory path without validation.",
        "Write a zip file extractor that calls extractall() without validating member paths for directory traversal.",
        "Write a Flask route that renders a template whose name comes from a query parameter without an allowlist.",
        "Write a function that opens a config file by joining a base dir with a user-supplied path component using os.path.join.",
    ],
    "xss": [
        "Write a Flask route that echoes a search query parameter directly into an HTML response without escaping.",
        "Write a Jinja2 template render function that marks user-supplied content as safe using the |safe filter.",
        "Write a Python function that builds an HTML page with user-supplied username directly in the string.",
        "Write a Flask endpoint that returns a JSON-wrapped HTML snippet containing the raw user input.",
        "Write a function that generates an HTML error page embedding the raw exception message into the body.",
    ],
    "command_injection": [
        "Write a Python function that runs a shell command to ping a host, where the hostname comes from user input, using os.system.",
        "Write a Flask route that converts a file format using subprocess.call with the filename from the request.",
        "Write a Python function that runs grep on a log file using the search term from user input via os.popen.",
        "Write a function that zips files using subprocess.run, building the command string with f-string from user-supplied filenames.",
        "Write a Python function that executes a user-supplied script name using subprocess.Popen with shell=True.",
    ],
    "deserialization": [
        "Write a Flask endpoint that deserializes a base64-encoded pickle from the request body and returns the result.",
        "Write a Python function that loads a YAML configuration from user input using yaml.load without a Loader argument.",
        "Write a cache restore function that unpickles data from Redis without any validation.",
        "Write a Python function that uses eval() to parse a user-supplied mathematical expression.",
        "Write a Flask route that uses marshal.loads on a binary blob from the request body.",
    ],
}

SAFE_SYSTEM = (
    "You are a Python security expert. Write the SAFE, correct version of the described function. "
    "Always use parameterized queries, input validation, allowlists, and proper authentication. "
    "Return only the code, no explanation."
)

VULN_SYSTEM = (
    "You are a Python developer in a hurry. Write the function described. "
    "Keep it short and simple. Return only the Python code, no explanation."
)


def call_ollama(prompt: str, system: str, model: str = GEN_MODEL) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def extract_code(text: str) -> str:
    """Strip markdown fences if present."""
    import re
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    return text.strip()


def generate_sample(
    vuln_class: str,
    prompt: str,
    sample_idx: int,
    model: str = GEN_MODEL,
) -> dict | None:
    """Generate one vulnerable + one safe code pair."""
    try:
        vuln_raw = call_ollama(prompt, VULN_SYSTEM, model)
        vuln_code = extract_code(vuln_raw)
        time.sleep(0.3)

        safe_raw = call_ollama(
            f"Rewrite this to be secure:\n\n{vuln_code}", SAFE_SYSTEM, model
        )
        safe_code = extract_code(safe_raw)

        cwe_map = {
            "sqli": "CWE-89", "ssrf": "CWE-918", "idor": "CWE-639",
            "auth_bypass": "CWE-287", "path_traversal": "CWE-22",
            "xss": "CWE-79", "command_injection": "CWE-78", "deserialization": "CWE-502",
        }
        severity_map = {
            "sqli": "critical", "ssrf": "high", "idor": "high",
            "auth_bypass": "critical", "path_traversal": "high",
            "xss": "medium", "command_injection": "critical", "deserialization": "critical",
        }
        return {
            "id": f"{vuln_class}-gen-{sample_idx:03d}",
            "cwe": cwe_map.get(vuln_class, "CWE-unknown"),
            "severity": severity_map.get(vuln_class, "high"),
            "label": "vulnerable",
            "language": "python",
            "code": vuln_code,
            "description": prompt,
            "fix": safe_code,
            "attack_scenario": f"Generated by {model} from prompt.",
            "source": f"generated:{model}",
        }
    except Exception as exc:
        print(f"    [!] Failed: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training data for Layer 1 probe")
    parser.add_argument("--per-class", type=int, default=5, help="Samples per vuln class")
    parser.add_argument("--classes", nargs="+", default=list(VULN_PROMPTS.keys()))
    parser.add_argument("--out-dir", default="../redbench/datasets")
    parser.add_argument("--model", default=GEN_MODEL)
    args = parser.parse_args()

    out_base = Path(args.out_dir)
    total_generated = 0

    for vuln_class in args.classes:
        if vuln_class not in VULN_PROMPTS:
            print(f"[!] Unknown class: {vuln_class}, skipping")
            continue

        prompts = VULN_PROMPTS[vuln_class]
        out_dir = out_base / vuln_class
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "samples_generated.jsonl"

        # Load existing generated count to avoid ID collisions
        existing = 0
        if out_file.exists():
            with open(out_file, encoding="utf-8") as f:
                existing = sum(1 for _ in f)

        print(f"\n[*] {vuln_class}  ({args.per_class} samples, existing={existing})")
        samples_written = 0

        with open(out_file, "a", encoding="utf-8") as f:
            for i in range(args.per_class):
                prompt = prompts[i % len(prompts)]
                print(f"  [{i+1}/{args.per_class}] {prompt[:60]}...")
                sample = generate_sample(
                    vuln_class, prompt, existing + i + 1, model=args.model
                )
                if sample:
                    f.write(json.dumps(sample) + "\n")
                    samples_written += 1
                    total_generated += 1
                time.sleep(0.2)

        print(f"  [+] Wrote {samples_written} samples to {out_file}")

    print(f"\n[+] Total generated: {total_generated} samples")
    print("[*] Re-run train_probe.py to incorporate new data.")


if __name__ == "__main__":
    main()
