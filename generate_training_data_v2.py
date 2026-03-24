"""
Expanded training data generation — v2 with 15+ diverse prompts per class.

Targets 500+ total samples for statistical rigour (PhD application threshold).
Run after generate_training_data.py to supplement existing samples.

Usage:
    python generate_training_data_v2.py --per-class 30    # 240 new samples
    python generate_training_data_v2.py --classes sqli ssrf --per-class 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from generate_training_data import (
    OLLAMA_BASE, GEN_MODEL, SAFE_SYSTEM, VULN_SYSTEM,
    call_ollama, extract_code, generate_sample,
)

# Extended prompts — 15 per class (supplements the 5 in v1)
VULN_PROMPTS_V2: dict[str, list[str]] = {
    "sqli": [
        "Write a Python function that authenticates a user by querying MySQL with username and password using string concatenation.",
        "Write a FastAPI endpoint that searches blog posts by tag, building the WHERE clause with f-string formatting.",
        "Write a Python script that generates a report by querying SQLite with a date range from command-line arguments using % formatting.",
        "Write a SQLAlchemy raw query function that filters products by price range using string interpolation instead of bind parameters.",
        "Write a Django management command that bulk-deletes records matching a user-supplied pattern using raw SQL cursor.execute.",
        "Write a Python function that inserts audit log entries with os.getlogin() in the query string via f-string.",
        "Write a Flask search endpoint that queries a PostgreSQL full-text search index using string concatenation for the query term.",
        "Write a function that retrieves paginated results from MySQL, building LIMIT and OFFSET with user-supplied page/size params using format().",
        "Write a Python admin script that counts records per category, building GROUP BY clause from user-supplied column name with f-string.",
        "Write a REST API handler that updates user preferences by constructing SET clause from a dict using string join.",
    ],
    "ssrf": [
        "Write a Python function that validates a URL by attempting to connect to it using requests.get, taking the URL from a form field.",
        "Write a FastAPI route that downloads and caches remote images from user-supplied URLs using urllib.request.",
        "Write a webhook relay service that forwards POST requests to URLs stored in a database without validation.",
        "Write a Python function that checks if a remote API is healthy by fetching its URL from environment variables + user input suffix.",
        "Write a metadata scraper that fetches Open Graph tags from URLs provided by users via a REST parameter.",
        "Write a PDF generator that fetches HTML from a URL parameter using requests and passes it to weasyprint.",
        "Write a Python function that imports a remote JSON schema from a URL supplied in the request body using urllib.",
        "Write a Flask route that loads a remote configuration file from a URL in the POST body using requests.get.",
        "Write an async Python function that fetches content from user-supplied URLs concurrently using aiohttp.",
        "Write a Python function that resolves a short URL by following redirects to the final destination using requests.",
    ],
    "idor": [
        "Write a Flask route that downloads an invoice PDF by invoice_id from the URL path without any access control.",
        "Write a Django view that returns a patient's medical record by record_id without checking the authenticated user.",
        "Write a FastAPI endpoint that streams an uploaded file by file_uuid from the path without ownership check.",
        "Write a REST handler that marks a notification as read by notification_id from the request body without user verification.",
        "Write a Python function that retrieves API keys for a project by project_id without validating the calling user owns the project.",
        "Write a GraphQL mutation that updates a payment method by payment_method_id without confirming it belongs to the current user.",
        "Write a Flask route that returns chat history by conversation_id path parameter without session validation.",
        "Write a Django admin action that exports user data by user_id list from POST body without role check.",
        "Write a REST endpoint that cancels a subscription by subscription_id query parameter without ownership verification.",
        "Write a Python function that fetches order tracking info by tracking_id without checking the order belongs to the session user.",
    ],
    "auth_bypass": [
        "Write a Flask route that skips authentication if the request comes from 127.0.0.1 or localhost.",
        "Write a Django middleware that grants admin access if the request has a debug=true query parameter.",
        "Write a Python decorator that bypasses permission checks when the environment is 'development'.",
        "Write a JWT verification function that accepts tokens with 'none' as the algorithm.",
        "Write a session validation function that trusts the user_role value stored in the client-side cookie without server-side verification.",
        "Write a password change handler that does not require the current password if an admin_token header is present.",
        "Write a Flask route that allows file upload without authentication if the Content-Type is multipart/form-data.",
        "Write a Django view that grants superuser privileges if the username ends with '@admin.internal'.",
        "Write an API endpoint that skips rate limiting for requests with an X-Internal-Request header.",
        "Write a Python function that validates API keys by checking only the first 8 characters for performance.",
    ],
    "path_traversal": [
        "Write a Flask route that reads and returns files from a user-controlled subdirectory without path canonicalization.",
        "Write a Python function that writes uploaded files to disk using the original filename from the upload without sanitization.",
        "Write a Django view that loads a locale file by language code from the URL parameter using open().",
        "Write a FastAPI endpoint that copies a file to a destination path provided in the POST body using shutil.copy.",
        "Write a Python function that deletes files by filename from request parameters using os.remove without validation.",
        "Write a Flask route that serves static assets by joining BASE_DIR with a path from the URL without checking for traversal.",
        "Write a Python script that extracts tar.gz archives without checking for path traversal in member names.",
        "Write a function that reads plugin configuration by joining a plugins directory with the plugin name from user input.",
        "Write a Flask endpoint that moves uploaded files to a directory specified in the form data using os.rename.",
        "Write a Python function that imports a module by constructing the module path from user-supplied component names.",
    ],
    "xss": [
        "Write a Flask route that renders a user profile page with the username from the database directly in a Jinja2 string.",
        "Write a Python function that generates a welcome email HTML using f-string with the user's display name.",
        "Write a FastAPI endpoint that returns an HTML snippet containing the user's search query for display.",
        "Write a Django view that renders a comment thread with raw comment text inserted into the template.",
        "Write a Python function that generates a CSV download link page with the filename from the query parameter embedded in HTML.",
        "Write a Flask error handler that returns an HTML page with the exception message in the response body.",
        "Write a Python function that creates an HTML report with user-supplied column headers directly in table tags.",
        "Write a Django view that displays form validation errors by rendering the user's input back in the error message HTML.",
        "Write a Flask route that generates a shareable link preview page with the URL title fetched from the target site embedded raw.",
        "Write a Python function that builds an HTML notification email with the message body from a database field without escaping.",
    ],
    "command_injection": [
        "Write a Python function that resizes an image using ImageMagick, taking the dimensions from user input via os.system.",
        "Write a Flask route that compresses files using tar, building the command with filenames from the request.",
        "Write a Python script that sends email using sendmail, constructing the recipient address with string formatting.",
        "Write a FastAPI endpoint that runs ffmpeg to convert video files, using the output format from query parameters via subprocess.call.",
        "Write a Python function that generates PDF from HTML using wkhtmltopdf, taking the URL from user input with os.popen.",
        "Write a Flask route that runs nmap on a host provided by the user using subprocess.run with shell=True.",
        "Write a Python function that executes git commands with a branch name from user input using os.system.",
        "Write a Flask endpoint that runs a user-specified linter on uploaded Python code using subprocess with shell=True.",
        "Write a Python function that backs up a database by running pg_dump with a db name from request parameters via os.system.",
        "Write a Flask route that runs a user-supplied test command via subprocess.Popen with shell=True and returns stdout.",
    ],
    "deserialization": [
        "Write a Flask session handler that restores user state by unpickling a base64-decoded cookie value.",
        "Write a Python function that loads a machine learning model from user-supplied bytes using pickle.loads.",
        "Write a FastAPI endpoint that accepts a serialized Python object in the request body and calls a method on it after unpickling.",
        "Write a Python function that parses a YAML config file from user input using yaml.load without SafeLoader.",
        "Write a Flask route that restores a game save state by deserializing a JSON-like format using eval().",
        "Write a Python function that loads a configuration object by unpickling data from a Redis key named after the user.",
        "Write a FastAPI handler that reconstructs a query filter object from a pickled base64 URL parameter.",
        "Write a Python function that imports and executes a user-supplied code string using exec() for dynamic scripting.",
        "Write a Flask endpoint that processes a serialized numpy array from the request body using pickle without validation.",
        "Write a Python function that restores task state from a message queue by calling pickle.loads on the message body.",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanded training data generation (v2)")
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--classes", nargs="+", default=list(VULN_PROMPTS_V2.keys()))
    parser.add_argument("--out-dir", default="../redbench/datasets")
    parser.add_argument("--model", default=GEN_MODEL)
    args = parser.parse_args()

    out_base = Path(args.out_dir)
    total_generated = 0

    for vuln_class in args.classes:
        if vuln_class not in VULN_PROMPTS_V2:
            print(f"[!] Unknown class: {vuln_class}")
            continue

        prompts = VULN_PROMPTS_V2[vuln_class]
        out_dir = out_base / vuln_class
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "samples_generated_v2.jsonl"

        existing = 0
        if out_file.exists():
            with open(out_file, encoding="utf-8") as f:
                existing = sum(1 for _ in f)

        print(f"\n[*] {vuln_class}  ({args.per_class} new, existing_v2={existing})")

        written = 0
        with open(out_file, "a", encoding="utf-8") as f:
            for i in range(args.per_class):
                prompt = prompts[i % len(prompts)]
                print(f"  [{i+1}/{args.per_class}] {prompt[:65]}...")
                # Use v2 IDs to avoid collision with v1
                sample = generate_sample(vuln_class, prompt, existing + i + 1 + 1000, args.model)
                if sample:
                    sample["id"] = f"{vuln_class}-v2-{existing + i + 1:03d}"
                    sample["source"] = f"generated_v2:{args.model}"
                    f.write(json.dumps(sample) + "\n")
                    written += 1
                    total_generated += 1
                time.sleep(0.2)

        print(f"  [+] Wrote {written} samples to {out_file}")

    print(f"\n[+] Total: {total_generated} new samples")
    print(f"[*] Re-run train_layer_probe.py --reembed to incorporate new data.")


if __name__ == "__main__":
    main()
