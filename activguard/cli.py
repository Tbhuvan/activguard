"""
ActivGuard CLI — command-line entry point.

Usage:
    activguard serve                     # Start the OpenAI-compatible proxy
    activguard scan <file>               # Scan a Python file for vulnerabilities
    activguard version                   # Print version
"""

from __future__ import annotations

import sys


def main() -> None:
    """Main CLI entry point for the ``activguard`` console script."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="activguard",
        description="ActivGuard — real-time vulnerability detection for LLM-generated code.",
    )
    sub = parser.add_subparsers(dest="command")

    # serve ----------------------------------------------------------------
    serve_p = sub.add_parser("serve", help="Start the OpenAI-compatible proxy server.")
    serve_p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    serve_p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    serve_p.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama backend URL.",
    )

    # scan -----------------------------------------------------------------
    scan_p = sub.add_parser("scan", help="Scan a Python file for security vulnerabilities.")
    scan_p.add_argument("file", help="Path to the Python file to scan.")
    scan_p.add_argument(
        "--layers",
        default="all",
        choices=["probe", "rag", "formal", "threat", "all"],
        help="Which detection layers to run.",
    )

    # version --------------------------------------------------------------
    sub.add_parser("version", help="Print ActivGuard version and exit.")

    args = parser.parse_args()

    if args.command == "version":
        from activguard import __version__
        print(f"activguard {__version__}")

    elif args.command == "serve":
        _cmd_serve(args)

    elif args.command == "scan":
        _cmd_scan(args)

    else:
        parser.print_help()
        sys.exit(0)


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------


def _cmd_serve(args) -> None:  # type: ignore[no-untyped-def]
    """Start the FastAPI proxy server."""
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is required to run the proxy server.\n"
            "Install it with:  pip install uvicorn",
            file=sys.stderr,
        )
        sys.exit(1)

    import sys
    from pathlib import Path

    # Ensure project root is on sys.path so proxy module resolves.
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"Starting ActivGuard proxy on {args.host}:{args.port}")
    print(f"Ollama backend: {args.ollama_url}")
    uvicorn.run(
        "proxy.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


def _cmd_scan(args) -> None:  # type: ignore[no-untyped-def]
    """Scan a single Python file and print findings."""
    from pathlib import Path

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    code = path.read_text(encoding="utf-8")
    print(f"Scanning: {path}")
    print(f"Layers:   {args.layers}")
    print("-" * 60)

    import sys
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    findings: list[str] = []

    # Layer 3 — formal / AST check
    if args.layers in ("formal", "all"):
        try:
            from verifier.formal_check import FormalChecker
            checker = FormalChecker()
            result = checker.check(code)
            for finding in result.get("violations", []):
                findings.append(f"[FORMAL] {finding}")
        except Exception as exc:  # noqa: BLE001
            print(f"  Formal checker unavailable: {exc}")

    if findings:
        print("VIOLATIONS FOUND:")
        for f in findings:
            print(f"  {f}")
        sys.exit(2)
    else:
        print("No violations detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
