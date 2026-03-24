"""
ActivGuard Proxy — CLI entry point.

Starts the FastAPI proxy server using uvicorn.  All tuneable parameters are
exposed as CLI flags so that the server can be launched without editing config
files — useful for research experiments where threshold and probe_interval are
varied systematically.

Usage::

    python -m proxy
    python -m proxy --port 8080 --threshold 0.55
    python -m proxy --port 8080 --threshold 0.40 --stop-on-violation
    python -m proxy --port 8080 --probe-interval 10 --no-bandit

Example with environment variable override::

    ACTIVGUARD_OLLAMA_BASE=http://192.168.1.10:11434 python -m proxy --port 8080

The server exposes:
    POST /v1/chat/completions  — OpenAI-compatible proxy endpoint
    GET  /v1/models            — Available Ollama models
    GET  /health               — Liveness and probe-readiness check
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from proxy.config import (
    BANDIT_SECOND_PASS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    LOG_LEVEL,
    PROBE_INTERVAL,
    PROBE_WEIGHTS,
    STOP_ON_VIOLATION,
    THRESHOLD,
)
from proxy.server import configure

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the proxy CLI.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m proxy",
        description=(
            "ActivGuard: OpenAI-compatible vulnerability-scanning proxy for Ollama. "
            "Intercepts LLM-generated code and scans it with a CodeBERT activation "
            "probe (L1) and Bandit static analysis (L3) in real time."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Network interface to bind the server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="TCP port to listen on.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        metavar="FLOAT",
        help=(
            "P(vulnerable) decision boundary for the CodeBERT probe (0.0–1.0). "
            "Lower values increase recall at the cost of precision."
        ),
    )
    parser.add_argument(
        "--probe-interval",
        type=int,
        default=PROBE_INTERVAL,
        metavar="N",
        help="Probe every N generated tokens.",
    )
    parser.add_argument(
        "--probe-weights",
        default=PROBE_WEIGHTS,
        metavar="PATH",
        help="Path to the trained CodeBERT probe weights .pkl file.",
    )
    parser.add_argument(
        "--stop-on-violation",
        action="store_true",
        default=STOP_ON_VIOLATION,
        help=(
            "Halt token streaming immediately on the first L1-activation VIOLATION. "
            "Provides earliest interruption but may truncate partial responses."
        ),
    )
    parser.add_argument(
        "--no-bandit",
        action="store_true",
        default=False,
        help="Disable the L3-bandit second-pass static analysis.",
    )
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL,
        choices=["debug", "info", "warning", "error"],
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable uvicorn hot-reload (development only — do not use in production).",
    )
    return parser


def main() -> None:
    """Parse CLI arguments, apply configuration, and start uvicorn.

    This function validates argument ranges, applies overrides to the server
    module, then hands off to uvicorn.run().  It does not return under normal
    operation.
    """
    import proxy.config as _config_module  # noqa: PLC0415 — intentional late import for mutation

    parser = _build_parser()
    args = parser.parse_args()

    # Validate threshold range
    if not 0.0 <= args.threshold <= 1.0:
        parser.error(f"--threshold must be between 0.0 and 1.0, got {args.threshold}")

    # Validate probe interval
    if args.probe_interval < 1:
        parser.error(f"--probe-interval must be >= 1, got {args.probe_interval}")

    # Apply CLI overrides to config module (affects default values in server)
    _config_module.THRESHOLD = args.threshold
    _config_module.PROBE_INTERVAL = args.probe_interval
    _config_module.PROBE_WEIGHTS = args.probe_weights
    _config_module.STOP_ON_VIOLATION = args.stop_on_violation
    if args.no_bandit:
        _config_module.BANDIT_SECOND_PASS = False

    # Apply overrides to the already-imported server module state
    configure(
        probe_weights=args.probe_weights,
        threshold=args.threshold,
        probe_interval=args.probe_interval,
        stop_on_violation=args.stop_on_violation,
    )

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    logger.info(
        "Starting ActivGuard proxy on %s:%d  threshold=%.2f  interval=%d  "
        "stop_on_violation=%s  bandit=%s",
        args.host,
        args.port,
        args.threshold,
        args.probe_interval,
        args.stop_on_violation,
        not args.no_bandit,
    )

    uvicorn.run(
        "proxy.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        access_log=True,
    )


if __name__ == "__main__":
    main()
