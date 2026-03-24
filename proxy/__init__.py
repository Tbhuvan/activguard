"""
ActivGuard Proxy — OpenAI-Compatible Vulnerability-Scanning Proxy Server.

This package exposes a FastAPI application that sits between any coding
assistant (Cursor, VS Code Copilot, Open WebUI, etc.) and an Ollama backend.
It intercepts LLM-generated code in real time, applying a two-pass security
analysis pipeline before returning content to the caller:

  Layer 1 (L1-activation): CodeBERT probe sampled every PROBE_INTERVAL tokens
      during streaming, providing sub-completion latency detection.
  Layer 3 (L3-bandit): Bandit static analysis on the full completed output,
      providing a second-pass structural verification.

Research context (RQ4):
    The proxy measures the precision/recall tradeoff at both detection layers
    and exposes the per-request score timeline via SSE violation events,
    enabling downstream evaluation of the combined detection strategy.

Usage::

    python -m proxy --port 8080 --threshold 0.55 --stop-on-violation

Public API:
    app — FastAPI application instance (import for ASGI mounting or testing).
"""

from __future__ import annotations

from proxy.server import app

__all__ = ["app"]
__version__ = "0.1.0"
