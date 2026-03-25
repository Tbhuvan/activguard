"""
ActivGuard Proxy — Runtime Configuration.

All tuneable parameters are defined here as module-level constants so that
the server, probe orchestration, and CLI entry point share a single source
of truth.  Each constant can be overridden at startup via CLI flags or
environment variables without modifying this file.

Configuration hierarchy (highest priority wins):
    1. CLI arguments passed to __main__.py
    2. Environment variables (prefixed ACTIVGUARD_)
    3. Defaults defined in this module
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path


def get_probe_auc(weights_path: str | None = None) -> float | None:
    """Return the CV AUC of the active probe from saved weights — never hardcoded.

    Args:
        weights_path: Path to .pkl file. Defaults to PROBE_WEIGHTS env var or
                      .activguard/layer_probe_weights.pkl.

    Returns:
        float AUC or None if weights not found.
    """
    path = weights_path or os.environ.get(
        "ACTIVGUARD_PROBE_WEIGHTS", ".activguard/layer_probe_weights.pkl"
    )
    for candidate in (Path(path), Path(__file__).parent.parent / path):
        if candidate.exists():
            try:
                with open(candidate, "rb") as f:
                    return float(pickle.load(f).get("auc_cv", 0.0))
            except Exception:
                return None
    return None

# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

OLLAMA_BASE: str = os.environ.get("ACTIVGUARD_OLLAMA_BASE", "http://localhost:11434")
"""Base URL of the Ollama instance that serves generation requests."""

# ---------------------------------------------------------------------------
# Probe configuration
# ---------------------------------------------------------------------------

PROBE_WEIGHTS: str = os.environ.get(
    "ACTIVGUARD_PROBE_WEIGHTS",
    ".activguard/layer_probe_weights.pkl",
)
"""Path to the trained CodeBERT probe weights .pkl file (relative to CWD)."""

THRESHOLD: float = float(os.environ.get("ACTIVGUARD_THRESHOLD", "0.55"))
"""P(vulnerable) decision boundary.  Values >= THRESHOLD raise a VIOLATION.

Lower values increase recall at the cost of precision.  The default 0.55 was
selected empirically against the CodeBERT layer-9 probe (current AUC loaded
at runtime from .activguard/layer_probe_weights.pkl via get_probe_auc()).

Reference: Feng et al., "CodeBERT", arXiv:2002.08155 (2020).
"""

PROBE_INTERVAL: int = int(os.environ.get("ACTIVGUARD_PROBE_INTERVAL", "20"))
"""Number of newly generated tokens between consecutive probe calls.

Smaller values increase detection latency sensitivity but add compute overhead
(each probe call is a CodeBERT forward pass ~10 ms on CPU).
"""

STOP_ON_VIOLATION: bool = os.environ.get("ACTIVGUARD_STOP_ON_VIOLATION", "0") == "1"
"""If True, halt token streaming immediately when the first VIOLATION fires.

Setting this to True provides the earliest-possible interruption but may
truncate syntactically complete — but safe — partial responses.  Defaults to
False so that the full response is always returned alongside the warning.
"""

BANDIT_SECOND_PASS: bool = os.environ.get("ACTIVGUARD_BANDIT_SECOND_PASS", "1") != "0"
"""If True, run Bandit static analysis on the complete output after generation.

Bandit (PyCQA/bandit) provides complementary AST-based detection that catches
vulnerability patterns the probe may miss.  The combined two-pass strategy
improves overall recall (see run_baselines.py for benchmark comparison).
"""

# ---------------------------------------------------------------------------
# Server defaults
# ---------------------------------------------------------------------------

DEFAULT_HOST: str = "0.0.0.0"
"""Network interface for the uvicorn server."""

DEFAULT_PORT: int = 8080
"""Default TCP port.  Override with --port at startup."""

LOG_LEVEL: str = os.environ.get("ACTIVGUARD_LOG_LEVEL", "info")
"""Uvicorn / standard-library logging level."""

# ---------------------------------------------------------------------------
# CodeBERT detector tag (informational, not used at runtime)
# ---------------------------------------------------------------------------

DETECTOR_TAG: str = "codebert-layer9"
"""Human-readable detector identifier emitted in /health and violation events."""

# ---------------------------------------------------------------------------
# HF residual stream probe (true L1 — uses generating model's own hidden states)
# ---------------------------------------------------------------------------

HF_PROBE_MODEL: str = os.environ.get(
    "ACTIVGUARD_HF_PROBE_MODEL",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
)
"""HuggingFace model whose residual stream is probed for vulnerability signals.

This is the true L1 implementation — the probe trained on this model's own
hidden states encodes code using the same representation space as the
generating model, giving model-specific vulnerability signatures.

Reference: Zou et al., "Representation Engineering", arXiv:2310.01405 (2023).
"""

HF_PROBE_WEIGHTS: str = os.environ.get(
    "ACTIVGUARD_HF_PROBE_WEIGHTS",
    "",   # Empty = auto-derive from model slug at startup
)
"""Path to trained HFHiddenProbe weights .pkl.  Empty = auto-derive from model name."""

HF_PROBE_ENABLED: bool = os.environ.get("ACTIVGUARD_HF_PROBE_ENABLED", "1") == "1"
"""Set to '0' to disable the HF residual stream probe (e.g. low-RAM environments)."""

# ---------------------------------------------------------------------------
# Auth model extraction (project-specific IDOR context for L2 RAG)
# ---------------------------------------------------------------------------

AUTH_CONTEXT_PATH: str = os.environ.get("ACTIVGUARD_AUTH_CONTEXT_PATH", "")
"""Path to a Python codebase whose auth model should be indexed into ChromaDB.

If set, AuthExtractor scans this path at startup and adds the extracted
auth/ownership patterns to the RAG `auth_model` collection.  This enables
project-specific IDOR detection — the RAG layer learns which endpoints are
protected and can flag code that bypasses the project's own auth patterns.

Example:
    ACTIVGUARD_AUTH_CONTEXT_PATH=/home/user/myproject python -m proxy
"""
