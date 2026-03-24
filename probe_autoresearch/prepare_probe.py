"""
prepare_probe.py — Fixed constants, data loading, and evaluation harness.

DO NOT MODIFY THIS FILE. The agent only modifies train_probe_auto.py.

Mirrors the role of prepare.py in Karpathy's autoresearch repo:
  - Loads pre-computed embeddings from cache (no LLM calls at experiment time)
  - Provides the single evaluation function the agent optimises against
  - Exports: load_data(), evaluate_auc()

Embedding sources available:
  - dolphin3:8b  → EMBED_DOLPHIN  shape (100, 4096)  final-layer residual stream
  - CodeBERT L9  → EMBED_CODEBERT shape (100, 768)   layer-9 hidden state (best per-layer)
  - CodeBERT all → LAYERS_CODEBERT shape (100, 13, 768) all 13 layers

Labels: 1 = vulnerable, 0 = safe  (50 each, from redbench)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / ".activguard"

EMBED_DOLPHIN_PATH   = CACHE_DIR / "embed_cache_dolphin3_8b.npz"
LAYERS_CODEBERT_PATH = CACHE_DIR / "layer_cache" / "codebert_layers.npz"

N_FOLDS   = 5
N_SAMPLES = 100   # 50 vuln + 50 safe


def load_data() -> dict:
    """Load all available embedding sources.

    Returns dict with keys:
      X_dolphin   — (100, 4096) final-layer dolphin3:8b embeddings
      X_codebert  — (100, 768)  CodeBERT layer-9 (best single layer)
      X_layers    — (100, 13, 768) all CodeBERT layers
      y           — (100,) binary labels
    """
    out: dict = {}

    # dolphin3:8b final-layer
    if EMBED_DOLPHIN_PATH.exists():
        d = np.load(EMBED_DOLPHIN_PATH)
        out["X_dolphin"] = d["X"].astype(np.float32)

    # CodeBERT per-layer
    if LAYERS_CODEBERT_PATH.exists():
        d = np.load(LAYERS_CODEBERT_PATH)
        layers = d["X"].astype(np.float32)          # (100, 13, 768)
        out["X_layers"] = layers
        out["X_codebert"] = layers[:, 9, :]         # layer 9 = best per analysis

    n = N_SAMPLES
    out["y"] = np.array([1] * (n // 2) + [0] * (n // 2), dtype=int)

    return out


def evaluate_auc(model, X: np.ndarray, y: np.ndarray) -> float:
    """5-fold stratified cross-validated ROC-AUC. Higher is better.

    This is the single metric the agent optimises.  Analogous to val_bpb
    in Karpathy's autoresearch (but higher = better here).

    Args:
        model: sklearn-compatible estimator with fit/predict_proba.
        X:     Feature matrix (n_samples, n_features).
        y:     Binary labels.

    Returns:
        float: Mean CV AUC across 5 folds.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
    return float(scores.mean())


def print_summary(val_auc: float, model_desc: str, extras: dict | None = None) -> None:
    """Print results in a format the autoresearch loop can grep."""
    print("---")
    print(f"val_auc:     {val_auc:.6f}")
    print(f"model_desc:  {model_desc}")
    if extras:
        for k, v in extras.items():
            print(f"{k}:  {v}")
    print("---")
