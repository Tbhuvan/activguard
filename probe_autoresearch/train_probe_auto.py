"""
train_probe_auto.py — The file the agent edits to improve val_auc.

The agent should try to beat the best known val_auc (printed at startup
from .activguard/layer_probe_weights.pkl — never hardcoded here).
Everything in this file is fair game EXCEPT the final three lines
(load_data, evaluate_auc, print_summary calls).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path


def _current_best_auc() -> float:
    """Read current best probe AUC from saved weights — no hardcoding."""
    for candidate in (
        Path(__file__).parent.parent / ".activguard" / "layer_probe_weights.pkl",
        Path(".activguard/layer_probe_weights.pkl"),
    ):
        if candidate.exists():
            try:
                with open(candidate, "rb") as f:
                    return float(pickle.load(f).get("auc_cv", 0.0))
            except Exception:
                pass
    return 0.0


_CURRENT_BEST = _current_best_auc()
print(f"[autoresearch] Current best AUC to beat: {_CURRENT_BEST:.4f} (from layer_probe_weights.pkl)")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent.parent))
from probe_autoresearch.prepare_probe import evaluate_auc, load_data, print_summary

# ---------------------------------------------------------------------------
# Load data — do not change this
# ---------------------------------------------------------------------------
data = load_data()
y = data["y"]

# ---------------------------------------------------------------------------
# EXPERIMENT: modify everything below this line
# ---------------------------------------------------------------------------

# Feature selection: choose which embedding to use
# Options:
#   data["X_dolphin"]       — (100, 4096)  dolphin3:8b final layer
#   data["X_codebert"]      — (100, 768)   CodeBERT layer 9 (best single layer)
#   data["X_layers"][:, N, :] — (100, 768) CodeBERT layer N (0-12)
#   np.hstack([...])        — concatenate multiple sources
X = data["X_codebert"]   # baseline: CodeBERT layer 9

# Model: replace with anything sklearn-compatible that has predict_proba
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )),
])

model_desc = "LogReg C=1.0 | CodeBERT layer-9"

# ---------------------------------------------------------------------------
# Evaluate — do not change these three lines
# ---------------------------------------------------------------------------
val_auc = evaluate_auc(model, X, y)
print_summary(val_auc, model_desc, {"X_shape": str(X.shape)})
