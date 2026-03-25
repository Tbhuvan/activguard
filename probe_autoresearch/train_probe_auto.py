"""
train_probe_auto.py — The file the agent edits to improve val_auc.

Baseline: LogisticRegression on dolphin3:8b final-layer embeddings → AUC 0.644
          LogisticRegression on CodeBERT layer-9 embeddings → AUC 0.900

The agent should try to beat the best known val_auc.
Everything in this file is fair game EXCEPT the final three lines
(load_data, evaluate_auc, print_summary calls).

Current best: 0.9001 (CodeBERT layer-9, LogisticRegression C=1.0)
"""

from __future__ import annotations

import sys
from pathlib import Path

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
