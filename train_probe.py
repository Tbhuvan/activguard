"""
Train the Layer 1 residual stream probe on redbench data.

Usage:
    python train_probe.py                          # default: dolphin3:8b
    python train_probe.py --model qwen3:8b
    python train_probe.py --model nous-hermes2:10.7b
    python train_probe.py --redbench ../redbench/datasets

Methodology:
    1. Load all 50 vulnerable samples from redbench (code field).
    2. Load 50 safe counterparts (fix field from same samples).
    3. Call Ollama /api/embed for each snippet → 4096-dim residual stream.
    4. Train LogisticRegression with L2 regularisation.
    5. 5-fold cross-validation → report per-fold AUC and mean accuracy.
    6. Save probe weights to .activguard/probe_weights.pkl

Reference: Zou et al., "Representation Engineering: A Top-Down Approach to
AI Transparency", arXiv:2310.01405 (2023).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from probe.residual_stream_probe import ResidualStreamProbe

REDBENCH_DEFAULT = "../redbench/datasets"
WEIGHTS_OUT = ".activguard/probe_weights.pkl"
EMBED_CACHE = ".activguard/embed_cache.npz"


def load_redbench(datasets_dir: str) -> tuple[list[str], list[str], list[int]]:
    """Load all redbench samples.

    Returns:
        codes: Code snippets (vulnerable + safe interleaved).
        labels: 1 = vulnerable, 0 = safe.
        sources: Source identifier strings.
    """
    base = Path(datasets_dir)
    vuln_codes: list[str] = []
    safe_codes: list[str] = []
    sources: list[str] = []

    vuln_classes = sorted([d.name for d in base.iterdir() if d.is_dir()])
    for vc in vuln_classes:
        for fname in ("samples.jsonl", "samples_generated.jsonl"):
            jsonl = base / vc / fname
            if not jsonl.exists():
                continue
            with open(jsonl, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sample = json.loads(line)
                    vuln_codes.append(sample["code"])
                    safe_codes.append(sample["fix"])
                    sources.append(f"{vc}/{sample['id']}")

    codes = vuln_codes + safe_codes
    labels = [1] * len(vuln_codes) + [0] * len(safe_codes)
    return codes, labels, sources + [s + "_safe" for s in sources]


def load_or_embed(
    probe: ResidualStreamProbe,
    codes: list[str],
    cache_path: str,
    force: bool = False,
) -> np.ndarray:
    """Load embeddings from cache or compute and cache them."""
    cache = Path(cache_path)
    if cache.exists() and not force:
        print(f"[*] Loading cached embeddings from {cache}")
        data = np.load(cache)
        X = data["X"]
        n_cached = data["n_samples"].item()
        if n_cached == len(codes) and X.shape[0] == len(codes):
            print(f"    {X.shape[0]} embeddings, dim={X.shape[1]}")
            return X
        print(f"[!] Cache mismatch (expected {len(codes)}, got {n_cached}). Re-embedding.")

    print(f"[*] Embedding {len(codes)} snippets with {probe.model}...")
    print(f"    This calls Ollama {len(codes)} times (~{len(codes) * 0.5:.0f}s expected)")
    t0 = time.time()
    X = probe.embed_batch(codes, delay_s=0.05)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s — dim={X.shape[1]}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, n_samples=np.array(len(codes)))
    print(f"    Cached to {cache}")
    return X


def cross_validate(X: np.ndarray, y: np.ndarray) -> dict:
    """5-fold stratified cross-validation with AUC and accuracy."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                              solver="lbfgs", random_state=42)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="roc_auc")
    acc_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="accuracy")

    return {
        "auc_mean": float(auc_scores.mean()),
        "auc_std": float(auc_scores.std()),
        "acc_mean": float(acc_scores.mean()),
        "acc_std": float(acc_scores.std()),
        "auc_per_fold": auc_scores.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ActivGuard Layer 1 residual stream probe")
    parser.add_argument("--model", default="dolphin3:8b", help="Ollama model for embedding")
    parser.add_argument("--redbench", default=REDBENCH_DEFAULT, help="Path to redbench datasets/")
    parser.add_argument("--out", default=WEIGHTS_OUT, help="Output path for probe weights")
    parser.add_argument("--force-embed", action="store_true", help="Ignore embedding cache")
    parser.add_argument("--threshold", type=float, default=0.55, help="Decision threshold")
    args = parser.parse_args()

    print("=" * 60)
    print("  ActivGuard Layer 1 — Residual Stream Probe Training")
    print(f"  Model: {args.model}")
    print("=" * 60)

    # --- Step 1: Load data ---
    codes, labels, sources = load_redbench(args.redbench)
    y = np.array(labels, dtype=int)
    print(f"\n[*] Dataset: {len(codes)} samples ({y.sum()} vulnerable, {(1-y).sum()} safe)")
    print(f"    Classes: {sorted(set(s.split('/')[0] for s in sources[:len(y)//2]))}")

    # --- Step 2: Embed ---
    # Fresh probe for training — weights_path points to output location.
    # Reset any loaded state so stale pkl from a previous run is ignored.
    probe = ResidualStreamProbe(model=args.model, threshold=args.threshold,
                                weights_path=args.out)
    probe._clf = None
    probe._scaler = None
    probe._is_trained = False
    model_tag = args.model.replace(":", "_").replace(".", "_")
    cache_file = f".activguard/embed_cache_{model_tag}.npz"
    X = load_or_embed(probe, codes, cache_file, force=args.force_embed)

    # --- Step 3: Cross-validate ---
    print("\n[*] 5-fold stratified cross-validation...")
    cv = cross_validate(X, y)
    print(f"\n  ROC-AUC:  {cv['auc_mean']:.4f} +/- {cv['auc_std']:.4f}")
    print(f"  Accuracy: {cv['acc_mean']:.4f} +/- {cv['acc_std']:.4f}")
    print(f"  Per-fold AUC: {[f'{a:.3f}' for a in cv['auc_per_fold']]}")

    # --- Step 4: Train final probe on all data ---
    print("\n[*] Training final probe on full dataset...")
    metrics = probe.fit(X, y)
    print(f"    Train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"    Embed dim:      {metrics['embed_dim']}")

    # --- Step 5: Held-out sanity check ---
    print("\n[*] Probe sanity check (training set, for diagnostics):")
    scaler = probe._scaler
    clf = probe._clf
    assert scaler is not None and clf is not None
    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled)
    y_proba = clf.predict_proba(X_scaled)[:, 1]
    print(classification_report(y, y_pred, target_names=["safe", "vulnerable"]))
    print(f"  ROC-AUC (train): {roc_auc_score(y, y_proba):.4f}")

    # --- Step 6: Show most discriminative dimensions ---
    coef = clf.coef_[0]
    top5 = np.argsort(np.abs(coef))[-5:][::-1]
    print(f"\n[*] Top-5 most discriminative residual stream dimensions:")
    for rank, dim in enumerate(top5, 1):
        print(f"    #{rank}: dim {dim:4d}  coef={coef[dim]:+.4f}")

    # --- Step 7: Save ---
    out_path = probe.save(args.out)
    print(f"\n[+] Probe saved to {out_path}")
    print(f"\n[*] CV Summary:")
    print(f"    AUC  {cv['auc_mean']:.3f} +/- {cv['auc_std']:.3f}")
    print(f"    Acc  {cv['acc_mean']:.3f} +/- {cv['acc_std']:.3f}")
    print("\n[+] Run pipeline.py to use the probe as Layer 1.")


if __name__ == "__main__":
    main()
