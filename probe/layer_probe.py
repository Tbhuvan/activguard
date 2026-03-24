"""
Per-layer Residual Stream Probe — Layer-wise Vulnerability Signal Analysis.

Downloads microsoft/codebert-base (125M params, 12 layers, ~500MB) and
trains a linear probe at each transformer layer to find which layer carries
the strongest vulnerability signal in activation space.

Why CodeBERT?
  Decoder-only models (Llama, Qwen) require llama-cpp-python C API hooks
  for per-layer access.  CodeBERT is a BERT-style encoder trained on code
  (CodeSearchNet, 6 languages) that natively exposes all 12 layer hidden
  states via output_hidden_states=True.  The probe methodology is identical
  to what applies to Llama-family decoders — only the extraction mechanism
  differs.

Research question addressed:
  RQ1: Which transformer layers carry strongest vulnerability signal?
  Hypothesis (from Zou et al. 2023): middle layers (6–9 of 12) are
  most discriminative for semantic properties; early layers encode
  syntax, late layers encode task-specific representations.

Reference: Zou et al., "Representation Engineering: A Top-Down Approach to
AI Transparency", arXiv:2310.01405 (2023).
Reference: Feng et al., "CodeBERT: A Pre-Trained Model for Programming and
Natural Languages", arXiv:2002.08155 (2020).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

CODEBERT_MODEL = "microsoft/codebert-base"
LAYER_CACHE_DIR = ".activguard/layer_cache"
N_LAYERS = 13  # 12 transformer blocks + embedding layer (layer 0)


def load_redbench(datasets_dir: str) -> tuple[list[str], np.ndarray]:
    """Load redbench code snippets and binary labels."""
    base = Path(datasets_dir)
    codes: list[str] = []
    labels: list[int] = []
    for vc in ["sqli", "idor", "ssrf", "auth_bypass", "path_traversal"]:
        jsonl = base / vc / "samples.jsonl"
        if not jsonl.exists():
            continue
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                s = json.loads(line.strip())
                codes.append(s["code"])
                labels.append(1)
                codes.append(s["fix"])
                labels.append(0)
    return codes, np.array(labels, dtype=int)


def extract_all_layers(
    codes: list[str],
    model_name: str = CODEBERT_MODEL,
    max_length: int = 512,
    cache_dir: str = LAYER_CACHE_DIR,
    force: bool = False,
) -> np.ndarray:
    """Extract hidden states at every transformer layer for each code snippet.

    Uses mean-pooling over token positions (same as the Ollama /api/embed
    endpoint uses for final-layer embeddings).

    Args:
        codes: List of code strings.
        model_name: HuggingFace model ID.
        max_length: Token sequence truncation length.
        cache_dir: Directory for caching extracted features.
        force: If True, recompute even if cache exists.

    Returns:
        np.ndarray: Shape (n_samples, n_layers, hidden_dim).
                    hidden_dim = 768 for codebert-base.
    """
    cache_path = Path(cache_dir) / "codebert_layers.npz"
    if cache_path.exists() and not force:
        print(f"[*] Loading layer cache from {cache_path}")
        data = np.load(cache_path)
        X = data["X"]
        if X.shape[0] == len(codes):
            print(f"    Shape: {X.shape}  (samples x layers x hidden_dim)")
            return X
        print("[!] Cache shape mismatch — recomputing.")

    print(f"[*] Downloading / loading {model_name}...")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        _TORCH = True
    except ImportError:
        _TORCH = False

    if not _TORCH:
        raise RuntimeError(
            "torch is not installed. Install it with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    hidden_dim = model.config.hidden_size
    all_layers = np.zeros((len(codes), n_layers, hidden_dim), dtype=np.float32)

    print(f"[*] Extracting {n_layers}-layer hidden states for {len(codes)} snippets...")
    with torch.no_grad():
        for i, code in enumerate(codes):
            inputs = tokenizer(
                code,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            outputs = model(**inputs)
            # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
            for layer_idx, hs in enumerate(outputs.hidden_states):
                # Mean-pool over sequence length
                mean_pooled = hs[0].mean(dim=0).numpy()  # (hidden_dim,)
                all_layers[i, layer_idx] = mean_pooled
            if (i + 1) % 20 == 0:
                print(f"    {i + 1}/{len(codes)}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=all_layers)
    print(f"[+] Cached layer features to {cache_path}")
    return all_layers


def probe_all_layers(
    X_layers: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
) -> list[dict]:
    """Train and cross-validate a linear probe at each layer.

    Args:
        X_layers: Shape (n_samples, n_layers, hidden_dim).
        y: Binary labels (1 = vulnerable, 0 = safe).
        n_folds: Cross-validation folds.

    Returns:
        list[dict]: One entry per layer with auc_mean, auc_std, acc_mean.
    """
    n_layers = X_layers.shape[1]
    results: list[dict] = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for layer_idx in range(n_layers):
        X = X_layers[:, layer_idx, :]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42,
        )
        auc_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="roc_auc")
        acc_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="accuracy")
        results.append({
            "layer": layer_idx,
            "layer_name": "embedding" if layer_idx == 0 else f"transformer_{layer_idx}",
            "auc_mean": float(auc_scores.mean()),
            "auc_std": float(auc_scores.std()),
            "acc_mean": float(acc_scores.mean()),
            "acc_std": float(acc_scores.std()),
        })
        bar = "#" * int(auc_scores.mean() * 40)
        print(
            f"  Layer {layer_idx:2d} ({results[-1]['layer_name']:20s})  "
            f"AUC={auc_scores.mean():.3f}+/-{auc_scores.std():.3f}  "
            f"Acc={acc_scores.mean():.3f}  |{bar}"
        )
    return results


def find_best_layer(results: list[dict]) -> dict:
    """Return the layer with highest mean ROC-AUC."""
    return max(results, key=lambda r: r["auc_mean"])


def run(
    redbench_dir: str = "../redbench/datasets",
    model_name: str = CODEBERT_MODEL,
    force: bool = False,
) -> dict:
    """Full per-layer probe pipeline.

    Returns:
        dict: best_layer, all_results, model_name.
    """
    print("=" * 64)
    print("  ActivGuard Layer 1 — Per-Layer Residual Stream Analysis")
    print(f"  Model: {model_name}")
    print("=" * 64)

    codes, y = load_redbench(redbench_dir)
    print(f"[*] Dataset: {len(codes)} samples ({y.sum()} vuln, {(~y.astype(bool)).sum()} safe)")

    X_layers = extract_all_layers(codes, model_name=model_name, force=force)

    print(f"\n[*] Probing {X_layers.shape[1]} layers (5-fold CV)...")
    results = probe_all_layers(X_layers, y)

    best = find_best_layer(results)
    print(f"\n[+] Best layer: {best['layer']} ({best['layer_name']})")
    print(f"    AUC = {best['auc_mean']:.4f} +/- {best['auc_std']:.4f}")
    print(f"    Acc = {best['acc_mean']:.4f}")

    print("\n[*] Layer AUC ranking (top 5):")
    sorted_results = sorted(results, key=lambda r: r["auc_mean"], reverse=True)
    for rank, r in enumerate(sorted_results[:5], 1):
        print(f"    #{rank}  Layer {r['layer']:2d} ({r['layer_name']:20s})  AUC={r['auc_mean']:.4f}")

    return {"best_layer": best, "all_results": results, "model": model_name}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Per-layer residual stream probe")
    parser.add_argument("--redbench", default="../redbench/datasets")
    parser.add_argument("--model", default=CODEBERT_MODEL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(redbench_dir=args.redbench, model_name=args.model, force=args.force)
