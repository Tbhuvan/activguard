"""
Train the CodeBERT layer-9 probe for the universal streaming detector.

This probe is the DETECTOR used by UniversalStreamingProbe — it runs on
CodeBERT hidden states extracted from partial generated outputs, independent
of which LLM is generating.

Usage:
    python train_layer_probe.py                    # use existing layer cache
    python train_layer_probe.py --reembed          # re-extract CodeBERT features
    python train_layer_probe.py --layer 9          # default best layer
    python train_layer_probe.py --all-layers       # sweep all 13 layers, pick best

Reference: Feng et al., "CodeBERT", arXiv:2002.08155 (2020).
Reference: Zou et al., "Representation Engineering", arXiv:2310.01405 (2023).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

REDBENCH_DEFAULT = "../redbench/datasets"
LAYER_CACHE_PATH = ".activguard/layer_cache/codebert_layers.npz"
WEIGHTS_OUT = ".activguard/layer_probe_weights.pkl"
N_LAYERS = 13   # embedding + 12 transformer blocks


def load_redbench(datasets_dir: str) -> tuple[list[str], list[int]]:
    """Load code snippets and binary labels from redbench."""
    base = Path(datasets_dir)
    vuln_codes: list[str] = []
    safe_codes: list[str] = []
    for vc in sorted(d.name for d in base.iterdir() if d.is_dir()):
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
    codes = vuln_codes + safe_codes
    labels = [1] * len(vuln_codes) + [0] * len(safe_codes)
    return codes, labels


def extract_codebert_layers(
    codes: list[str],
    cache_path: str,
    force: bool = False,
) -> np.ndarray:
    """Extract per-layer hidden states from CodeBERT for all code snippets.

    Args:
        codes: List of code strings.
        cache_path: Path to .npz cache file.
        force: Ignore cache and re-extract.

    Returns:
        np.ndarray: Shape (n_samples, n_layers, 768).
    """
    cache = Path(cache_path)
    if cache.exists() and not force:
        data = np.load(cache)
        X = data["X"]
        if X.shape[0] == len(codes):
            print(f"[*] Loaded layer cache: {X.shape} from {cache}")
            return X
        print(f"[!] Cache size mismatch ({X.shape[0]} vs {len(codes)}). Re-extracting.")

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("[!] transformers / torch not installed. pip install torch transformers")
        sys.exit(1)

    print("[*] Loading CodeBERT (microsoft/codebert-base)...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    model.eval()

    n = len(codes)
    all_layers = np.zeros((n, N_LAYERS, 768), dtype=np.float32)
    t0 = time.time()

    for i, code in enumerate(codes):
        inputs = tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        for layer_idx, h in enumerate(outputs.hidden_states):
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1)
            all_layers[i, layer_idx] = pooled.squeeze(0).numpy()

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{n} ({elapsed:.0f}s)")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=all_layers)
    print(f"[*] Saved layer cache: {all_layers.shape} -> {cache}")
    return all_layers


def train_layer_probe(
    X_layers: np.ndarray,
    y: np.ndarray,
    layer: int,
    C: float = 1.0,
) -> tuple[Pipeline, dict]:
    """Train a LogisticRegression probe on hidden states from one layer.

    Args:
        X_layers: Shape (n_samples, n_layers, hidden_dim).
        y: Binary labels.
        layer: Which layer index to probe.
        C: Regularisation strength.

    Returns:
        Fitted sklearn Pipeline and CV metrics dict.
    """
    X = X_layers[:, layer, :]
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42,
        )),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")
    acc_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    clf.fit(X, y)
    return clf, {
        "layer": layer,
        "auc_mean": float(auc_scores.mean()),
        "auc_std": float(auc_scores.std()),
        "acc_mean": float(acc_scores.mean()),
        "n_samples": len(y),
        "embed_dim": X.shape[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CodeBERT layer probe for universal streaming detector")
    parser.add_argument("--redbench", default=REDBENCH_DEFAULT)
    parser.add_argument("--cache", default=LAYER_CACHE_PATH)
    parser.add_argument("--out", default=WEIGHTS_OUT)
    parser.add_argument("--layer", type=int, default=9, help="Which CodeBERT layer to use (default: 9)")
    parser.add_argument("--all-layers", action="store_true", help="Sweep all layers and pick best")
    parser.add_argument("--reembed", action="store_true", help="Ignore cache and re-extract features")
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 60)
    print("  CodeBERT Layer Probe — Universal Streaming Detector")
    print("=" * 60)

    codes, labels = load_redbench(args.redbench)
    y = np.array(labels, dtype=int)
    print(f"\n[*] Dataset: {len(codes)} samples ({y.sum()} vuln, {(1-y).sum()} safe)")

    X_layers = extract_codebert_layers(codes, args.cache, force=args.reembed)
    n_layers = X_layers.shape[1]
    print(f"[*] Layer features: {X_layers.shape}")

    if args.all_layers:
        print("\n[*] Sweeping all layers...")
        print(f"  {'Layer':<8} {'AUC mean':<12} {'AUC std':<10} {'Acc mean'}")
        print("  " + "-" * 42)
        best_layer, best_auc = 0, 0.0
        layer_results = []
        for layer_idx in range(n_layers):
            _, metrics = train_layer_probe(X_layers, y, layer_idx, C=args.C)
            layer_results.append(metrics)
            marker = " <- best" if metrics["auc_mean"] > best_auc else ""
            if metrics["auc_mean"] > best_auc:
                best_auc = metrics["auc_mean"]
                best_layer = layer_idx
            print(
                f"  Layer {layer_idx:<4} {metrics['auc_mean']:.4f}       "
                f"{metrics['auc_std']:.4f}     {metrics['acc_mean']:.4f}{marker}"
            )
        print(f"\n[+] Best layer: {best_layer} (AUC={best_auc:.4f})")
        args.layer = best_layer
    else:
        best_layer = args.layer

    print(f"\n[*] Training final probe on layer {best_layer}...")
    clf_pipeline, metrics = train_layer_probe(X_layers, y, best_layer, C=args.C)
    print(f"    AUC:  {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}")
    print(f"    Acc:  {metrics['acc_mean']:.4f}")

    # Full-dataset sanity check
    X_best = X_layers[:, best_layer, :]
    y_proba = clf_pipeline.predict_proba(X_best)[:, 1]
    y_pred = clf_pipeline.predict(X_best)
    print(f"\n[*] Sanity check (train set):")
    print(classification_report(y, y_pred, target_names=["safe", "vulnerable"]))
    print(f"  ROC-AUC (train): {roc_auc_score(y, y_proba):.4f}")

    # Save — extract scaler and clf from pipeline for streaming probe compatibility
    scaler = clf_pipeline.named_steps["scaler"]
    raw_clf = clf_pipeline.named_steps["clf"]
    coef = raw_clf.coef_[0]
    top5 = np.argsort(np.abs(coef))[-5:][::-1]
    print(f"\n[*] Top-5 discriminative dims at layer {best_layer}:")
    for rank, dim in enumerate(top5, 1):
        print(f"    #{rank}: dim {dim:4d}  coef={coef[dim]:+.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clf": raw_clf,
        "scaler": scaler,
        "embed_dim": X_best.shape[1],
        "best_layer": best_layer,
        "n_layers": n_layers,
        "encoder": "codebert-base",
        "auc_cv": metrics["auc_mean"],
        "n_samples": len(y),
    }
    with open(out_path, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"\n[+] Probe saved to {out_path.resolve()}")
    print(f"\n[*] To use: python -m probe.universal_streaming_probe --weights {args.out} --models dolphin3:8b qwen3-coder:30b")


if __name__ == "__main__":
    main()
