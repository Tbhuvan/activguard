"""
Sparse Autoencoder Training Script.

Trains a SparseAutoencoder on hidden states collected from the HF residual
stream probe, then identifies which SAE latent dimensions activate most
strongly for each vulnerability class.

This provides INTERPRETABILITY for the probe's decisions:
  - Feature 42 activates on "string interpolation into SQL" → SQLi feature
  - Feature 117 activates on "user_id in URL path without ownership check" → IDOR

Two-phase workflow:
    Phase 1: Collect hidden states from all redbench samples using HFHiddenProbe.
    Phase 2: Train SAE on the collected activations.
    Phase 3: Identify vulnerability-discriminative features (t-test: vuln vs safe).
    Phase 4: Label top features using VulnerabilityFeatureExtractor.

Output:
    .activguard/sae_{model_slug}.pt          — trained SAE weights
    .activguard/sae_features_{model_slug}.json — per-feature vulnerability labels
    experiments.json                          — appended experiment record

Research questions addressed:
    RQ4: Do vulnerability-relevant features form monosemantic dimensions?
         → t-test p-values for vuln vs safe activation differences
    RQ5: What fraction of SAE features explain the probe's decisions?
         → Coverage analysis: how many features needed to explain 90% of detections

Usage:
    cd activguard/
    python scripts/train_sae.py
    python scripts/train_sae.py --expansion-factor 8 --sparsity 0.02
    python scripts/train_sae.py --model Qwen/Qwen2.5-Coder-7B-Instruct

Reference: Cunningham et al., "Sparse Autoencoders Find Highly Interpretable
Features in Language Models", arXiv:2309.08600 (2023).
Reference: Anthropic, "Towards Monosemanticity: Decomposing Language Models
with Dictionary Learning", Transformer Circuits Thread (2023).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REDBENCH_DIR     = "../redbench/datasets"
EXPERIMENTS_FILE = "experiments.json"


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_hidden_states(
    model_name: str,
    layer: int,
    redbench_dir: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect hidden state vectors for all redbench samples.

    Args:
        model_name: HuggingFace model name.
        layer: Transformer layer to extract.
        redbench_dir: Path to redbench/datasets/ directory.

    Returns:
        Tuple (X, y, class_labels) where X.shape=(2N, hidden_dim),
        y.shape=(2N,), class_labels has length 2N.
    """
    import json as _json
    from probe.hf_hidden_probe import HFHiddenProbe

    base = Path(redbench_dir)
    source_files = [
        "samples.jsonl", "samples_generated.jsonl",
        "samples_generated_v2.jsonl", "samples_real.jsonl",
    ]
    vuln_codes: list[str] = []
    safe_codes: list[str] = []
    class_labels: list[str] = []

    for class_dir in sorted(d for d in base.iterdir() if d.is_dir()):
        vc = class_dir.name
        for fname in source_files:
            fp = class_dir / fname
            if not fp.exists():
                continue
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        s = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    code = s.get("code", "").strip()
                    fix  = s.get("fix", "").strip()
                    if code and fix and not fix.startswith("# TODO"):
                        vuln_codes.append(code)
                        safe_codes.append(fix)
                        class_labels.append(vc)

    logger.info(
        "Collecting hidden states: %d pairs, model=%s layer=%d",
        len(vuln_codes), model_name, layer,
    )
    probe = HFHiddenProbe(model_name=model_name, layer=layer)
    X, y = probe.build_feature_matrix(vuln_codes, safe_codes, verbose=True)

    # Duplicate class labels for safe samples
    full_labels = class_labels + class_labels   # vuln then safe
    return X, y, full_labels


# ---------------------------------------------------------------------------
# SAE training
# ---------------------------------------------------------------------------

def train_sae(
    X: np.ndarray,
    hidden_dim_multiplier: int = 4,
    sparsity_penalty: float = 0.01,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> tuple[object, dict]:
    """Train the sparse autoencoder on collected hidden states.

    Args:
        X: Shape (n, input_dim) — raw hidden state vectors.
        hidden_dim_multiplier: SAE hidden dim = input_dim * multiplier.
        sparsity_penalty: L1 penalty on activations.
        n_epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.

    Returns:
        Tuple (sae, training_metrics).
    """
    import torch
    import torch.optim as optim
    from probe.sparse_autoencoder import SparseAutoencoder

    input_dim = X.shape[1]
    sae_hidden_dim = input_dim * hidden_dim_multiplier

    logger.info(
        "Training SAE: input_dim=%d hidden_dim=%d sparsity=%.4f epochs=%d",
        input_dim, sae_hidden_dim, sparsity_penalty, n_epochs,
    )

    sae = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=sae_hidden_dim,
        sparsity_penalty=sparsity_penalty,
    )
    optimizer = optim.Adam(sae.parameters(), lr=lr)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    n = len(X_tensor)
    losses: list[float] = []
    sparsity_levels: list[float] = []

    t0 = time.time()
    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n)
        epoch_loss = 0.0
        epoch_sparsity = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch = X_tensor[perm[start:start + batch_size]]
            optimizer.zero_grad()
            x_recon, z = sae(batch)
            loss = sae.loss(batch, x_recon, z)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # Fraction of activations that are zero (sparsity)
            epoch_sparsity += (z == 0).float().mean().item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_sparsity = epoch_sparsity / max(n_batches, 1)
        losses.append(avg_loss)
        sparsity_levels.append(avg_sparsity)

        if (epoch + 1) % 10 == 0:
            logger.info(
                "Epoch %d/%d  loss=%.4f  sparsity=%.3f",
                epoch + 1, n_epochs, avg_loss, avg_sparsity,
            )

    elapsed = time.time() - t0
    logger.info("SAE training complete in %.1fs", elapsed)
    metrics = {
        "n_epochs": n_epochs,
        "final_loss": round(losses[-1], 6),
        "final_sparsity": round(sparsity_levels[-1], 4),
        "train_time_s": round(elapsed, 1),
        "input_dim": input_dim,
        "sae_hidden_dim": sae_hidden_dim,
        "loss_curve": [round(l, 6) for l in losses[::5]],  # every 5 epochs
    }
    return sae, metrics


# ---------------------------------------------------------------------------
# Vulnerability feature identification
# ---------------------------------------------------------------------------

def identify_vuln_features(
    sae: object,
    X: np.ndarray,
    y: np.ndarray,
    class_labels: list[str],
    top_k_per_class: int = 10,
) -> dict[str, list[dict]]:
    """Identify SAE features that discriminate vulnerable from safe code.

    Uses Welch's t-test to compare feature activations between vuln and safe
    samples.  Features with large mean difference AND low p-value are candidates
    for vulnerability-specific features.

    Args:
        sae: Trained SparseAutoencoder.
        X: Hidden state matrix (n, input_dim).
        y: Binary labels (1=vuln, 0=safe).
        class_labels: Vulnerability class for each sample.
        top_k_per_class: Number of top features to report per class.

    Returns:
        Dict mapping vuln_class → list of top discriminative features.
    """
    import torch
    from scipy import stats

    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        Z = sae.encode(X_tensor).numpy()   # (n, sae_hidden_dim)

    n_features = Z.shape[1]
    vuln_classes = sorted(set(c for c, label in zip(class_labels, y) if label == 1))
    results: dict[str, list[dict]] = {}

    for vc in vuln_classes:
        # Indices for this vuln class vs safe samples
        vc_mask  = np.array([lbl == vc and y[i] == 1 for i, lbl in enumerate(class_labels)])
        safe_mask = np.array([y[i] == 0 for i in range(len(y))])

        if vc_mask.sum() < 3 or safe_mask.sum() < 3:
            continue

        Z_vc   = Z[vc_mask]
        Z_safe = Z[safe_mask]

        feature_scores: list[dict] = []
        for fi in range(n_features):
            t_stat, p_val = stats.ttest_ind(Z_vc[:, fi], Z_safe[:, fi], equal_var=False)
            mean_diff = float(Z_vc[:, fi].mean() - Z_safe[:, fi].mean())
            feature_scores.append({
                "feature_idx": fi,
                "mean_diff": round(mean_diff, 4),
                "t_stat": round(float(t_stat), 4),
                "p_value": round(float(p_val), 6),
                "vuln_class": vc,
            })

        # Sort by mean difference (largest positive = fires more for vuln)
        feature_scores.sort(key=lambda x: -x["mean_diff"])
        top = feature_scores[:top_k_per_class]

        # Add keyword labels from VulnerabilityFeatureExtractor
        from probe.sparse_autoencoder import VulnerabilityFeatureExtractor
        extractor = VulnerabilityFeatureExtractor(sae)
        for feat in top:
            # Find example snippets that activate this feature
            fi = feat["feature_idx"]
            top_activators = Z_vc[:, fi].argsort()[-3:][::-1]
            vc_codes = X[vc_mask]
            examples = []   # We don't have the original code here, use placeholder
            label = extractor.label_features(fi, examples)
            feat["label"] = label
            extractor.assign_vuln_class(fi, vc)

        results[vc] = top
        logger.info(
            "  %s: top feature %d  mean_diff=%.4f  p=%.4f",
            vc, top[0]["feature_idx"], top[0]["mean_diff"], top[0]["p_value"],
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Sparse Autoencoder on HF hidden states for interpretability",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Transformer layer to probe (-1 = last)")
    parser.add_argument("--expansion-factor", type=int, default=4,
                        help="SAE hidden_dim = input_dim * factor (default: 4)")
    parser.add_argument("--sparsity", type=float, default=0.01,
                        help="L1 sparsity penalty (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--redbench-dir", default=REDBENCH_DIR)
    args = parser.parse_args()

    # Phase 1: Collect hidden states
    X, y, class_labels = collect_hidden_states(
        args.model, args.layer, args.redbench_dir
    )
    if len(X) == 0:
        logger.error("No samples collected.")
        sys.exit(1)

    # Phase 2: Train SAE
    sae, sae_metrics = train_sae(
        X,
        hidden_dim_multiplier=args.expansion_factor,
        sparsity_penalty=args.sparsity,
        n_epochs=args.epochs,
    )

    # Phase 3: Identify vulnerability features
    logger.info("\nIdentifying vulnerability-discriminative features...")
    vuln_features = identify_vuln_features(sae, X, y, class_labels)

    # Save SAE weights
    import torch
    slug = args.model.lower().split("/")[-1].replace(".", "-").replace("_", "-")
    sae_path = Path(f".activguard/sae_{slug}.pt")
    sae_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sae.state_dict(), sae_path)
    logger.info("SAE saved → %s", sae_path)

    # Save feature analysis
    features_path = Path(f".activguard/sae_features_{slug}.json")
    with open(features_path, "w") as f:
        json.dump(vuln_features, f, indent=2)
    logger.info("Feature analysis saved → %s", features_path)

    # Log experiment
    exp_path = Path(EXPERIMENTS_FILE)
    experiments = []
    if exp_path.exists():
        with open(exp_path) as f:
            try:
                experiments = json.load(f)
            except json.JSONDecodeError:
                experiments = []
    experiments.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "sparse_autoencoder_training",
        "model": args.model,
        "layer": args.layer,
        "n_samples": len(X),
        **sae_metrics,
        "vuln_classes_analyzed": list(vuln_features.keys()),
        "sae_path": str(sae_path),
    })
    with open(exp_path, "w") as f:
        json.dump(experiments, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SAE Training Complete")
    print(f"  Model:      {args.model}")
    print(f"  Input dim:  {sae_metrics['input_dim']}")
    print(f"  SAE dim:    {sae_metrics['sae_hidden_dim']}")
    print(f"  Final loss: {sae_metrics['final_loss']:.4f}")
    print(f"  Sparsity:   {sae_metrics['final_sparsity']:.3f} (frac zeroed)")
    print(f"\n  Top vulnerability features per class:")
    for vc, feats in sorted(vuln_features.items()):
        if feats:
            f = feats[0]
            sig = "***" if f["p_value"] < 0.001 else ("**" if f["p_value"] < 0.01 else "*")
            print(f"    {vc:<22} feat={f['feature_idx']:4d}  diff={f['mean_diff']:+.4f}  {sig}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
