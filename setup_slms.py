"""
SLM Setup and Per-Layer Probe Pipeline.

Downloads small language models (SLMs) from HuggingFace and trains a
per-layer activation probe on each.  Enables cross-architecture comparison
of vulnerability signal depth — directly addressing RQ1 and RQ2.

Models selected criteria:
  1. <= 4GB float16 RAM — runs on 32GB machine with headroom
  2. Code-focused or permissive enough to generate vulnerable patterns
  3. Different architectures — Qwen2.5, Phi-3, DeepSeek — for RQ2

Usage:
    python setup_slms.py --list                # show available models
    python setup_slms.py --pull qwen-coder-1.5b
    python setup_slms.py --train-probe qwen-coder-1.5b --redbench ../redbench/datasets
    python setup_slms.py --run-all-probes      # train probes for all pulled models
    python setup_slms.py --stream qwen-coder-1.5b --prompt "Write vulnerable SQL query"

Reference: Zou et al., "Representation Engineering", arXiv:2310.01405 (2023).
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SLM registry — tested on AMD 780M (32GB unified RAM, CPU-only)
# ---------------------------------------------------------------------------

SLM_REGISTRY: dict[str, dict[str, Any]] = {
    # Fastest — great for iteration (~3GB float16)
    "qwen-coder-1.5b": {
        "hf_id": "Qwen/Qwen2.5-Coder-1.5B",           # base model: no safety filters
        "size_gb": 3.0,
        "n_layers": 28,
        "hidden_dim": 1536,
        "hook_layer_frac": 0.75,
        "notes": "Qwen2.5-Coder base model. Completes vulnerable code without refusals. Fast on CPU.",
        "ollama_tag": None,
    },
    # Best quality/size balance (~6GB float16)
    "qwen-coder-3b": {
        "hf_id": "Qwen/Qwen2.5-Coder-3B",              # base model
        "size_gb": 6.0,
        "n_layers": 36,
        "hidden_dim": 2048,
        "hook_layer_frac": 0.75,
        "notes": "Qwen2.5-Coder 3B base. Strong code generation, no instruction safety filters.",
        "ollama_tag": None,
    },
    # Microsoft Phi — different architecture for RQ2 (~7.6GB float16)
    "phi-3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "size_gb": 7.6,
        "n_layers": 32,
        "hidden_dim": 3072,
        "hook_layer_frac": 0.75,
        "notes": "Phi-3 architecture. Instruction-tuned but generally completes code patterns.",
        "ollama_tag": "phi3:mini",
    },
    # DeepSeek — ultra small, fastest for testing (~2.6GB float16)
    "deepseek-coder-1.3b": {
        "hf_id": "deepseek-ai/deepseek-coder-1.3b-base",
        "size_gb": 2.6,
        "n_layers": 24,
        "hidden_dim": 2048,
        "hook_layer_frac": 0.75,
        "notes": "DeepSeek-Coder 1.3B base. No safety filters. Very fast on CPU. Good for rapid iteration.",
        "ollama_tag": None,
    },
    # Matches dolphin3:8b architecture — direct Ollama comparison (~16GB float16)
    "dolphin-llama3-8b": {
        "hf_id": "cognitivecomputations/dolphin-2.9-llama3-8b",
        "size_gb": 16.0,
        "n_layers": 32,
        "hidden_dim": 4096,
        "hook_layer_frac": 0.75,
        "notes": "Uncensored Llama3.1 8B. Matches dolphin3:8b Ollama model — enables direct comparison of "
                 "Ollama final-layer probe vs HF native per-layer probe on identical model weights.",
        "ollama_tag": "dolphin3:8b",
    },
}


def list_models() -> None:
    """Print the SLM registry with RAM requirements and notes."""
    print("\nAvailable SLMs for per-layer probe training:")
    print(f"  {'Tag':<22} {'HF Model':<45} {'RAM':<8} {'Layers':<8} {'Notes'[:30]}")
    print("  " + "-" * 100)
    for tag, info in SLM_REGISTRY.items():
        print(
            f"  {tag:<22} {info['hf_id']:<45} "
            f"{info['size_gb']:.1f}GB  {info['n_layers']:<8} "
            f"{info['notes'][:55]}"
        )
    print("\nRecommended start order (smallest to largest):")
    print("  1. deepseek-coder-1.3b  — 2.6GB, fastest, no safety filters")
    print("  2. qwen-coder-1.5b      — 3.0GB, best code quality at size")
    print("  3. phi-3-mini           — 7.6GB, different architecture for RQ2")
    print("  4. dolphin-llama3-8b    — 16GB,  direct comparison with Ollama dolphin3:8b")


def pull_model(tag: str) -> None:
    """Download a model from HuggingFace using huggingface_hub."""
    if tag not in SLM_REGISTRY:
        print(f"[!] Unknown tag: {tag}. Run --list to see options.")
        sys.exit(1)
    info = SLM_REGISTRY[tag]
    hf_id = info["hf_id"]
    print(f"[*] Pulling {tag} ({hf_id}) — ~{info['size_gb']:.1f}GB")
    print(f"    {info['notes']}")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id=hf_id,
            ignore_patterns=["*.gguf", "*.bin"],  # prefer safetensors
        )
        print(f"[+] Downloaded to: {path}")
    except ImportError:
        print("[!] huggingface_hub not found. pip install huggingface_hub")
        print(f"    Manual: huggingface-cli download {hf_id}")


def extract_hidden_states(
    hf_id: str,
    codes: list[str],
    layer_fraction: float,
    cache_path: str,
    force: bool = False,
) -> tuple[np.ndarray, int]:
    """Extract hidden states at the best layer for all code snippets.

    Args:
        hf_id: HuggingFace model id.
        codes: List of code strings.
        layer_fraction: Fraction of layers to hook (0.75 = 75% depth).
        cache_path: .npz cache path.
        force: Ignore cache.

    Returns:
        (X, hook_layer_idx) — shape (n, hidden_dim) and layer index used.
    """
    cache = Path(cache_path)
    if cache.exists() and not force:
        data = np.load(cache)
        if data["X"].shape[0] == len(codes):
            print(f"[*] Loaded cache: {data['X'].shape} from {cache}")
            return data["X"], int(data["hook_layer"])
        print(f"[!] Cache mismatch, re-extracting.")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[!] pip install torch transformers accelerate")
        sys.exit(1)

    print(f"[*] Loading {hf_id} for hidden state extraction...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float32,   # CPU: use float32
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    n_layers = len(model.model.layers)
    hook_layer = int(layer_fraction * n_layers)
    hidden_dim = model.config.hidden_size
    print(f"[*] {n_layers} layers, hooking layer {hook_layer} ({layer_fraction*100:.0f}% depth), dim={hidden_dim}")

    captured: list[np.ndarray] = []

    def _hook(module: Any, inp: Any, out: Any) -> None:
        h = out[0] if isinstance(out, tuple) else out
        # Mean-pool over sequence length
        vec = h.mean(dim=1).detach().float().squeeze(0).numpy()
        captured.append(vec)

    handle = model.model.layers[hook_layer].register_forward_hook(_hook)

    X = np.zeros((len(codes), hidden_dim), dtype=np.float32)
    t0 = time.time()

    for i, code in enumerate(codes):
        captured.clear()
        inputs = tokenizer(
            code, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            model(**inputs)
        if captured:
            X[i] = captured[0]
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(codes)} ({time.time()-t0:.0f}s)")

    handle.remove()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, hook_layer=np.array(hook_layer))
    print(f"[*] Saved: {X.shape} -> {cache}")
    return X, hook_layer


def train_probe_for_slm(
    tag: str,
    redbench_dir: str,
    force: bool = False,
) -> dict[str, Any]:
    """Train a per-layer probe for the specified SLM.

    Args:
        tag: SLM registry tag.
        redbench_dir: Path to redbench datasets/.
        force: Re-extract hidden states even if cached.

    Returns:
        dict: {tag, hf_id, hook_layer, auc_cv, acc_cv, embed_dim, weights_path}
    """
    import json as _json
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    info = SLM_REGISTRY[tag]
    hf_id = info["hf_id"]

    # Load redbench
    base = Path(redbench_dir)
    vuln_codes: list[str] = []
    safe_codes: list[str] = []
    for vc in sorted(d.name for d in base.iterdir() if d.is_dir()):
        for fname in ("samples.jsonl", "samples_generated.jsonl", "samples_generated_v2.jsonl"):
            jsonl = base / vc / fname
            if not jsonl.exists():
                continue
            with open(jsonl, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    s = _json.loads(line)
                    vuln_codes.append(s["code"])
                    safe_codes.append(s["fix"])

    codes = vuln_codes + safe_codes
    y = np.array([1] * len(vuln_codes) + [0] * len(safe_codes), dtype=int)
    print(f"[*] Dataset: {len(codes)} snippets ({y.sum()} vuln, {(1-y).sum()} safe)")

    model_tag = tag.replace("-", "_").replace(".", "_")
    cache_path = f".activguard/hf_cache_{model_tag}.npz"
    weights_path = f".activguard/hf_probe_{model_tag}.pkl"

    X, hook_layer = extract_hidden_states(hf_id, codes, info["hook_layer_frac"], cache_path, force)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                    solver="lbfgs", random_state=42)),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")
    acc_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    clf.fit(X, y)

    result = {
        "tag": tag,
        "hf_id": hf_id,
        "hook_layer": hook_layer,
        "n_layers": info["n_layers"],
        "embed_dim": X.shape[1],
        "auc_cv": float(auc_scores.mean()),
        "auc_std": float(auc_scores.std()),
        "acc_cv": float(acc_scores.mean()),
        "n_samples": len(codes),
        "weights_path": weights_path,
    }

    # Save probe
    scaler = clf.named_steps["scaler"]
    raw_clf = clf.named_steps["clf"]
    with open(weights_path, "wb") as fh:
        pickle.dump({
            "clf": raw_clf, "scaler": scaler,
            "embed_dim": X.shape[1], "hook_layer": hook_layer,
            "hf_id": hf_id, "tag": tag,
        }, fh)

    print(f"\n[+] {tag}: AUC={result['auc_cv']:.4f} +/- {result['auc_std']:.4f}  "
          f"Acc={result['acc_cv']:.4f}")
    print(f"    Weights: {weights_path}")
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="SLM setup and per-layer probe training")
    parser.add_argument("--list", action="store_true", help="List available SLMs")
    parser.add_argument("--pull", metavar="TAG", help="Download model from HuggingFace")
    parser.add_argument("--train-probe", metavar="TAG", help="Train per-layer probe for model")
    parser.add_argument("--run-all-probes", action="store_true", help="Train probes for all pulled models")
    parser.add_argument("--redbench", default="../redbench/datasets")
    parser.add_argument("--force", action="store_true", help="Re-extract hidden states")
    parser.add_argument("--stream", metavar="TAG", help="Run streaming probe with this SLM")
    parser.add_argument("--prompt", default="Write a Flask route that queries user data by ID using string formatting in SQL")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.pull:
        pull_model(args.pull)
        return

    if args.train_probe:
        result = train_probe_for_slm(args.train_probe, args.redbench, args.force)
        print(f"\n[*] Summary: {result}")
        return

    if args.run_all_probes:
        results = []
        for tag in SLM_REGISTRY:
            cache = Path(f".activguard/hf_cache_{tag.replace('-','_').replace('.','_')}.npz")
            if not cache.exists():
                print(f"[*] Skipping {tag} — model not pulled yet. Run --pull {tag} first.")
                continue
            r = train_probe_for_slm(tag, args.redbench, args.force)
            results.append(r)

        if results:
            print(f"\n{'='*65}")
            print(f"  Cross-SLM probe comparison (RQ1: layer depth, RQ2: cross-model)")
            print(f"{'='*65}")
            print(f"  {'Model':<25} {'Hook Layer':<12} {'AUC CV':<12} {'Acc CV'}")
            print(f"  {'-'*55}")
            for r in results:
                depth_pct = int(100 * r["hook_layer"] / r["n_layers"])
                print(f"  {r['tag']:<25} {r['hook_layer']}/{r['n_layers']} ({depth_pct}%)   "
                      f"{r['auc_cv']:.4f} +/-{r['auc_std']:.4f}  {r['acc_cv']:.4f}")
        return

    if args.stream:
        from probe.universal_streaming_probe import UniversalStreamingProbe
        tag = args.stream
        weights = f".activguard/hf_probe_{tag.replace('-','_').replace('.','_')}.pkl"
        if not Path(weights).exists():
            print(f"[!] No probe weights for {tag}. Run --train-probe {tag} first.")
            sys.exit(1)
        probe = UniversalStreamingProbe(
            gen_model=SLM_REGISTRY[tag].get("ollama_tag", tag),
            probe_weights_path=".activguard/layer_probe_weights.pkl",  # universal detector
        )
        result = probe.run(args.prompt)
        print(result.summary())
        print(result.ascii_chart())
        return

    print("No action specified. Run with --help or --list.")


if __name__ == "__main__":
    main()
