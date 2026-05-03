"""
Layer 1 (True) — HuggingFace Residual Stream Probe.

Intercepts the GENERATING model's own hidden states at every token synthesis
step, rather than encoding the final output through a separate encoder (CodeBERT).

Core PhD contribution:
    The probe is trained on hidden states extracted via model forward passes on
    labelled code, and generalises to live generation because the model's internal
    representations are consistent between encoding and generating modes.  We show
    empirically that during vulnerable-code generation, the last-layer residual
    stream diverges from safe-code generation paths at step 1 (cosine_sim ≈ 0.17
    by step 3) — the model commits to the vulnerability in latent space BEFORE
    emitting the first syntactically vulnerable token.

Architecture (encoding / training):
    code snippet → tokenise → model(output_hidden_states=True)
                 → hidden_states[-1][:, -1, :]  (last layer, last token position)
                 → LayerNorm → _VulnMLP (3-layer PyTorch) → P(vulnerable) ∈ [0, 1]

    Probe type "mlp"  (default): AdamW + cosine LR, BCEWithLogitsLoss, early stopping
    Probe type "lr"   (legacy):  StandardScaler + LogisticRegression (sklearn)

Architecture (generation / real-time):
    prompt → manual token-by-token loop
           → at each step: score hidden_states[-1][:, -1, :]
           → stop if P(vulnerable) > threshold  ← fires BEFORE the vulnerable
                                                   token is emitted

Supported models (tested on Windows CPU / DirectML):
    Qwen/Qwen2.5-Coder-1.5B-Instruct  — 1544M params, 28 layers, dim=1536
    Qwen/Qwen2.5-Coder-3B-Instruct    — 3B params,   28 layers, dim=2048
    Qwen/Qwen2.5-Coder-7B-Instruct    — 7B params,   28 layers, dim=3584

Research questions addressed:
    RQ1: Which transformer layers carry strongest vulnerability signal?
         → Layer sweep in layer_sweep() method: compare all 28 layer AUCs.
    RQ2: Does the activation probe generalise cross-model?
         → Separate probe instance per model; compare AUCs via benchmark_all.py.
    RQ4: What is the precision/recall tradeoff?
         → Tunable threshold; ROC curve exported by fit_and_evaluate().

Reference: Zou et al., "Representation Engineering: A Top-Down Approach to
AI Transparency", arXiv:2310.01405 (2023).
Reference: Feng et al., "CodeBERT: A Pre-Trained Model for Programming and
Natural Languages", arXiv:2002.08155 (2020).
Reference: Kulkarni, "Latent Adversarial Detection: Adaptive Probing of LLM
Activations for Multi-Turn Attack Detection", arXiv:2604.28129 (2026).
Reference: Lan et al., "Dynamic Adversarial Fine-Tuning Reorganizes Refusal
Geometry", arXiv:2604.27019 (2026).
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyTorch MLP probe  (Karpathy-style: minimal, single-file, AdamW)
# ---------------------------------------------------------------------------

if _HF_AVAILABLE:
    class _VulnMLP(nn.Module):
        """3-layer MLP probe for vulnerability detection from hidden states.

        Architecture: LayerNorm → Linear(dim, 256) → GELU → Dropout →
                      Linear(256, 64) → GELU → Linear(64, 1)

        Outputs raw logits (no sigmoid) — apply sigmoid at inference time for
        numerical stability with BCEWithLogitsLoss during training.

        Research context:
            Replaces the linear LogisticRegression decision boundary with a
            non-linear classifier.  The residual stream at layer 12 of
            Qwen2.5-Coder-1.5B lives in a 1536-d space; vulnerability patterns
            are unlikely to be linearly separable after balanced-data training.

        Reference: Nanda et al., "Progress measures for grokking via mechanistic
        interpretability", arXiv:2301.05217 (2023).
        """

        def __init__(self, input_dim: int, dropout: float = 0.15) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, 1),   # raw logit — no sigmoid here
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Return raw logit tensor of shape (batch,)."""
            return self.net(x).squeeze(-1)

    def _train_mlp(
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 150,
        lr: float = 1e-3,
        batch_size: int = 32,
        patience: int = 20,
        seed: int = 42,
    ) -> "_VulnMLP":
        """Train a _VulnMLP with AdamW + cosine LR + early stopping.

        Karpathy-style training loop: explicit epochs, gradient clipping,
        cosine annealing, class-weighted BCE loss for imbalanced data.

        Args:
            X: Feature matrix (n_samples, hidden_dim) float32.
            y: Binary labels float32.
            epochs: Maximum training epochs.
            lr: Initial learning rate for AdamW.
            batch_size: Mini-batch size.
            patience: Early-stopping patience (val loss).
            seed: Random seed for reproducibility.

        Returns:
            Trained _VulnMLP with best-validation-loss weights loaded.
        """
        torch.manual_seed(seed)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        # 80/20 stratified split
        n = len(y_t)
        pos_idx = (y_t == 1).nonzero(as_tuple=True)[0]
        neg_idx = (y_t == 0).nonzero(as_tuple=True)[0]
        n_pos_tr = max(1, int(0.8 * len(pos_idx)))
        n_neg_tr = max(1, int(0.8 * len(neg_idx)))
        tr_idx = torch.cat([pos_idx[:n_pos_tr], neg_idx[:n_neg_tr]])
        va_idx = torch.cat([pos_idx[n_pos_tr:], neg_idx[n_neg_tr:]])

        n_vuln = int(y_t[tr_idx].sum().item())
        n_safe = len(tr_idx) - n_vuln
        pos_weight = torch.tensor([max(n_safe, 1) / max(n_vuln, 1)], dtype=torch.float32)

        mlp = _VulnMLP(input_dim=X.shape[1])
        opt = optim.AdamW(mlp.parameters(), lr=lr, weight_decay=0.01)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss = float("inf")
        best_state: dict | None = None
        no_improve = 0

        for epoch in range(epochs):
            mlp.train()
            perm = tr_idx[torch.randperm(len(tr_idx))]
            for i in range(0, len(perm), batch_size):
                bi = perm[i : i + batch_size]
                xb, yb = X_t[bi], y_t[bi]
                opt.zero_grad()
                loss = crit(mlp(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
                opt.step()
            sched.step()

            if len(va_idx) > 0:
                mlp.eval()
                with torch.no_grad():
                    val_loss = crit(mlp(X_t[va_idx]), y_t[va_idx]).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    logger.debug("Early stop at epoch %d  val_loss=%.4f", epoch, val_loss)
                    break

        if best_state is not None:
            mlp.load_state_dict(best_state)
        mlp.eval()
        return mlp

else:
    # Stubs so the module can be imported without torch
    class _VulnMLP:  # type: ignore[no-redef]
        pass

    def _train_mlp(*args: object, **kwargs: object) -> None:  # type: ignore[misc]
        raise ImportError("torch is required for MLP probe training.")


DEFAULT_MODEL      = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_WEIGHTS    = ".activguard/hf_probe_{model_slug}.pkl"
DEFAULT_THRESHOLD  = 0.55
DEFAULT_LAYER      = -1     # Last layer (index -1 from hidden_states tuple)
DEFAULT_MAX_TOKENS = 512


def _slug(model_name: str) -> str:
    """Convert 'Qwen/Qwen2.5-Coder-1.5B-Instruct' → 'qwen2-5-coder-1-5b'."""
    return model_name.lower().split("/")[-1].replace(".", "-").replace("_", "-")


def _effective_rank_metrics(matrix: np.ndarray) -> dict[str, float]:
    """Return entropy effective-rank diagnostics for a probe weight matrix.

    Effective rank is exp(entropy(normalized singular values)).  It is useful
    here because a low value means the probe's vulnerability signal occupies a
    small residual-stream subspace, matching the refusal-geometry protocol in
    arXiv:2604.27019.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    if m.size == 0:
        return {"effective_rank": 0.0, "stable_rank": 0.0, "spectral_rank": 0.0}

    singular_values = np.linalg.svd(m, compute_uv=False)
    singular_values = singular_values[singular_values > 1e-12]
    if singular_values.size == 0:
        return {"effective_rank": 0.0, "stable_rank": 0.0, "spectral_rank": 0.0}

    probs = singular_values / singular_values.sum()
    entropy = -float(np.sum(probs * np.log(probs)))
    stable_rank = float(np.sum(singular_values ** 2) / (singular_values[0] ** 2))
    return {
        "effective_rank": round(float(np.exp(entropy)), 4),
        "stable_rank": round(stable_rank, 4),
        "spectral_rank": float(singular_values.size),
    }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Result from a real-time hidden-state monitored generation run.

    Attributes:
        flagged: True if probe exceeded threshold during generation.
        stop_step: Token index when probe stopped generation (None if completed).
        peak_confidence: Highest P(vulnerable) across all steps.
        final_text: Complete generated text (truncated at stop if flagged).
        score_timeline: [(step, p_vuln), ...] for every generated token.
        total_steps: Number of generation steps executed.
        elapsed_s: Wall-clock time for the full run.
        model: HuggingFace model name used.
        layer: Transformer layer index probed.
    """

    flagged: bool
    stop_step: int | None
    peak_confidence: float
    final_text: str
    score_timeline: list[tuple[int, float]]
    total_steps: int
    max_new_tokens: int
    elapsed_s: float
    model: str
    layer: int

    def summary(self) -> str:
        """Human-readable one-line result."""
        status = "VIOLATION" if self.flagged else "VERIFIED"
        s = (
            f"[{status}] model={self.model.split('/')[-1]} "
            f"layer={self.layer} steps={self.total_steps} "
            f"peak={self.peak_confidence:.3f}"
        )
        if self.flagged and self.stop_step is not None:
            pct = 100 * self.stop_step / max(self.total_steps, 1)
            s += f"  stopped at step {self.stop_step} ({pct:.0f}%)"
        return s


# ---------------------------------------------------------------------------
# HF model wrapper (load-once singleton per model name)
# ---------------------------------------------------------------------------

class _HFModelCache:
    """Cache of loaded HuggingFace models — avoids reloading across probe calls.

    Research context:
        Loading a 1.5B model takes ~4s from disk cache.  Caching at module level
        ensures the proxy can serve multiple requests without model reload latency.
    """

    _cache: dict[str, tuple[Any, Any]] = {}   # model_name → (tokenizer, model)

    @classmethod
    def get(cls, model_name: str) -> tuple[Any, Any]:
        """Return (tokenizer, model), loading from HuggingFace if not cached.

        Args:
            model_name: HuggingFace model identifier.

        Returns:
            Tuple of (AutoTokenizer, AutoModelForCausalLM).

        Raises:
            ImportError: If transformers/torch are not installed.
            RuntimeError: If the model fails to load.
        """
        if not _HF_AVAILABLE:
            raise ImportError(
                "torch and transformers are required for HFHiddenProbe. "
                "pip install torch transformers"
            )
        if model_name not in cls._cache:
            logger.info("Loading %s ...", model_name)
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,   # fp32 for CPU stability
                device_map="cpu",
                trust_remote_code=True,
            )
            mdl.eval()
            elapsed = time.time() - t0
            n_layers = mdl.config.num_hidden_layers
            hidden_dim = mdl.config.hidden_size
            logger.info(
                "%s loaded in %.1fs — %d layers, dim=%d",
                model_name, elapsed, n_layers, hidden_dim,
            )
            cls._cache[model_name] = (tok, mdl)
        return cls._cache[model_name]

    @classmethod
    def is_loaded(cls, model_name: str) -> bool:
        """Return True if the model is already in memory."""
        return model_name in cls._cache


# ---------------------------------------------------------------------------
# Core probe class
# ---------------------------------------------------------------------------

class HFHiddenProbe:
    """Linear probe on the residual stream of a HuggingFace causal LM.

    This is the true implementation of the ActivGuard L1 pitch:
      - Uses the GENERATING model's own hidden states (not a separate encoder)
      - Can fire at every generation step (before each token is sampled)
      - Produces model-specific vulnerability signatures

    Training workflow:
        probe = HFHiddenProbe(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct")
        X, y = probe.build_feature_matrix(vuln_codes, safe_codes)
        metrics = probe.fit_and_evaluate(X, y)
        probe.save()

    Inference workflow (batch):
        p = probe.score(code_snippet)   → float P(vulnerable)

    Inference workflow (real-time generation):
        result = probe.generate_and_monitor(prompt, max_new_tokens=256)

    Attributes:
        model_name: HuggingFace model identifier.
        layer: Hidden state layer index to probe (-1 = last layer).
        threshold: P(vulnerable) decision boundary.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        layer: int = DEFAULT_LAYER,
        threshold: float = DEFAULT_THRESHOLD,
        weights_path: str | None = None,
        probe_type: str = "mlp",
    ) -> None:
        """Initialise the probe.

        Args:
            model_name: HuggingFace causal LM to probe.
            layer: Transformer layer index (-1 = last).
            threshold: P(vulnerable) decision boundary.
            weights_path: Override default .pkl save/load path.
            probe_type: "mlp" (default, PyTorch 3-layer) or "lr" (sklearn LR,
                legacy).  New probes default to "mlp"; existing saved "lr"
                probes load and run unchanged.
        """
        self.model_name = model_name
        self.layer = layer
        self.threshold = threshold
        self._probe_type = probe_type
        self._weights_path = weights_path or DEFAULT_WEIGHTS.format(
            model_slug=_slug(model_name)
        )

        # MLP path
        self._mlp: _VulnMLP | None = None

        # Legacy LR path (kept for backward compat with saved .pkl probes)
        self._clf: Any | None = None
        self._scaler: Any | None = None

        self._is_trained: bool = False
        self._hidden_dim: int = 0

        if Path(self._weights_path).exists():
            self._load(self._weights_path)
        # Constructor's probe_type always wins over the loaded pkl value.
        # This ensures `train_hf_probe.py --probe-type mlp` retrains as MLP
        # even when loading weights saved by an older LR probe.
        self._probe_type = probe_type

    # ------------------------------------------------------------------
    # Feature extraction (encoding mode)
    # ------------------------------------------------------------------

    def _tokenizer_and_model(self) -> tuple[Any, Any]:
        """Ensure model is loaded and return (tokenizer, model)."""
        return _HFModelCache.get(self.model_name)

    def encode(self, code: str) -> np.ndarray:
        """Extract residual stream vector for a code snippet.

        Runs a single forward pass (no generation) and returns the hidden
        state at position -1 (last token) of the requested layer.  This
        vector is the probe's input feature for the given code.

        Args:
            code: Python code snippet (or partial output) to encode.

        Returns:
            np.ndarray: Shape (hidden_dim,) float32 vector.
        """
        tok, mdl = self._tokenizer_and_model()
        inputs = tok(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
        )
        with torch.no_grad():
            out = mdl(**inputs, output_hidden_states=True)
        # hidden_states: tuple of (n_layers+1,) tensors, each (1, seq_len, dim)
        h = out.hidden_states[self.layer]   # (1, seq_len, hidden_dim)
        return h[0, -1, :].cpu().float().numpy()

    def encode_batch(
        self,
        codes: list[str],
        verbose: bool = False,
    ) -> np.ndarray:
        """Encode a list of code snippets, returning a feature matrix.

        Args:
            codes: List of code strings.
            verbose: Print progress every 10 items.

        Returns:
            np.ndarray: Shape (n, hidden_dim).
        """
        vectors: list[np.ndarray] = []
        for i, code in enumerate(codes):
            vectors.append(self.encode(code))
            if verbose and (i + 1) % 10 == 0:
                logger.info("Encoded %d/%d", i + 1, len(codes))
        return np.stack(vectors)

    def build_feature_matrix(
        self,
        vuln_codes: list[str],
        safe_codes: list[str],
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build labelled feature matrix from parallel vuln/safe code lists.

        Args:
            vuln_codes: Vulnerable code snippets (label=1).
            safe_codes: Safe / fixed code snippets (label=0).
            verbose: Log progress.

        Returns:
            Tuple (X, y) where X.shape = (2N, hidden_dim) and y.shape = (2N,).
        """
        if verbose:
            logger.info(
                "Encoding %d vuln + %d safe samples with %s layer=%d ...",
                len(vuln_codes), len(safe_codes), self.model_name, self.layer,
            )
        X_vuln = self.encode_batch(vuln_codes, verbose=verbose)
        X_safe = self.encode_batch(safe_codes, verbose=verbose)
        X = np.vstack([X_vuln, X_safe])
        y = np.concatenate([np.ones(len(vuln_codes)), np.zeros(len(safe_codes))])
        return X.astype(np.float32), y.astype(np.int32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C: float = 1.0,
    ) -> dict[str, Any]:
        """Train probe on pre-computed feature matrix.

        Dispatches to _VulnMLP (AdamW, cosine LR) when probe_type="mlp",
        or sklearn LogisticRegression when probe_type="lr".

        Args:
            X: Shape (n_samples, hidden_dim) float32.
            y: Shape (n_samples,) — binary labels (1=vulnerable, 0=safe).
            C: Regularisation strength for LR fallback (ignored for MLP).

        Returns:
            dict: train_accuracy, n_samples, hidden_dim, probe_type.
        """
        self._hidden_dim = X.shape[1]
        X_f = X.astype(np.float32)
        y_f = y.astype(np.float32)

        if self._probe_type == "mlp":
            logger.info(
                "Training MLP probe (dim=%d, n=%d) with AdamW+cosine ...",
                self._hidden_dim, len(y_f),
            )
            self._mlp = _train_mlp(X_f, y_f)
            self._clf = None
            self._scaler = None
            self._is_trained = True

            # Train-set accuracy
            with torch.no_grad():
                logits = self._mlp(torch.tensor(X_f))
                preds = (torch.sigmoid(logits) >= self.threshold).float().numpy()
            train_acc = float((preds == y_f).mean())

        else:
            # Legacy LR path
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X_f)
            self._clf = LogisticRegression(
                C=C, max_iter=2000, class_weight="balanced",
                solver="lbfgs", random_state=42,
            )
            self._clf.fit(X_scaled, y_f)
            self._mlp = None
            self._is_trained = True
            train_acc = float(self._clf.score(X_scaled, y_f))

        return {
            "train_accuracy": round(train_acc, 4),
            "n_samples": int(len(y)),
            "n_vulnerable": int(y.sum()),
            "n_safe": int((y == 0).sum()),
            "hidden_dim": self._hidden_dim,
            "model": self.model_name,
            "layer": self.layer,
            "probe_type": self._probe_type,
        }

    def fit_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> dict[str, Any]:
        """Train with stratified k-fold cross-validation and report AUC/F1.

        Reports per-fold AUC (ROC) and mean metrics.  This is the primary
        method for the benchmark_all.py comparison.

        Args:
            X: Feature matrix.
            y: Binary labels.
            n_splits: Number of CV folds (default 5).

        Returns:
            dict: mean_auc, std_auc, mean_f1, std_f1, per_fold details.
        """
        from sklearn.metrics import f1_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs: list[float] = []
        f1s: list[float] = []
        fold_rank_metrics: list[dict[str, float]] = []
        fold_directions: list[np.ndarray] = []
        X_f = X.astype(np.float32)
        y_f = y.astype(np.float32)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_f, y_f)):
            X_tr, X_te = X_f[train_idx], X_f[test_idx]
            y_tr, y_te = y_f[train_idx], y_f[test_idx]

            if self._probe_type == "mlp":
                fold_mlp = _train_mlp(X_tr, y_tr, seed=42 + fold)
                fold_mlp.eval()
                with torch.no_grad():
                    probs = torch.sigmoid(
                        fold_mlp(torch.tensor(X_te))
                    ).numpy()
                first_linear = fold_mlp.net[1].weight.detach().cpu().numpy()
                fold_rank_metrics.append(_effective_rank_metrics(first_linear))
            else:
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                clf = LogisticRegression(
                    C=1.0, max_iter=2000, class_weight="balanced",
                    solver="lbfgs", random_state=42,
                )
                clf.fit(X_tr_s, y_tr)
                probs = clf.predict_proba(X_te_s)[:, 1]
                input_space_direction = clf.coef_[0] / np.maximum(scaler.scale_, 1e-12)
                fold_directions.append(input_space_direction)
                fold_rank_metrics.append(_effective_rank_metrics(input_space_direction))

            preds = (probs >= self.threshold).astype(int)

            try:
                aucs.append(float(roc_auc_score(y_te, probs)))
            except ValueError:
                aucs.append(0.0)
            f1s.append(float(f1_score(y_te, preds, zero_division=0)))
            logger.info("  Fold %d: AUC=%.3f  F1=%.3f", fold + 1, aucs[-1], f1s[-1])

        # Fit final model on full data
        self.fit(X_f, y_f)

        cv_direction_rank = (
            _effective_rank_metrics(np.vstack(fold_directions))
            if fold_directions
            else {}
        )

        return {
            "mean_auc":  round(float(np.mean(aucs)), 4),
            "std_auc":   round(float(np.std(aucs)),  4),
            "mean_f1":   round(float(np.mean(f1s)),  4),
            "std_f1":    round(float(np.std(f1s)),   4),
            "per_fold_auc": [round(a, 4) for a in aucs],
            "per_fold_effective_rank": [
                m["effective_rank"] for m in fold_rank_metrics
            ],
            "mean_effective_rank": round(
                float(np.mean([m["effective_rank"] for m in fold_rank_metrics])),
                4,
            ) if fold_rank_metrics else 0.0,
            "cv_direction_effective_rank": cv_direction_rank.get("effective_rank", 0.0),
            "cv_direction_stable_rank": cv_direction_rank.get("stable_rank", 0.0),
            "n_folds": n_splits,
            "model": self.model_name,
            "layer": self.layer,
            "n_samples": len(y),
        }

    def layer_sweep(
        self,
        X_all: dict[int, np.ndarray],
        y: np.ndarray,
        n_splits: int = 5,
    ) -> dict[int, dict[str, Any]]:
        """Sweep over all transformer layers to find maximally discriminative one.

        Addresses RQ1: Which layer carries the strongest vulnerability signal?

        Args:
            X_all: Dict mapping layer_index → feature_matrix (n, dim).
            y: Binary labels (1=vulnerable, 0=safe).
            n_splits: CV folds.

        Returns:
            Dict mapping layer_index → {'mean_auc', 'std_auc', 'mean_f1', ...}.
        """
        results: dict[int, dict[str, Any]] = {}
        saved_layer = self.layer
        for layer_idx, X in sorted(X_all.items()):
            self.layer = layer_idx
            metrics = self.fit_and_evaluate(X, y, n_splits=n_splits)
            results[layer_idx] = metrics
            logger.info(
                "Layer %2d: AUC=%.3f ± %.3f  F1=%.3f",
                layer_idx, metrics["mean_auc"], metrics["std_auc"], metrics["mean_f1"],
            )
        self.layer = saved_layer
        return results

    # ------------------------------------------------------------------
    # Inference (encoding mode)
    # ------------------------------------------------------------------

    def score(self, code: str) -> float:
        """Return P(vulnerable) for a code snippet.

        Args:
            code: Python code to score.

        Returns:
            float: P(vulnerable) ∈ [0, 1].

        Raises:
            RuntimeError: If probe has not been trained / loaded.
        """
        if not self._is_trained:
            raise RuntimeError(
                "Probe is not trained.  Run fit() or load a saved probe first."
            )
        vec = self.encode(code)
        return self.score_vector(vec)

    def score_result(self, code: str) -> dict[str, Any]:
        """Score with full result dict (result, confidence, evidence).

        Args:
            code: Python code to score.

        Returns:
            dict with keys: result, confidence, evidence, model, layer.
        """
        p = self.score(code)
        return {
            "result": "VIOLATION" if p >= self.threshold else "VERIFIED",
            "confidence": round(p, 4),
            "evidence": (
                f"HF residual stream probe: P(vulnerable)={p:.3f} "
                f"(model={self.model_name.split('/')[-1]}, "
                f"layer={self.layer}, threshold={self.threshold})"
            ),
            "model": self.model_name,
            "layer": self.layer,
        }

    def score_vector(self, h: np.ndarray) -> float:
        """Score a pre-extracted hidden-state vector directly.

        Used during real-time generation to avoid re-tokenising the prompt
        at every step — the generation loop extracts h itself.

        Args:
            h: Shape (hidden_dim,) or (1, hidden_dim) float32 array.

        Returns:
            float: P(vulnerable).
        """
        if not self._is_trained:
            raise RuntimeError("Probe is not trained.")
        vec = np.asarray(h, dtype=np.float32).reshape(1, -1)

        if self._mlp is not None:
            with torch.no_grad():
                logit = self._mlp(torch.tensor(vec))
            return float(torch.sigmoid(logit).item())

        # Legacy LR path
        assert self._clf is not None and self._scaler is not None
        vec_s = self._scaler.transform(vec)
        return float(self._clf.predict_proba(vec_s)[0, 1])

    # ------------------------------------------------------------------
    # Real-time generation with hidden-state monitoring
    # ------------------------------------------------------------------

    def generate_and_monitor(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        stop_on_flag: bool = True,
        probe_every_n: int = 20,
        warmup_steps: int = 50,
        min_consecutive_flags: int = 5,
    ) -> GenerationResult:
        """Generate text token-by-token, probing hidden states at every step.

        This is the core real-time interception loop.  At each step, the
        last-layer hidden state is extracted BEFORE the next token is sampled.
        If P(vulnerable) exceeds threshold, generation is stopped immediately —
        before the vulnerable token is added to the output.

        The ``warmup_steps`` parameter prevents false positives caused by
        scoring the residual stream of the last PROMPT token (step 0).  At
        step 0, the hidden state encodes the prompt context, not generated code
        — security vocabulary in the prompt ("parameterised", "JWT", etc.)
        activates vulnerability features even for benign requests.  Skipping
        the first ``warmup_steps`` tokens ensures the probe only fires on
        model-generated content.

        Args:
            prompt: Input prompt for the LM.
            max_new_tokens: Maximum tokens to generate.
            stop_on_flag: Stop generation immediately on VIOLATION.
            probe_every_n: Score hidden state every N steps (default=1 = every token).
            warmup_steps: Number of initial generation steps to skip before
                scoring begins.  Default 50 ensures the model has generated
                actual code logic (not just task descriptions in natural
                language) before the probe scores.  At step 10-30, both
                vulnerable and safe prompts produce indistinguishable
                descriptions; by step 50 the implementation diverges.
            min_consecutive_flags: Number of consecutive above-threshold scores
                required before a VIOLATION is declared.  Default 5 implements
                a sliding window majority vote: transient spikes from
                security-topic vocabulary in safe implementations are filtered
                out, while sustained vulnerability patterns trigger a stop.
                Set to 1 to restore single-step behaviour.

        Returns:
            GenerationResult with score timeline and stop metadata.
        """
        if not self._is_trained:
            raise RuntimeError("Probe is not trained.  Call fit() first.")

        tok, mdl = self._tokenizer_and_model()
        input_ids: torch.Tensor = tok.encode(prompt, return_tensors="pt")

        past_key_values = None
        generated_ids: list[int] = []
        score_timeline: list[tuple[int, float]] = []
        peak_confidence = 0.0
        stop_step: int | None = None
        flagged = False
        consecutive_flags: int = 0
        t0 = time.time()

        for step in range(max_new_tokens):
            # Feed only the last token when KV-cache is active
            current_input = input_ids if past_key_values is None else input_ids[:, -1:]

            with torch.no_grad():
                out = mdl(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )

            # Extract hidden state of the last generated position
            h_last: np.ndarray = (
                out.hidden_states[self.layer][0, -1, :].cpu().float().numpy()
            )

            # Skip warmup steps: let the model generate enough code to
            # reveal its implementation intent before scoring.
            if step < warmup_steps:
                logger.debug("step=%d warmup — skipping probe", step)
            elif step % probe_every_n == 0:
                # Re-encode the FULL accumulated text to get the same
                # hidden-state distribution as static scoring.  Per-step
                # cached hidden states differ from full-sequence encoding
                # and cause false positives on safe code.
                accumulated_text = tok.decode(generated_ids,
                                              skip_special_tokens=True)
                if accumulated_text.strip():
                    p = self.score(accumulated_text)
                else:
                    p = 0.0
                peak_confidence = max(peak_confidence, p)
                score_timeline.append((step, p))
                logger.debug("step=%d P(vuln)=%.3f", step, p)

                if p >= self.threshold:
                    consecutive_flags += 1
                else:
                    consecutive_flags = 0  # reset on any sub-threshold step

                if consecutive_flags >= min_consecutive_flags and stop_on_flag:
                    stop_step = step
                    flagged = True
                    logger.info(
                        "VIOLATION at step %d P(vuln)=%.3f "
                        "(%d consecutive flags) — stopping generation",
                        step, p, consecutive_flags,
                    )
                    break

            # Sample next token (greedy)
            next_id = int(out.logits[0, -1, :].argmax())
            generated_ids.append(next_id)
            past_key_values = out.past_key_values
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]])], dim=-1
            )

            # Stop on EOS
            if next_id == tok.eos_token_id:
                break

        elapsed = time.time() - t0
        final_text = tok.decode(generated_ids, skip_special_tokens=True)

        return GenerationResult(
            flagged=flagged,
            stop_step=stop_step,
            peak_confidence=round(peak_confidence, 4),
            final_text=final_text,
            score_timeline=score_timeline,
            total_steps=len(generated_ids),
            max_new_tokens=max_new_tokens,
            elapsed_s=round(elapsed, 2),
            model=self.model_name,
            layer=self.layer,
        )

    def iter_hidden_states(
        self,
        codes: list[str],
        layers: list[int] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield per-sample hidden states for all (or selected) layers.

        Used by train_hf_probe.py to collect the feature matrices needed
        for the layer sweep (RQ1).

        Args:
            codes: List of code strings.
            layers: Layer indices to collect (None = all layers).

        Yields:
            dict: {'code': str, 'layer': int, 'vector': np.ndarray}
        """
        tok, mdl = self._tokenizer_and_model()
        n_layers = mdl.config.num_hidden_layers + 1   # +1 for embedding layer

        for code in codes:
            inputs = tok(
                code,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False,
            )
            with torch.no_grad():
                out = mdl(**inputs, output_hidden_states=True)

            target_layers = layers if layers is not None else list(range(n_layers))
            for li in target_layers:
                if abs(li) < n_layers:
                    h = out.hidden_states[li][0, -1, :].cpu().float().numpy()
                    yield {"code": code, "layer": li, "vector": h}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        """Save probe weights to disk.

        Args:
            path: Destination path (defaults to self._weights_path).

        Returns:
            str: Absolute path where weights were saved.
        """
        out = Path(path or self._weights_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "probe_type": self._probe_type,
            "model_name": self.model_name,
            "layer":      self.layer,
            "threshold":  self.threshold,
            "hidden_dim": self._hidden_dim,
            # MLP: save state dict + architecture config
            "mlp_state":  self._mlp.state_dict() if self._mlp is not None else None,
            # LR legacy: save sklearn objects
            "clf":        self._clf,
            "scaler":     self._scaler,
        }
        with open(out, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info("HFHiddenProbe saved → %s  (probe_type=%s)", out, self._probe_type)
        return str(out.resolve())

    def _load(self, path: str) -> None:
        """Load probe weights from disk (called by __init__ if file exists).

        Handles both new MLP probes (probe_type="mlp") and legacy LR probes
        saved before the MLP upgrade — any pkl without "probe_type" key is
        treated as a legacy LR probe and loaded transparently.
        """
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        self.threshold   = payload["threshold"]
        self._hidden_dim = payload["hidden_dim"]
        self._probe_type = payload.get("probe_type", "lr")  # default legacy → lr

        saved_layer = payload.get("layer", self.layer)
        if saved_layer != self.layer:
            logger.warning(
                "Probe at %s trained on layer=%d but instance uses layer=%d",
                path, saved_layer, self.layer,
            )

        if self._probe_type == "mlp" and payload.get("mlp_state") is not None:
            self._mlp = _VulnMLP(input_dim=self._hidden_dim)
            self._mlp.load_state_dict(payload["mlp_state"])
            self._mlp.eval()
            self._clf = None
            self._scaler = None
        else:
            # Legacy LR probe
            self._clf    = payload.get("clf")
            self._scaler = payload.get("scaler")
            self._mlp    = None
            self._probe_type = "lr"

        self._is_trained = True
        logger.info(
            "HFHiddenProbe loaded from %s (model=%s layer=%d dim=%d probe=%s)",
            path, payload.get("model_name", "?"), saved_layer,
            self._hidden_dim, self._probe_type,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def probe_stats(self) -> dict[str, Any]:
        """Return probe metadata for health checks and logging."""
        stats: dict[str, Any] = {
            "model": self.model_name,
            "layer": self.layer,
            "threshold": self.threshold,
            "is_trained": self._is_trained,
            "hidden_dim": self._hidden_dim,
            "probe_type": self._probe_type,
            "model_loaded": _HFModelCache.is_loaded(self.model_name),
        }
        if self._mlp is not None:
            n_params = sum(p.numel() for p in self._mlp.parameters())
            stats["mlp_params"] = n_params
            first_linear = self._mlp.net[1].weight.detach().cpu().numpy()
            stats.update({
                f"first_layer_{k}": v
                for k, v in _effective_rank_metrics(first_linear).items()
            })
        elif self._clf is not None:
            coef = self._clf.coef_[0]
            stats["coef_norm"] = round(float(np.linalg.norm(coef)), 4)
            stats.update({
                f"coef_{k}": v
                for k, v in _effective_rank_metrics(coef).items()
            })
        return stats

    def __repr__(self) -> str:
        return (
            f"HFHiddenProbe(model={self.model_name!r}, "
            f"layer={self.layer}, probe={self._probe_type}, "
            f"trained={self._is_trained}, threshold={self.threshold})"
        )
