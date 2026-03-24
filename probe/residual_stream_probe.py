"""
Layer 1 — Residual Stream Activation Probe.

Reads the final hidden state (residual stream output) of a locally running
Ollama LLM and scores it with a trained linear probe to detect vulnerability
cluster signatures in activation space.

Architecture (Zou et al. "Representation Engineering", arXiv:2310.01405, 2023):
  code snippet → Ollama /api/embed → 4096-dim hidden state
                                   → LogisticRegression probe
                                   → P(vulnerable) ∈ [0, 1]

Why the final hidden state?
  The /api/embed endpoint returns the mean-pooled last-layer residual stream
  output — the same representation used by the model's unembedding matrix to
  predict the next token.  This is where vulnerability-related semantic
  content is maximally compressed after all 32 transformer layers have
  processed the code.

Research questions addressed:
  RQ1: Which transformer layers carry strongest vulnerability signal?
       (Partially — we probe the last layer here; per-layer requires
        llama-cpp-python or direct transformers access.)
  RQ2: Does the probe generalise cross-model?
       (Test by swapping PROBE_MODEL to qwen3:8b or nous-hermes2.)
  RQ4: What is the precision/recall tradeoff at Layer 1?
       (Measured in train_probe.py via 5-fold cross-validation.)

Reference: Zou et al., "Representation Engineering: A Top-Down Approach to
AI Transparency", arXiv:2310.01405 (2023).
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
PROBE_MODEL = "dolphin3:8b"   # uncensored — more likely to generate vulnerable code
PROBE_WEIGHTS_PATH = ".activguard/probe_weights.pkl"
VIOLATION_THRESHOLD = 0.55    # P(vulnerable) above this → VIOLATION flag


class ResidualStreamProbe:
    """Linear probe on the LLM residual stream for vulnerability detection.

    Attributes:
        model: Ollama model name used for embedding.
        threshold: Decision boundary for P(vulnerable).
        _clf: Trained LogisticRegression classifier.
        _scaler: StandardScaler fitted on training embeddings.
        _is_trained: Whether the probe has been fitted.
        _embed_dim: Dimension of the residual stream (4096 for 8B models).
    """

    def __init__(
        self,
        model: str = PROBE_MODEL,
        threshold: float = VIOLATION_THRESHOLD,
        weights_path: str = PROBE_WEIGHTS_PATH,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self._weights_path = weights_path
        self._clf: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._is_trained: bool = False
        self._embed_dim: int = 0

        if Path(weights_path).exists():
            self._load(weights_path)

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def embed(self, code: str) -> np.ndarray:
        """Extract residual stream hidden state for a code snippet.

        Calls the Ollama /api/embed endpoint, which returns the mean-pooled
        final-layer hidden state — the residual stream output before the
        unembedding matrix projects to vocabulary logits.

        Args:
            code: Python code snippet to embed.

        Returns:
            np.ndarray: 1-D float32 array of shape (embed_dim,).

        Raises:
            RuntimeError: If Ollama is unreachable or returns an error.
        """
        # Try /api/embed first (Ollama >= 0.3.0, returns {"embeddings": [[...]]}).
        # Fall back to /api/embeddings (legacy, returns {"embedding": [...]}).
        # Some models (e.g. qwen3:8b) only support the legacy endpoint.
        for endpoint, key, wrap in [
            ("/api/embed",       "embeddings", True),
            ("/api/embeddings",  "embedding",  False),
        ]:
            try:
                payload = {"model": self.model, "input" if wrap else "prompt": code}
                resp = requests.post(
                    f"{OLLAMA_BASE}{endpoint}",
                    json=payload,
                    timeout=30,
                )
                if resp.status_code == 501:
                    continue
                resp.raise_for_status()
                data = resp.json()
                if wrap:
                    vec = data.get("embeddings", [[]])[0]
                else:
                    vec = data.get("embedding", [])
                if vec:
                    return np.array(vec, dtype=np.float32)
            except requests.RequestException:
                continue
        raise RuntimeError(
            f"Ollama embed failed for model={self.model}: "
            "neither /api/embed nor /api/embeddings returned a vector."
        )

    def embed_batch(self, codes: list[str], delay_s: float = 0.05) -> np.ndarray:
        """Embed a list of code snippets into a matrix.

        Args:
            codes: List of code strings.
            delay_s: Throttle delay between requests (seconds).

        Returns:
            np.ndarray: Shape (n_samples, embed_dim).
        """
        vectors = []
        for i, code in enumerate(codes):
            vec = self.embed(code)
            vectors.append(vec)
            if delay_s > 0 and i < len(codes) - 1:
                time.sleep(delay_s)
            if (i + 1) % 10 == 0:
                logger.info("Embedded %d/%d snippets", i + 1, len(codes))
        return np.stack(vectors)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C: float = 1.0,
    ) -> dict:
        """Train the linear probe on pre-computed embeddings.

        Args:
            X: Shape (n_samples, embed_dim) — residual stream vectors.
            y: Shape (n_samples,) — binary labels (1 = vulnerable, 0 = safe).
            C: LogisticRegression regularisation strength (inverse of lambda).

        Returns:
            dict: Training metrics — accuracy, n_samples, class_counts.
        """
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._clf = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        self._clf.fit(X_scaled, y)
        self._is_trained = True
        self._embed_dim = X.shape[1]

        train_acc = self._clf.score(X_scaled, y)
        return {
            "train_accuracy": round(train_acc, 4),
            "n_samples": len(y),
            "n_vulnerable": int(y.sum()),
            "n_safe": int((1 - y).sum()),
            "embed_dim": self._embed_dim,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, code: str) -> dict:
        """Score a code snippet against the trained probe.

        Args:
            code: Python code to analyse.

        Returns:
            dict:
                - result (str): "VIOLATION" or "VERIFIED" or "UNTRAINED".
                - confidence (float): P(vulnerable) from the probe.
                - evidence (str): Human-readable explanation.
                - model (str): Ollama model used for embedding.
        """
        if not self._is_trained:
            return {
                "result": "UNTRAINED",
                "confidence": 0.0,
                "evidence": "Probe has not been trained. Run train_probe.py first.",
                "model": self.model,
            }

        try:
            vec = self.embed(code)
        except RuntimeError as exc:
            return {
                "result": "ERROR",
                "confidence": 0.0,
                "evidence": str(exc),
                "model": self.model,
            }

        assert self._scaler is not None and self._clf is not None
        vec_scaled = self._scaler.transform(vec.reshape(1, -1))
        p_vuln = float(self._clf.predict_proba(vec_scaled)[0, 1])

        is_violation = p_vuln >= self.threshold
        return {
            "result": "VIOLATION" if is_violation else "VERIFIED",
            "confidence": round(p_vuln, 4),
            "evidence": (
                f"Residual stream probe: P(vulnerable)={p_vuln:.3f} "
                f"(threshold={self.threshold}, model={self.model})"
            ),
            "model": self.model,
        }

    def score_functions(self, code: str) -> dict:
        """Score each top-level function in a code block separately.

        Returns the worst (highest P(vulnerable)) result across all functions.

        Args:
            code: Full Python source code.

        Returns:
            dict: Same schema as score(), plus 'per_function' list.
        """
        import ast as _ast

        functions: list[tuple[str, str]] = []
        try:
            tree = _ast.parse(code)
            for node in _ast.walk(tree):
                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    seg = _ast.get_source_segment(code, node)
                    if seg and len(seg.strip()) > 20:
                        functions.append((node.name, seg.strip()))
        except SyntaxError:
            pass

        if not functions:
            result = self.score(code)
            result["per_function"] = []
            return result

        per_function: list[dict] = []
        worst: dict | None = None

        for fn_name, fn_code in functions:
            r = self.score(fn_code)
            r["function"] = fn_name
            per_function.append(r)
            if worst is None or r["confidence"] > worst["confidence"]:
                worst = r

        assert worst is not None
        return {**worst, "per_function": per_function}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        """Persist probe weights to disk.

        Args:
            path: Output path (defaults to PROBE_WEIGHTS_PATH).

        Returns:
            str: Absolute path where weights were saved.
        """
        out = Path(path or self._weights_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "clf": self._clf,
            "scaler": self._scaler,
            "model": self.model,
            "threshold": self.threshold,
            "embed_dim": self._embed_dim,
        }
        with open(out, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Probe saved to %s", out)
        return str(out.resolve())

    def _load(self, path: str) -> None:
        """Load probe weights from disk.

        Note: self.model is intentionally NOT overwritten from the payload.
        The model is set by the constructor and governs which Ollama endpoint
        is called at inference time.  Saved model metadata is only used for
        logging/diagnostics.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._clf = payload["clf"]
        self._scaler = payload["scaler"]
        saved_model = payload.get("model", "unknown")
        if saved_model != self.model:
            logger.warning(
                "Probe at %s was trained with model=%s but this instance "
                "uses model=%s.  Embeddings will use %s.",
                path, saved_model, self.model, self.model,
            )
        self.threshold = payload["threshold"]
        self._embed_dim = payload["embed_dim"]
        self._is_trained = True
        logger.info(
            "Probe loaded from %s (saved_model=%s, active_model=%s, embed_dim=%d)",
            path, saved_model, self.model, self._embed_dim,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def probe_stats(self) -> dict:
        """Return probe metadata and classifier diagnostics.

        Returns:
            dict: embed_dim, model, threshold, is_trained, coef_norm.
        """
        stats: dict[str, Any] = {
            "model": self.model,
            "threshold": self.threshold,
            "is_trained": self._is_trained,
            "embed_dim": self._embed_dim,
        }
        if self._clf is not None:
            coef = self._clf.coef_[0]
            stats["coef_norm"] = float(np.linalg.norm(coef))
            top_idx = np.argsort(np.abs(coef))[-5:][::-1]
            stats["top5_dims"] = top_idx.tolist()
        return stats

    def __repr__(self) -> str:
        return (
            f"ResidualStreamProbe(model={self.model!r}, "
            f"trained={self._is_trained}, "
            f"threshold={self.threshold})"
        )
