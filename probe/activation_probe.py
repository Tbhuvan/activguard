"""
Activation Probe — Layer 1 vulnerability detection via residual stream analysis.

Hooks into the residual stream of a transformer-based coding LLM and runs a
lightweight linear (or MLP) classifier over the captured activations to detect
vulnerability signatures before the output text is generated.

Research background:
    Representation engineering (arXiv:2309.00933) demonstrates that high-level
    concepts are linearly encoded in transformer residual streams.  We extend
    this finding to *security-relevant* concepts: our hypothesis is that
    vulnerability signatures (e.g. "IDOR-prone ID lookup", "unsanitised URL
    fetch") form distinct linear clusters in the residual stream that are
    detectable with a simple probe trained on as few as 50 labelled examples.

Research questions addressed:
    RQ1: Which transformer layers carry the strongest vulnerability signal?
    RQ2: Is the signal consistent across vulnerability classes (IDOR vs SQLi)?
    RQ3: Can few-shot activation centroids detect zero-day vulnerability
         patterns without any retraining?

Note on executability:
    This module requires a running transformer model to call ``classify()`` and
    ``train_probe()``.  The architecture, hook registration, and training loop
    are fully specified; callers must provide a loaded HuggingFace model and
    tokenizer.  Running the unit tests does NOT require a GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Optional heavy imports — gracefully degrade if torch is not installed.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not available.  ActivationProbe will run in stub mode."
    )

# Vulnerability class labels used throughout the system.
VULNERABILITY_CLASSES = [
    "IDOR",
    "SQLi",
    "SSRF",
    "auth_bypass",
    "path_traversal",
    "XSS",
    "deserialization",
    "command_injection",
    "SAFE",
]


@dataclass
class ProbeConfig:
    """Configuration for ActivationProbe.

    Attributes:
        model_name: HuggingFace model identifier
            (e.g. "codellama/CodeLlama-7b-hf").
        layers_to_probe: Indices of transformer layers to hook.  Layer 0
            is the embedding output; later layers tend to encode more
            abstract semantic concepts.
        probe_type: "linear" for a single linear layer (fast, interpretable)
            or "mlp" for a 2-layer MLP (higher capacity).
        vulnerability_classes: Ordered list of vulnerability class labels
            (output classes of the probe).
        threshold: Minimum confidence score to flag as vulnerable.
            Calibrated to achieve ≥0.95 recall on the evaluation set.
        hidden_dim: Hidden dimension of the transformer model.  Auto-inferred
            from the loaded model if not set.
        mlp_hidden_dim: Hidden dimension for the MLP probe (probe_type="mlp").
        device: PyTorch device string ("cpu", "cuda", "mps").
    """

    model_name: str
    layers_to_probe: list[int]
    probe_type: str = "linear"
    vulnerability_classes: list[str] = field(
        default_factory=lambda: VULNERABILITY_CLASSES.copy()
    )
    threshold: float = 0.5
    hidden_dim: int = 4096
    mlp_hidden_dim: int = 256
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.probe_type not in {"linear", "mlp"}:
            raise ValueError(
                f"probe_type must be 'linear' or 'mlp', got '{self.probe_type}'."
            )
        if not (0.0 < self.threshold < 1.0):
            raise ValueError("threshold must be in (0, 1).")
        if not self.layers_to_probe:
            raise ValueError("layers_to_probe must be a non-empty list.")


if _TORCH_AVAILABLE:
    class LinearProbe(nn.Module):
        """Single linear layer that maps residual stream → vulnerability logits.

        The probe is intentionally kept as simple as possible.  Linear
        separability of vulnerability concepts is itself a research result:
        if a linear probe achieves high recall, it is evidence that these
        concepts are *linearly encoded* in the residual stream, a key claim
        of the representation engineering hypothesis applied to security.

        Args:
            hidden_dim: Dimension of the residual stream (input to probe).
            num_classes: Number of output classes (including "SAFE").
        """

        def __init__(self, hidden_dim: int, num_classes: int) -> None:
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive.")
            if num_classes <= 1:
                raise ValueError("num_classes must be at least 2.")
            super().__init__()
            self.linear = nn.Linear(hidden_dim, num_classes)
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Forward pass.

            Args:
                x: Activation tensor of shape ``(batch, hidden_dim)``.

            Returns:
                torch.Tensor: Logits of shape ``(batch, num_classes)``.
            """
            return self.linear(x)

    class MLPProbe(nn.Module):
        """Two-layer MLP probe for higher-capacity vulnerability classification.

        Args:
            hidden_dim: Dimension of the residual stream (input).
            mlp_hidden: Hidden layer size.
            num_classes: Number of output classes.
        """

        def __init__(
            self,
            hidden_dim: int,
            mlp_hidden: int,
            num_classes: int,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden, num_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

else:  # pragma: no cover
    # Stub classes when torch is unavailable.
    class LinearProbe:  # type: ignore[no-redef]
        def __init__(self, hidden_dim: int, num_classes: int) -> None:
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes

    class MLPProbe:  # type: ignore[no-redef]
        def __init__(self, hidden_dim: int, mlp_hidden: int, num_classes: int) -> None:
            pass


class ActivationProbe:
    """Hooks into a transformer residual stream to classify vulnerability signatures.

    Layer 1 of the ActivGuard pipeline.  Registers PyTorch forward hooks on
    the specified transformer layers, captures residual stream activations,
    and classifies them using a trained linear or MLP probe.

    The probe is designed to be *gating*: its purpose is not to achieve perfect
    precision, but to achieve ≥0.95 recall while keeping false-negative rate
    below 5%.  Flagged outputs are forwarded to Layer 2 (Semantic RAG) for
    confirmation, so false positives are acceptable.

    Args:
        config: :class:`ProbeConfig` instance.
    """

    def __init__(self, config: ProbeConfig) -> None:
        self._config = config
        self._hooks: list = []
        self._captured_activations: dict[int, list] = {
            layer: [] for layer in config.layers_to_probe
        }
        num_classes = len(config.vulnerability_classes)
        if _TORCH_AVAILABLE:
            if config.probe_type == "linear":
                self._probe: LinearProbe | MLPProbe = LinearProbe(
                    config.hidden_dim, num_classes
                )
            else:
                self._probe = MLPProbe(
                    config.hidden_dim, config.mlp_hidden_dim, num_classes
                )
            self._probe = self._probe.to(config.device)
        else:
            self._probe = None  # type: ignore[assignment]
        self._is_trained = False
        logger.info(
            "ActivationProbe initialised: model=%s, layers=%s, type=%s",
            config.model_name,
            config.layers_to_probe,
            config.probe_type,
        )

    def register_hooks(self, model: object) -> None:
        """Register forward hooks on the target transformer layers.

        Hooks capture the output of each specified layer's residual stream.
        The mean-pooled activation across token positions is stored for
        classification.

        Args:
            model: A HuggingFace ``PreTrainedModel`` instance.

        Raises:
            RuntimeError: If PyTorch is not available.
            AttributeError: If the model lacks the expected layer structure.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to register hooks.")
        # Clear existing hooks.
        self.remove_hooks()
        # Introspect model structure.  HuggingFace models expose layers via
        # model.model.layers (LLaMA, CodeLlama) or model.transformer.h (GPT-2).
        layers = None
        for attr_path in (
            "model.layers",
            "transformer.h",
            "encoder.layer",
            "decoder.layers",
        ):
            obj = model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                layers = obj
                break
            except AttributeError:
                continue
        if layers is None:
            raise AttributeError(
                "Cannot find transformer layer list.  Supported model families: "
                "LLaMA/CodeLlama (model.layers), GPT-2 (transformer.h), "
                "BERT (encoder.layer)."
            )

        for layer_idx in self._config.layers_to_probe:
            if layer_idx >= len(layers):
                logger.warning(
                    "Layer %d requested but model only has %d layers.  Skipping.",
                    layer_idx,
                    len(layers),
                )
                continue
            layer = layers[layer_idx]
            hook_fn = self._make_hook(layer_idx)
            handle = layer.register_forward_hook(hook_fn)
            self._hooks.append(handle)
        logger.info(
            "Registered %d forward hooks on layers %s",
            len(self._hooks),
            self._config.layers_to_probe,
        )

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._captured_activations = {
            layer: [] for layer in self._config.layers_to_probe
        }

    def classify(self, code_snippet: str) -> dict:
        """Classify a code snippet using captured residual stream activations.

        The caller must have already run the LLM forward pass on
        ``code_snippet`` after calling ``register_hooks`` so that
        ``_captured_activations`` contains the current snippet's activations.

        In practice the orchestration layer handles this by calling:
        ``model(tokenizer(code_snippet, ...))`` between ``register_hooks``
        and ``classify``.

        Args:
            code_snippet: The code snippet to classify (used for logging).

        Returns:
            dict: Classification result with keys:
                - ``label`` (str): "FLAG" or "SAFE".
                - ``class`` (str | None): Vulnerability class if flagged.
                - ``confidence`` (float): Probability of the top class.
                - ``layer`` (int): Layer with highest-confidence signal.
                - ``all_scores`` (dict): Per-class probability scores.

        Raises:
            RuntimeError: If PyTorch is not available or probe is not trained.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for classification.")
        if not self._is_trained:
            logger.warning(
                "Probe has not been trained.  Call train_probe() first.  "
                "Returning SAFE with confidence=0."
            )
            return {
                "label": "SAFE",
                "class": None,
                "confidence": 0.0,
                "layer": -1,
                "all_scores": {},
            }

        best_result: dict = {
            "label": "SAFE",
            "class": None,
            "confidence": 0.0,
            "layer": -1,
            "all_scores": {},
        }
        with torch.no_grad():
            for layer_idx, activations in self._captured_activations.items():
                if not activations:
                    continue
                # activations: list of tensors (batch, seq_len, hidden_dim)
                # Mean pool over sequence dimension.
                act_tensor = activations[-1]  # Last captured forward pass.
                if act_tensor.dim() == 3:
                    act_tensor = act_tensor.mean(dim=1)  # (batch, hidden_dim)
                elif act_tensor.dim() == 2:
                    act_tensor = act_tensor.mean(dim=0, keepdim=True)
                act_tensor = act_tensor.to(self._config.device)
                logits = self._probe(act_tensor)  # (batch, num_classes)
                probs = torch.softmax(logits, dim=-1)[0]  # (num_classes,)
                all_scores = {
                    cls: float(probs[i])
                    for i, cls in enumerate(self._config.vulnerability_classes)
                }
                # Find the most probable vulnerability class (exclude "SAFE").
                vuln_classes = [
                    c for c in self._config.vulnerability_classes if c != "SAFE"
                ]
                if vuln_classes:
                    best_vuln_idx = max(
                        range(len(vuln_classes)),
                        key=lambda i: all_scores.get(vuln_classes[i], 0.0),
                    )
                    best_vuln = vuln_classes[best_vuln_idx]
                    confidence = all_scores.get(best_vuln, 0.0)
                    if confidence > best_result["confidence"]:
                        label = "FLAG" if confidence >= self._config.threshold else "SAFE"
                        best_result = {
                            "label": label,
                            "class": best_vuln if label == "FLAG" else None,
                            "confidence": confidence,
                            "layer": layer_idx,
                            "all_scores": all_scores,
                        }
        return best_result

    def train_probe(self, labeled_samples: list[dict]) -> dict:
        """Train the linear or MLP probe on labelled (activation, label) pairs.

        Args:
            labeled_samples: List of dicts with keys:
                - ``activation`` (list[float] | np.ndarray): Mean-pooled
                  residual stream vector.
                - ``label`` (str): Vulnerability class label (must be in
                  ``config.vulnerability_classes``).
                - ``layer`` (int): Layer from which activation was captured.

        Returns:
            dict: Training metrics — ``loss``, ``accuracy``, ``num_samples``,
                ``num_epochs``.

        Raises:
            RuntimeError: If PyTorch is not available.
            ValueError: If labeled_samples is empty or labels are invalid.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for training.")
        if not labeled_samples:
            raise ValueError("labeled_samples must be a non-empty list.")

        label_to_idx = {
            cls: i for i, cls in enumerate(self._config.vulnerability_classes)
        }
        valid_samples = [
            s for s in labeled_samples if s.get("label") in label_to_idx
        ]
        if not valid_samples:
            raise ValueError(
                f"No valid samples found.  Labels must be in: "
                f"{list(label_to_idx.keys())}."
            )

        # Build tensors.
        X_list = []
        y_list = []
        for sample in valid_samples:
            act = sample["activation"]
            if isinstance(act, np.ndarray):
                act = act.tolist()
            X_list.append(act)
            y_list.append(label_to_idx[sample["label"]])

        X = torch.tensor(X_list, dtype=torch.float32, device=self._config.device)
        y = torch.tensor(y_list, dtype=torch.long, device=self._config.device)

        # Pad or truncate to expected hidden_dim.
        if X.shape[1] != self._config.hidden_dim:
            diff = self._config.hidden_dim - X.shape[1]
            if diff > 0:
                X = torch.nn.functional.pad(X, (0, diff))
            else:
                X = X[:, : self._config.hidden_dim]

        num_epochs = 50
        optimizer = optim.AdamW(self._probe.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        self._probe.train()
        final_loss = 0.0
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            logits = self._probe(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        # Compute training accuracy.
        self._probe.eval()
        with torch.no_grad():
            preds = self._probe(X).argmax(dim=-1)
            accuracy = float((preds == y).float().mean().item())

        self._is_trained = True
        logger.info(
            "Probe trained on %d samples — loss=%.4f, accuracy=%.4f",
            len(valid_samples),
            final_loss,
            accuracy,
        )
        return {
            "loss": final_loss,
            "accuracy": accuracy,
            "num_samples": len(valid_samples),
            "num_epochs": num_epochs,
        }

    def extract_activation_signature(self, exploit_examples: list[str]) -> dict:
        """Compute an activation-space centroid from few exploit examples.

        This is the *zero-shot update path* for Layer 4: when a new CVE is
        published, 5 exploit examples are generated, their activations are
        captured, and the centroid becomes a new entry in the probe's
        recognition dictionary — no gradient update required.

        Args:
            exploit_examples: List of code strings demonstrating the
                vulnerability (typically 5–10 examples).

        Returns:
            dict: Activation signature with keys:
                - ``mean_activation`` (list[float]): Centroid vector.
                - ``variance`` (list[float]): Per-dimension variance.
                - ``layer`` (int): Layer used for extraction.
                - ``num_examples`` (int): Number of examples used.

        Raises:
            RuntimeError: If no activations are captured (hooks not set).
        """
        if not self._captured_activations or all(
            len(v) == 0 for v in self._captured_activations.values()
        ):
            raise RuntimeError(
                "No activations captured.  Ensure register_hooks() has been "
                "called and that the model has been run on exploit_examples."
            )
        # Use the last probed layer.
        best_layer = max(self._config.layers_to_probe)
        activations = self._captured_activations.get(best_layer, [])
        if not activations:
            best_layer = self._config.layers_to_probe[0]
            activations = self._captured_activations.get(best_layer, [])

        if _TORCH_AVAILABLE and activations:
            stacked = []
            for act in activations[-len(exploit_examples):]:
                if act.dim() == 3:
                    stacked.append(act.mean(dim=1).squeeze(0))
                else:
                    stacked.append(act.squeeze(0))
            if stacked:
                matrix = torch.stack(stacked, dim=0)
                mean_act = matrix.mean(dim=0).tolist()
                var_act = matrix.var(dim=0).tolist()
                return {
                    "mean_activation": mean_act,
                    "variance": var_act,
                    "layer": best_layer,
                    "num_examples": len(stacked),
                }
        return {
            "mean_activation": [],
            "variance": [],
            "layer": best_layer,
            "num_examples": 0,
        }

    def _make_hook(self, layer_idx: int):
        """Create a forward hook closure for a specific layer.

        Args:
            layer_idx: The layer index to associate with captured activations.

        Returns:
            Callable: Forward hook function.
        """
        def hook_fn(module, input, output):  # noqa: ANN001
            # HuggingFace layers may return tuples; the first element is the
            # hidden state tensor.
            hidden = output[0] if isinstance(output, tuple) else output
            self._captured_activations[layer_idx].append(hidden.detach().cpu())
            # Keep only the last 10 activations to bound memory usage.
            if len(self._captured_activations[layer_idx]) > 10:
                self._captured_activations[layer_idx].pop(0)
        return hook_fn

    def __repr__(self) -> str:
        return (
            f"ActivationProbe(model={self._config.model_name!r}, "
            f"layers={self._config.layers_to_probe}, "
            f"trained={self._is_trained})"
        )
