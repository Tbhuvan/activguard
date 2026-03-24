"""
Sparse Autoencoder — interpretable feature extraction from transformer activations.

Learns a sparse overcomplete dictionary over the residual stream of a coding
LLM.  Each latent dimension in the learned dictionary ideally corresponds to a
single human-interpretable concept (e.g. "SQL query construction",
"URL fetch", "file path concatenation").

Research background:
    Anthropic's superposition hypothesis (arXiv:2209.11895) argues that neural
    networks represent more concepts than they have dimensions by encoding
    multiple concepts in superposition.  Sparse autoencoders are the leading
    method for decomposing this superposition into monosemantic features.

    We apply this technique to *security-relevant* features: our goal is to
    find latent dimensions that activate specifically in the presence of
    vulnerability-prone code constructs.  This provides both interpretability
    (researchers can inspect what each dimension encodes) and a compact update
    path for new vulnerabilities — finding the feature is enough to define a
    new detection rule.

Research questions:
    RQ4: Do vulnerability-relevant features form monosemantic dimensions in
         the SAE latent space?
    RQ5: What fraction of the SAE dictionary is required to explain
         vulnerability predictions made by the linear probe?
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available.  SAE will run in stub mode.")


if _TORCH_AVAILABLE:
    class SparseAutoencoder(nn.Module):
        """Sparse overcomplete autoencoder for transformer activation analysis.

        Learns a dictionary of hidden_dim features over the input_dim-dimensional
        residual stream.  A sparsity penalty (L1 on activations) encourages each
        input to activate only a small number of dictionary features.

        Architecture:
            Encoder: Linear(input_dim → hidden_dim) + ReLU
            Decoder: Linear(hidden_dim → input_dim)

        The reconstruction loss + L1 sparsity penalty training objective follows
        the setup in Cunningham et al. (arXiv:2309.08600).

        Args:
            input_dim: Residual stream dimension (e.g. 4096 for LLaMA-7B).
            hidden_dim: SAE dictionary size; typically 4× or 8× input_dim.
            sparsity_penalty: Coefficient for the L1 activation penalty.
                Higher values → sparser activations.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            sparsity_penalty: float = 0.01,
        ) -> None:
            if input_dim <= 0:
                raise ValueError("input_dim must be positive.")
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive.")
            if sparsity_penalty < 0:
                raise ValueError("sparsity_penalty must be non-negative.")
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.sparsity_penalty = sparsity_penalty
            # Encoder: bias pre-activation
            self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
            # Decoder: constrain columns to unit norm (enforced in forward)
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
            # Pre-encoder bias for centring (following Anthropic SAE design)
            self.pre_bias = nn.Parameter(torch.zeros(input_dim))
            self._init_weights()

        def _init_weights(self) -> None:
            """Initialise weights for stable training."""
            nn.init.kaiming_uniform_(self.encoder.weight)
            nn.init.zeros_(self.encoder.bias)
            # Decoder initialised as transpose of encoder.
            self.decoder.weight.data = self.encoder.weight.data.T.clone()

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            """Encode input activations to sparse feature space.

            Args:
                x: Input tensor of shape ``(..., input_dim)``.

            Returns:
                torch.Tensor: Sparse feature activations ``(..., hidden_dim)``.
                    Values are non-negative (ReLU).
            """
            x_centred = x - self.pre_bias
            return F.relu(self.encoder(x_centred))

        def decode(self, z: "torch.Tensor") -> "torch.Tensor":
            """Decode sparse features back to the original activation space.

            Args:
                z: Sparse feature tensor of shape ``(..., hidden_dim)``.

            Returns:
                torch.Tensor: Reconstructed activations ``(..., input_dim)``.
            """
            # Normalise decoder columns to unit norm before decoding.
            normed_weight = F.normalize(self.decoder.weight, dim=0)
            return F.linear(z, normed_weight.T) + self.pre_bias

        def forward(
            self, x: "torch.Tensor"
        ) -> "tuple[torch.Tensor, torch.Tensor]":
            """Full encode–decode pass.

            Args:
                x: Input activation tensor ``(batch, input_dim)``.

            Returns:
                tuple: ``(x_reconstructed, z)`` where ``x_reconstructed`` has
                    the same shape as ``x`` and ``z`` has shape
                    ``(batch, hidden_dim)``.
            """
            z = self.encode(x)
            x_reconstructed = self.decode(z)
            return x_reconstructed, z

        def loss(
            self, x: "torch.Tensor", x_reconstructed: "torch.Tensor", z: "torch.Tensor"
        ) -> "torch.Tensor":
            """Compute SAE training loss: reconstruction + L1 sparsity.

            Args:
                x: Original activations.
                x_reconstructed: Reconstructed activations from decoder.
                z: Sparse feature activations from encoder.

            Returns:
                torch.Tensor: Scalar loss.
            """
            reconstruction_loss = F.mse_loss(x_reconstructed, x)
            sparsity_loss = self.sparsity_penalty * z.abs().mean()
            return reconstruction_loss + sparsity_loss

else:  # pragma: no cover — stub when torch unavailable
    class SparseAutoencoder:  # type: ignore[no-redef]
        """Stub SAE when PyTorch is not installed."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            sparsity_penalty: float = 0.01,
        ) -> None:
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.sparsity_penalty = sparsity_penalty

        def encode(self, x: Any) -> Any:
            raise RuntimeError("PyTorch required for SAE.")

        def decode(self, z: Any) -> Any:
            raise RuntimeError("PyTorch required for SAE.")

        def forward(self, x: Any) -> tuple:
            raise RuntimeError("PyTorch required for SAE.")


class VulnerabilityFeatureExtractor:
    """Extracts human-interpretable vulnerability features from SAE latent space.

    After training the SAE on a large corpus of coding LLM activations, each
    latent dimension should correspond to a single concept.  This extractor
    identifies which dimensions activate strongly for vulnerability-prone code
    and labels them using automated and manual analysis.

    Research workflow:
        1. Train SAE on activations from 10k safe code snippets.
        2. Run SAE on vulnerability dataset; find dimensions that fire
           significantly more for vulnerable code.
        3. Use ``label_features`` to characterise high-activating dimensions.
        4. Register those dimensions as vulnerability detectors.

    Args:
        sae: A trained :class:`SparseAutoencoder` instance.
    """

    def __init__(self, sae: SparseAutoencoder) -> None:
        if not isinstance(sae, SparseAutoencoder):
            raise TypeError("sae must be a SparseAutoencoder instance.")
        self._sae = sae
        # feature_idx → human-readable label (populated by label_features)
        self._feature_labels: dict[int, str] = {}
        # feature_idx → vulnerability class (populated by assign_vuln_class)
        self._vuln_feature_map: dict[int, str] = {}

    def get_top_features(
        self,
        activation: "Any",
        top_k: int = 10,
    ) -> list[dict]:
        """Return the top-k most active SAE features for a given activation.

        Args:
            activation: Residual stream activation tensor ``(hidden_dim,)``
                or ``(1, hidden_dim)``.  May also be a Python list or
                numpy array.
            top_k: Number of top features to return.

        Returns:
            list[dict]: List of feature dicts, each with keys:
                - ``feature_idx`` (int): Index in SAE dictionary.
                - ``activation_value`` (float): Activation magnitude.
                - ``label`` (str): Human-readable label if assigned.
                - ``vuln_class`` (str | None): Vulnerability class if mapped.

        Raises:
            RuntimeError: If PyTorch is not available.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for feature extraction.")
        if not isinstance(activation, torch.Tensor):
            activation = torch.tensor(activation, dtype=torch.float32)
        if activation.dim() == 1:
            activation = activation.unsqueeze(0)
        with torch.no_grad():
            z = self._sae.encode(activation).squeeze(0)
        values, indices = torch.topk(z, min(top_k, len(z)))
        features = []
        for val, idx in zip(values.tolist(), indices.tolist()):
            features.append({
                "feature_idx": int(idx),
                "activation_value": float(val),
                "label": self._feature_labels.get(int(idx), f"feature_{idx}"),
                "vuln_class": self._vuln_feature_map.get(int(idx)),
            })
        return features

    def label_features(self, feature_idx: int, examples: list[str]) -> str:
        """Assign a human-readable label to a SAE feature dimension.

        In a full research pipeline, this would call an LLM to summarise the
        common theme across ``examples``.  Here we implement keyword-based
        labelling for reproducibility without an API key.

        Args:
            feature_idx: Index of the SAE feature dimension.
            examples: Code snippets that strongly activate this feature.

        Returns:
            str: Human-readable label assigned to the feature.
        """
        if not examples:
            label = f"feature_{feature_idx}_unlabelled"
            self._feature_labels[feature_idx] = label
            return label
        combined = "\n".join(examples).lower()
        # Keyword-based auto-labelling.
        label_candidates = [
            ("sql", "sql-query-construction"),
            ("execute", "database-execute"),
            ("request.get", "http-get-request"),
            ("urlopen", "url-open"),
            ("httpx", "httpx-client-call"),
            ("os.path.join", "path-join"),
            ("open(", "file-open"),
            ("pickle", "pickle-deserialise"),
            ("yaml.load", "yaml-load"),
            ("subprocess", "subprocess-call"),
            ("session", "session-access"),
            ("user_id", "user-id-lookup"),
            ("password", "credential-handling"),
            ("token", "token-handling"),
            ("eval(", "eval-call"),
            ("exec(", "exec-call"),
        ]
        for keyword, candidate_label in label_candidates:
            if keyword in combined:
                label = candidate_label
                self._feature_labels[feature_idx] = label
                return label
        label = f"feature_{feature_idx}_general-code"
        self._feature_labels[feature_idx] = label
        return label

    def assign_vuln_class(
        self, feature_idx: int, vuln_class: str
    ) -> None:
        """Mark a SAE feature dimension as associated with a vulnerability class.

        Args:
            feature_idx: SAE dictionary index.
            vuln_class: Vulnerability class (e.g. "SQLi", "IDOR").
        """
        self._vuln_feature_map[feature_idx] = vuln_class
        logger.info(
            "Feature %d mapped to vulnerability class '%s'",
            feature_idx,
            vuln_class,
        )

    def get_vuln_feature_summary(self) -> dict[str, list[int]]:
        """Return a summary of which SAE features are associated with each vuln class.

        Returns:
            dict: Maps vulnerability class → list of feature indices.
        """
        summary: dict[str, list[int]] = {}
        for idx, cls in self._vuln_feature_map.items():
            summary.setdefault(cls, []).append(idx)
        return summary

    def __repr__(self) -> str:
        return (
            f"VulnerabilityFeatureExtractor("
            f"sae_hidden_dim={self._sae.hidden_dim}, "
            f"labelled_features={len(self._feature_labels)}, "
            f"vuln_features={len(self._vuln_feature_map)})"
        )
