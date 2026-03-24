"""
ActivGuard Layer 1 — Activation Probe.

Mechanistic interpretability components for detecting vulnerability signatures
in the residual stream of code-generating LLMs.  Includes:

- ActivationProbe:           hooks into transformer layers and classifies vulns.
- SparseAutoencoder:         learns interpretable features from activation space.
- VulnerabilityFeatureExtractor: maps SAE latent dimensions to vuln concepts.
- CrossModelBenchmark:       evaluates cross-LLM generalisation of probes.
"""

from .activation_probe import ActivationProbe, ProbeConfig, LinearProbe
from .sparse_autoencoder import SparseAutoencoder, VulnerabilityFeatureExtractor
from .cross_model_eval import CrossModelBenchmark, SUPPORTED_MODELS

__all__ = [
    "ActivationProbe",
    "ProbeConfig",
    "LinearProbe",
    "SparseAutoencoder",
    "VulnerabilityFeatureExtractor",
    "CrossModelBenchmark",
    "SUPPORTED_MODELS",
]
