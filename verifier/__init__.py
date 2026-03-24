"""
ActivGuard Layer 3 — Formal Verification.

Probe-gated static analysis and formal verification.  Only invoked when
Layer 1 OR Layer 2 flags a code snippet, targeting <15% of all outputs
while catching >90% of real vulnerabilities.

Components:
- FormalChecker:          AST-based and optional Nagini/Viper verification.
- PropertyTemplateEngine: Manages formal property templates per vuln class.
- STIXToViperGenerator:   Auto-generates Viper specs from STIX indicators.
"""

from .formal_check import FormalChecker
from .property_templates import PropertyTemplateEngine, PROPERTY_TEMPLATES
from .stix_to_viper import STIXToViperGenerator

__all__ = [
    "FormalChecker",
    "PropertyTemplateEngine",
    "PROPERTY_TEMPLATES",
    "STIXToViperGenerator",
]
