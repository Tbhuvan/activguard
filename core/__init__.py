"""
ActivGuard core module.

Provides the ThreatIndicator dataclass and ACPConnector abstract base class
that form the backbone of the 4-layer vulnerability detection system.
"""

from .threat_indicator import ThreatIndicator
from .acp import ACPConnector, ConnectorRegistry

__all__ = ["ThreatIndicator", "ACPConnector", "ConnectorRegistry"]
