"""
ACP — ActivGuard Connector Protocol.

Defines the abstract interface that every threat-intelligence connector must
implement and the ConnectorRegistry that routes incoming indicators to the
three detection layers.

The ACP design principle is *schema-first heterogeneity*: regardless of the
upstream wire format (NVD JSON 5.0, OSV schema, STIX 2.1, MISP event, Splunk
ES alert), every connector normalises its output to a ThreatIndicator before
returning.  This allows the detection layers to be completely agnostic about
the intelligence source and updated through a single, typed interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Generator

from .threat_indicator import ThreatIndicator

logger = logging.getLogger(__name__)


class ACPConnector(ABC):
    """Abstract base class for all ActivGuard Connector Protocol connectors.

    Subclasses must implement :meth:`metadata`, :meth:`pull`, and
    :meth:`stream`.  The ``pull`` method is used for scheduled batch ingestion
    (e.g. hourly NVD sync), while ``stream`` supports real-time zero-day
    injection into the live ChromaDB collection.

    Research context:
        The ACP is designed to be the "sensory organ" of the system.
        Its separation from the detection layers is essential for measuring
        the *latency* between public disclosure (NVD publication timestamp)
        and detection capability update — one of the PhD research metrics.
    """

    @abstractmethod
    def metadata(self) -> dict:
        """Return connector metadata for registry introspection.

        Returns:
            dict: Must contain the following keys:
                - ``name`` (str): Human-readable connector name.
                - ``format`` (str): Wire format, e.g. "NVD-JSON-5.0", "OSV",
                  "STIX-2.1", "MISP-REST".
                - ``update_interval`` (int): Recommended pull interval in
                  seconds.
                - ``threat_categories`` (list[str]): Vulnerability classes
                  this connector covers.
                - ``version`` (str): Connector implementation version.
        """

    @abstractmethod
    def pull(self) -> list[ThreatIndicator]:
        """One-shot pull of the latest indicators from the upstream source.

        Normalises the upstream format to :class:`~core.ThreatIndicator`
        objects before returning.  Implementations must handle network errors
        gracefully and return an empty list (not raise) on transient failures,
        logging the error at WARNING level.

        Returns:
            list[ThreatIndicator]: Zero or more normalised indicators.
        """

    @abstractmethod
    def stream(self) -> Generator[ThreatIndicator, None, None]:
        """Real-time generator of indicators from the upstream source.

        Yields indicators as they arrive.  Implementations should handle
        reconnection logic internally and yield nothing (pause iteration)
        during network outages rather than propagating exceptions.

        Yields:
            ThreatIndicator: Normalised indicator as it arrives.
        """


class ConnectorRegistry:
    """Registry that manages multiple ACP connectors and routes indicators.

    The registry maintains a dictionary of named connectors and provides
    convenience methods for bulk pulls and source-specific lookups.  It is
    designed to be the single injection point used by the orchestration layer
    to populate Layers 1–3.

    Attributes:
        _connectors: Internal dict mapping source name → ACPConnector instance.

    Example::

        registry = ConnectorRegistry()
        registry.register(NVDConnector(api_key="..."))
        registry.register(OSVConnector(ecosystems=["PyPI"]))
        all_indicators = registry.pull_all()
    """

    def __init__(self) -> None:
        self._connectors: dict[str, ACPConnector] = {}

    def register(self, connector: ACPConnector) -> None:
        """Register a connector under its declared name.

        Args:
            connector: An instance implementing :class:`ACPConnector`.

        Raises:
            TypeError: If ``connector`` does not implement ACPConnector.
            ValueError: If a connector with the same name is already registered.
        """
        if not isinstance(connector, ACPConnector):
            raise TypeError(
                f"Expected ACPConnector subclass, got {type(connector).__name__}."
            )
        meta = connector.metadata()
        name: str = meta.get("name", type(connector).__name__)
        if name in self._connectors:
            raise ValueError(
                f"A connector named '{name}' is already registered.  "
                "Deregister it first or use a unique name."
            )
        self._connectors[name] = connector
        logger.info("Registered connector: %s (format=%s)", name, meta.get("format"))

    def deregister(self, name: str) -> None:
        """Remove a connector from the registry.

        Args:
            name: The connector's declared name.

        Raises:
            KeyError: If no connector with that name is registered.
        """
        if name not in self._connectors:
            raise KeyError(f"No connector named '{name}' is registered.")
        del self._connectors[name]
        logger.info("Deregistered connector: %s", name)

    def pull_all(self) -> list[ThreatIndicator]:
        """Pull indicators from every registered connector.

        Failures in one connector do not abort the pull from others; the
        error is logged at WARNING level and an empty list is merged for
        that connector.

        Returns:
            list[ThreatIndicator]: Combined list from all connectors,
                deduplicated by ``ThreatIndicator.id``.
        """
        seen_ids: set[str] = set()
        results: list[ThreatIndicator] = []
        for name, connector in self._connectors.items():
            try:
                indicators = connector.pull()
                for indicator in indicators:
                    if indicator.id not in seen_ids:
                        seen_ids.add(indicator.id)
                        results.append(indicator)
                logger.info(
                    "Pulled %d indicators from %s", len(indicators), name
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Connector '%s' pull failed: %s", name, exc, exc_info=True
                )
        return results

    def get_by_source(self, source: str) -> ACPConnector:
        """Retrieve a registered connector by its source name.

        Args:
            source: The connector's declared name (as returned by
                ``metadata()["name"]``).

        Returns:
            ACPConnector: The registered connector instance.

        Raises:
            KeyError: If no connector with that name is registered.
        """
        if source not in self._connectors:
            raise KeyError(
                f"No connector named '{source}'. "
                f"Registered: {list(self._connectors.keys())}"
            )
        return self._connectors[source]

    def list_connectors(self) -> list[dict]:
        """Return metadata for all registered connectors.

        Returns:
            list[dict]: One metadata dict per registered connector.
        """
        return [c.metadata() for c in self._connectors.values()]

    def __len__(self) -> int:
        return len(self._connectors)

    def __repr__(self) -> str:
        names = list(self._connectors.keys())
        return f"ConnectorRegistry(connectors={names})"
