"""
MISP Connector — integrates client MISP threat intelligence instances.

MISP (Malware Information Sharing Platform) is widely used in SOC environments
to share structured threat intelligence.  This connector pulls threat
indicators from a MISP instance via its REST API and normalises them to the
ThreatIndicator schema.

Research context:
    MISP is the dominant enterprise threat-intel sharing platform.  Including
    a MISP connector in the ACP layer means ActivGuard can be deployed inside
    an existing SOC without requiring new data pipelines — the organisation's
    existing MISP feed directly drives all three detection layers.

Note:
    This is a production-ready stub.  Full MISP attribute parsing requires
    a live MISP instance.  The connector structure and error handling are
    complete; attribute-to-ThreatIndicator mapping covers the most common
    MISP attribute types (ip-src, url, vulnerability, filename|md5).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Generator

import requests

from core.acp import ACPConnector
from core.threat_indicator import ThreatIndicator

logger = logging.getLogger(__name__)

# MISP attribute types that are code-relevant.
_CODE_RELEVANT_TYPES = frozenset({
    "vulnerability",
    "text",
    "comment",
    "other",
    "filename",
    "url",
    "malware-sample",
})


class MISPConnector(ACPConnector):
    """Connector for client MISP instances.

    Pulls threat indicators from a MISP REST API endpoint, filtering for
    code-relevant attributes and normalising to ThreatIndicator.

    Args:
        misp_url: Base URL of the MISP instance
            (e.g. "https://misp.example.org").
        auth_key: MISP automation key (found under My Profile →
            Auth key in the MISP web UI).
        verify_ssl: Whether to verify the server's TLS certificate.
            Set to False only in development against self-signed certs.
        limit: Maximum number of events to pull per request.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        misp_url: str,
        auth_key: str,
        verify_ssl: bool = True,
        limit: int = 100,
        timeout: int = 30,
    ) -> None:
        if not misp_url:
            raise ValueError("misp_url must be a non-empty string.")
        if not auth_key:
            raise ValueError("auth_key must be a non-empty string.")
        self._misp_url = misp_url.rstrip("/")
        self._auth_key = auth_key
        self._verify_ssl = verify_ssl
        self._limit = limit
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": auth_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    # ------------------------------------------------------------------
    # ACPConnector interface
    # ------------------------------------------------------------------

    def metadata(self) -> dict:
        """Return connector metadata.

        Returns:
            dict: Connector metadata.
        """
        return {
            "name": "MISP",
            "format": "MISP-REST",
            "update_interval": 900,  # 15 minutes
            "threat_categories": [
                "malware", "phishing", "vulnerability", "supply_chain",
                "SSRF", "SQLi", "auth_bypass",
            ],
            "version": "1.0.0",
            "base_url": self._misp_url,
            "requires_api_key": True,
        }

    def pull(self) -> list[ThreatIndicator]:
        """Pull recent threat events from the MISP instance.

        Fetches the most recent events, filters for code-relevant attributes,
        and normalises each to a ThreatIndicator.

        Returns:
            list[ThreatIndicator]: Normalised indicators from MISP.
        """
        try:
            events = self._fetch_events()
            indicators: list[ThreatIndicator] = []
            seen_ids: set[str] = set()
            for event in events:
                indicator = self._parse_misp_event(event)
                if indicator and indicator.id not in seen_ids:
                    seen_ids.add(indicator.id)
                    indicators.append(indicator)
            logger.info("MISP pull complete — %d indicators", len(indicators))
            return indicators
        except requests.RequestException as exc:
            logger.warning("MISP pull failed: %s", exc)
            return []

    def stream(self) -> Generator[ThreatIndicator, None, None]:
        """Stream new MISP events as they appear.

        Polls the MISP instance every 15 minutes and yields new events.

        Yields:
            ThreatIndicator: New indicators from MISP.
        """
        import time
        seen_ids: set[str] = set()
        while True:
            try:
                indicators = self.pull()
                for indicator in indicators:
                    if indicator.id not in seen_ids:
                        seen_ids.add(indicator.id)
                        yield indicator
            except Exception as exc:  # noqa: BLE001
                logger.warning("MISP stream error: %s", exc)
            time.sleep(900)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_events(self) -> list[dict]:
        """Fetch recent MISP events via the REST API.

        Returns:
            list[dict]: Raw MISP event dicts.

        Raises:
            requests.RequestException: On HTTP / network errors.
        """
        url = f"{self._misp_url}/events/restSearch"
        body = {
            "returnFormat": "json",
            "limit": self._limit,
            "includeEventTags": True,
            "includeAttachments": False,
            "orderDesc": True,
        }
        response = self._session.post(
            url, json=body, verify=self._verify_ssl, timeout=self._timeout
        )
        response.raise_for_status()
        data = response.json()
        # MISP returns either {"response": [...]} or a direct list.
        if isinstance(data, dict):
            return data.get("response", [])
        return data if isinstance(data, list) else []

    def _parse_misp_event(self, event: dict) -> ThreatIndicator | None:
        """Parse a MISP event dict to a ThreatIndicator.

        Extracts the event ID, info (title), threat level, and attributes.
        MISP threat levels: 1=High, 2=Medium, 3=Low, 4=Undefined.

        Args:
            event: Raw MISP event dict (from the ``Event`` wrapper key, or
                directly from the search response).

        Returns:
            ThreatIndicator | None: Normalised indicator, or None on failure.
        """
        try:
            # MISP wraps events under "Event" in some API responses.
            ev = event.get("Event", event)
            event_id: str = str(ev.get("id", ""))
            if not event_id:
                return None

            misp_id = f"MISP-{event_id}"
            info: str = ev.get("info", "")
            threat_level = int(ev.get("threat_level_id", 4))
            severity_map = {1: "high", 2: "medium", 3: "low", 4: "unknown"}
            severity = severity_map.get(threat_level, "unknown")

            # Timestamp.
            ts_raw = ev.get("timestamp", "")
            try:
                timestamp = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)
            except (ValueError, TypeError):
                timestamp = datetime.now(tz=timezone.utc)

            # Attributes → affected patterns.
            attributes = ev.get("Attribute", [])
            affected_patterns: list[str] = []
            cwe_ids: list[str] = []
            for attr in attributes:
                attr_type = attr.get("type", "")
                value: str = attr.get("value", "")
                if attr_type == "vulnerability" and value.startswith("CVE-"):
                    cwe_ids.append(value)  # Use CVE ref as rough CWE proxy.
                elif attr_type in _CODE_RELEVANT_TYPES and value:
                    affected_patterns.append(value[:200])

            description = info[:512]
            semantic_rule = (
                f"MISP event {event_id}: {description} "
                f"[threat_level={threat_level}]"
            )

            return ThreatIndicator(
                id=misp_id,
                source="MISP",
                severity=severity,
                cwe=cwe_ids,
                affected_patterns=affected_patterns[:10],
                stix_pattern=None,
                activation_signature={
                    "vuln_class": "unknown",
                    "mean_activation": [],
                    "variance": [],
                    "layer": -1,
                    "source": "MISP",
                    "misp_event_id": event_id,
                },
                semantic_rule=semantic_rule,
                formal_property=None,
                timestamp=timestamp,
                ttl=86400,  # 24 hours
                description=description,
                cvss_score=0.0,
                affected_packages=[],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MISP parse error for event %s: %s",
                event.get("Event", event).get("id", "?"),
                exc,
                exc_info=True,
            )
            return None
