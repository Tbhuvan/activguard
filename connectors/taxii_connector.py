"""
TAXII Connector — pulls STIX 2.1 bundles from TAXII 2.1 servers.

Supports commercial and government threat feeds:
- CISA Automated Indicator Sharing (AIS)
- FS-ISAC (Financial Services ISAC)
- MITRE ATT&CK TAXII server (https://attack-taxii.mitre.org)
- Any TAXII 2.1-compliant server

Research context:
    STIX/TAXII is the OASIS standard for structured threat intelligence
    exchange.  Including a TAXII connector enables ActivGuard to consume
    government and sector-specific threat feeds directly, bridging the gap
    between national cyber defence infrastructure and automated code review.
    The STIX→Viper translation pipeline (stix_to_viper.py) depends on this
    connector for generating formal verification properties from real-world
    threat actor TTPs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Generator

import requests

from core.acp import ACPConnector
from core.threat_indicator import ThreatIndicator

logger = logging.getLogger(__name__)

# Known public TAXII 2.1 servers for reference.
KNOWN_SERVERS = {
    "mitre_attack": "https://attack-taxii.mitre.org/api/v21/",
    "cisa_ais": "https://ais2.cisa.dhs.gov/taxii2/",
}

# STIX CWE label prefix.
_STIX_CWE_PREFIX = "CWE-"

# Pattern extraction: map STIX indicator pattern keywords to CWEs.
_PATTERN_CWE_HINTS: list[tuple[str, str]] = [
    ("sql", "CWE-89"),
    ("injection", "CWE-89"),
    ("ssrf", "CWE-918"),
    ("request.urlopen", "CWE-918"),
    ("httpx.get", "CWE-918"),
    ("path", "CWE-22"),
    ("traversal", "CWE-22"),
    ("idor", "CWE-639"),
    ("auth", "CWE-287"),
    ("deserializ", "CWE-502"),
    ("pickle", "CWE-502"),
    ("eval", "CWE-94"),
    ("exec", "CWE-94"),
    ("xss", "CWE-79"),
    ("overflow", "CWE-120"),
]


class TAXIIConnector(ACPConnector):
    """STIX/TAXII 2.1 connector for commercial and government threat feeds.

    Downloads STIX bundles from a TAXII 2.1 Collection endpoint and
    normalises STIX Indicator objects to ThreatIndicator.

    Args:
        taxii_url: Base URL of the TAXII 2.1 API root or collection endpoint
            (e.g. "https://attack-taxii.mitre.org/api/v21/").
        collection_id: TAXII collection UUID to pull from.
        username: Optional HTTP Basic auth username.
        password: Optional HTTP Basic auth password.
        token: Optional Bearer token for token-based auth.
        timeout: HTTP request timeout in seconds.
        page_limit: Max objects per TAXII page request.
    """

    def __init__(
        self,
        taxii_url: str,
        collection_id: str,
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
        timeout: int = 30,
        page_limit: int = 100,
    ) -> None:
        if not taxii_url:
            raise ValueError("taxii_url must be a non-empty string.")
        if not collection_id:
            raise ValueError("collection_id must be a non-empty string.")
        self._taxii_url = taxii_url.rstrip("/")
        self._collection_id = collection_id
        self._timeout = timeout
        self._page_limit = page_limit
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/taxii+json;version=2.1",
            "Content-Type": "application/taxii+json;version=2.1",
        })
        if username and password:
            self._session.auth = (username, password)
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"

    # ------------------------------------------------------------------
    # ACPConnector interface
    # ------------------------------------------------------------------

    def metadata(self) -> dict:
        """Return connector metadata.

        Returns:
            dict: Connector metadata.
        """
        return {
            "name": "TAXII",
            "format": "STIX-2.1",
            "update_interval": 1800,  # 30 minutes
            "threat_categories": [
                "APT", "malware", "vulnerability", "phishing",
                "supply_chain", "TTP",
            ],
            "version": "1.0.0",
            "base_url": self._taxii_url,
            "collection_id": self._collection_id,
            "requires_api_key": False,
        }

    def pull(self) -> list[ThreatIndicator]:
        """Pull all STIX Indicator objects from the configured collection.

        Returns:
            list[ThreatIndicator]: Normalised indicators from the TAXII feed.
        """
        try:
            objects = self._fetch_collection_objects()
            indicators: list[ThreatIndicator] = []
            seen_ids: set[str] = set()
            for obj in objects:
                if obj.get("type") == "indicator":
                    indicator = self._parse_stix_indicator(obj)
                    if indicator and indicator.id not in seen_ids:
                        seen_ids.add(indicator.id)
                        indicators.append(indicator)
            logger.info(
                "TAXII pull complete — %d indicators from collection %s",
                len(indicators),
                self._collection_id,
            )
            return indicators
        except requests.RequestException as exc:
            logger.warning("TAXII pull failed: %s", exc)
            return []

    def stream(self) -> Generator[ThreatIndicator, None, None]:
        """Stream STIX indicators from the collection, polling every 30 min.

        Yields:
            ThreatIndicator: New indicators as they appear.
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
                logger.warning("TAXII stream error: %s", exc)
            time.sleep(1800)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_collection_objects(self) -> list[dict]:
        """Fetch STIX objects from the TAXII collection.

        Handles pagination via the TAXII 2.1 ``next`` cursor.

        Returns:
            list[dict]: Raw STIX objects.

        Raises:
            requests.RequestException: On HTTP / network errors.
        """
        url = (
            f"{self._taxii_url}/collections/{self._collection_id}/objects/"
        )
        params: dict[str, str | int] = {
            "limit": self._page_limit,
            "match[type]": "indicator",
        }
        all_objects: list[dict] = []
        while url:
            response = self._session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
            objects = data.get("objects", [])
            all_objects.extend(objects)
            # TAXII 2.1 pagination: check for 'next' cursor.
            next_cursor = data.get("next")
            if next_cursor:
                params = {"next": next_cursor, "limit": self._page_limit}
            else:
                break
        return all_objects

    def _parse_stix_indicator(self, obj: dict) -> ThreatIndicator | None:
        """Parse a STIX 2.1 Indicator object to a ThreatIndicator.

        Args:
            obj: STIX 2.1 Indicator object dict.

        Returns:
            ThreatIndicator | None: Normalised indicator, or None on failure.
        """
        try:
            stix_id: str = obj.get("id", "")
            if not stix_id:
                return None

            name: str = obj.get("name", "")
            description: str = obj.get("description", "")
            pattern: str = obj.get("pattern", "")
            pattern_type: str = obj.get("pattern_type", "stix")

            # Extract CWEs from labels.
            labels: list[str] = obj.get("labels", [])
            cwe_ids: list[str] = [
                lbl for lbl in labels if lbl.startswith(_STIX_CWE_PREFIX)
            ]
            if not cwe_ids and pattern:
                cwe_ids = self._infer_cwe_from_pattern(pattern)

            # Kill chain phases for threat category.
            kill_chain = obj.get("kill_chain_phases", [])
            categories = [
                phase.get("phase_name", "") for phase in kill_chain
                if phase.get("phase_name")
            ]

            # Severity: STIX indicators don't have CVSS; map from labels.
            severity = "medium"
            for lbl in labels:
                if lbl.lower() in {"critical", "high", "medium", "low"}:
                    severity = lbl.lower()
                    break

            # Timestamp.
            created_str: str = obj.get("created", "")
            try:
                timestamp = datetime.fromisoformat(
                    created_str.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                timestamp = datetime.now(tz=timezone.utc)

            semantic_rule = (
                f"STIX Indicator [{stix_id}]: {name or description[:100]} "
                f"| Pattern: {pattern[:200]} "
                f"| Categories: {', '.join(categories)}"
            )

            return ThreatIndicator(
                id=stix_id,
                source="TAXII",
                severity=severity,
                cwe=cwe_ids,
                affected_patterns=[pattern[:200]] if pattern else [],
                stix_pattern=pattern if pattern_type == "stix" else None,
                activation_signature={
                    "vuln_class": cwe_ids[0] if cwe_ids else "unknown",
                    "mean_activation": [],
                    "variance": [],
                    "layer": -1,
                    "source": "TAXII",
                    "stix_id": stix_id,
                },
                semantic_rule=semantic_rule,
                formal_property=None,
                timestamp=timestamp,
                ttl=86400,
                description=(description or name)[:512],
                cvss_score=0.0,
                affected_packages=[],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TAXII parse error for STIX object %s: %s",
                obj.get("id", "?"),
                exc,
                exc_info=True,
            )
            return None

    def _infer_cwe_from_pattern(self, pattern: str) -> list[str]:
        """Infer CWE identifiers from a STIX pattern string.

        Args:
            pattern: STIX 2.1 pattern string.

        Returns:
            list[str]: Inferred CWE identifiers.
        """
        lower = pattern.lower()
        seen: set[str] = set()
        results: list[str] = []
        for keyword, cwe in _PATTERN_CWE_HINTS:
            if keyword in lower and cwe not in seen:
                seen.add(cwe)
                results.append(cwe)
        return results
