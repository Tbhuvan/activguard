"""
Splunk Connector — pulls security alerts from Splunk Enterprise Security.

Uses the Splunk REST API to query Notable Events (triggered correlations)
from Splunk ES and maps them to ThreatIndicator objects.

Research context:
    Many enterprise security teams already use Splunk ES as their primary SIEM.
    This connector allows ActivGuard to consume context from those existing
    investments: a triggered "SQL Injection Detected" correlation in Splunk ES
    can immediately update the RAG collection and formal property templates,
    closing the loop between runtime detections and code-review prevention.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Generator

import requests

from core.acp import ACPConnector
from core.threat_indicator import ThreatIndicator

logger = logging.getLogger(__name__)

# Splunk severity mapping (urgency field in Notable Events).
_SPLUNK_URGENCY_MAP: dict[str, str] = {
    "critical": "critical",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "informational": "low",
    "unknown": "unknown",
}

# Notable event signature → CWE hint.
_SIGNATURE_CWE_MAP: list[tuple[str, str]] = [
    ("sql injection", "CWE-89"),
    ("sqli", "CWE-89"),
    ("ssrf", "CWE-918"),
    ("server-side request", "CWE-918"),
    ("path traversal", "CWE-22"),
    ("directory traversal", "CWE-22"),
    ("authentication bypass", "CWE-287"),
    ("idor", "CWE-639"),
    ("insecure direct object", "CWE-639"),
    ("xss", "CWE-79"),
    ("cross-site scripting", "CWE-79"),
    ("command injection", "CWE-78"),
    ("privilege escalation", "CWE-269"),
    ("deserialization", "CWE-502"),
    ("buffer overflow", "CWE-120"),
    ("code execution", "CWE-94"),
    ("remote code", "CWE-94"),
]


class SplunkConnector(ACPConnector):
    """Connector for Splunk Enterprise Security (ES) Notable Events.

    Pulls triggered Notable Events from the Splunk ES REST API (saved search
    results) and converts them to ThreatIndicator objects.

    Args:
        splunk_url: Splunk management URL, e.g.
            "https://splunk.example.com:8089".
        username: Splunk username.
        password: Splunk password.
        index: Splunk index to query (default: "notable").
        verify_ssl: Whether to verify TLS certificates.
        earliest_time: Earliest time modifier for queries (e.g. "-24h").
        timeout: HTTP request timeout in seconds.
        max_results: Maximum number of Notable Events to fetch.
    """

    def __init__(
        self,
        splunk_url: str,
        username: str,
        password: str,
        index: str = "notable",
        verify_ssl: bool = True,
        earliest_time: str = "-24h",
        timeout: int = 30,
        max_results: int = 100,
    ) -> None:
        if not splunk_url:
            raise ValueError("splunk_url must be a non-empty string.")
        if not username or not password:
            raise ValueError("Splunk username and password are required.")
        self._splunk_url = splunk_url.rstrip("/")
        self._username = username
        self._password = password
        self._index = index
        self._verify_ssl = verify_ssl
        self._earliest_time = earliest_time
        self._timeout = timeout
        self._max_results = max_results
        self._session = requests.Session()
        self._session.auth = (username, password)
        self._session.headers.update({"Accept": "application/json"})
        self._token: str | None = None

    # ------------------------------------------------------------------
    # ACPConnector interface
    # ------------------------------------------------------------------

    def metadata(self) -> dict:
        """Return connector metadata.

        Returns:
            dict: Connector metadata.
        """
        return {
            "name": "Splunk",
            "format": "Splunk-ES-REST",
            "update_interval": 300,  # 5 minutes
            "threat_categories": [
                "SQLi", "SSRF", "XSS", "auth_bypass", "privilege_escalation",
                "IDOR", "deserialization", "RCE",
            ],
            "version": "1.0.0",
            "base_url": self._splunk_url,
            "requires_api_key": True,
        }

    def pull(self) -> list[ThreatIndicator]:
        """Pull recent Notable Events from Splunk ES.

        Executes a one-shot SPL search against the notable index and
        normalises results to ThreatIndicator.

        Returns:
            list[ThreatIndicator]: Normalised security alerts.
        """
        try:
            events = self._search_notable_events()
            indicators: list[ThreatIndicator] = []
            seen_ids: set[str] = set()
            for event in events:
                indicator = self._parse_notable_event(event)
                if indicator and indicator.id not in seen_ids:
                    seen_ids.add(indicator.id)
                    indicators.append(indicator)
            logger.info(
                "Splunk pull complete — %d indicators", len(indicators)
            )
            return indicators
        except requests.RequestException as exc:
            logger.warning("Splunk pull failed: %s", exc)
            return []

    def stream(self) -> Generator[ThreatIndicator, None, None]:
        """Stream new Splunk Notable Events, polling every 5 minutes.

        Yields:
            ThreatIndicator: New indicators from Splunk.
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
                logger.warning("Splunk stream error: %s", exc)
            time.sleep(300)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _authenticate(self) -> str:
        """Obtain a Splunk session token via Basic auth.

        Returns:
            str: Splunk session key.

        Raises:
            requests.RequestException: On HTTP / network errors.
        """
        url = f"{self._splunk_url}/services/auth/login"
        data = {"username": self._username, "password": self._password, "output_mode": "json"}
        response = self._session.post(
            url, data=data, verify=self._verify_ssl, timeout=self._timeout
        )
        response.raise_for_status()
        token: str = response.json()["sessionKey"]
        self._session.headers["Authorization"] = f"Splunk {token}"
        return token

    def _search_notable_events(self) -> list[dict]:
        """Execute an SPL search for recent Notable Events.

        Returns:
            list[dict]: Raw Notable Event dicts.

        Raises:
            requests.RequestException: On HTTP / network errors.
        """
        # Ensure authenticated.
        if not self._token:
            try:
                self._token = self._authenticate()
            except requests.RequestException as exc:
                logger.warning("Splunk auth failed: %s", exc)
                return []

        search_query = (
            f"search index={self._index} "
            f"earliest={self._earliest_time} "
            f"| head {self._max_results} "
            f"| fields event_id, signature, urgency, _time, src_ip, dest_ip, "
            f"  description, category, cve, rule_name"
        )
        # Submit search job.
        url = f"{self._splunk_url}/services/search/jobs"
        data = {
            "search": search_query,
            "output_mode": "json",
            "exec_mode": "oneshot",
            "count": self._max_results,
        }
        response = self._session.post(
            url, data=data, verify=self._verify_ssl, timeout=self._timeout
        )
        response.raise_for_status()
        result_data = response.json()
        results = result_data.get("results", [])
        return results

    def _parse_notable_event(self, event: dict) -> ThreatIndicator | None:
        """Parse a Splunk Notable Event to a ThreatIndicator.

        Args:
            event: Raw Splunk Notable Event dict (SPL search result).

        Returns:
            ThreatIndicator | None: Normalised indicator, or None on failure.
        """
        try:
            event_id: str = event.get("event_id", "")
            signature: str = event.get("signature", event.get("rule_name", ""))
            urgency: str = event.get("urgency", "unknown").lower()
            description: str = event.get("description", signature)
            cve_ref: str = event.get("cve", "")
            ts_raw = event.get("_time", "")

            if not (event_id or signature):
                return None

            indicator_id = (
                f"SPLUNK-{event_id}"
                if event_id
                else f"SPLUNK-{hash(signature) & 0xFFFFFF}"
            )

            severity = _SPLUNK_URGENCY_MAP.get(urgency, "unknown")

            # CWE inference.
            search_text = (signature + " " + description).lower()
            cwe_ids: list[str] = []
            seen_cwe: set[str] = set()
            for keyword, cwe in _SIGNATURE_CWE_MAP:
                if keyword in search_text and cwe not in seen_cwe:
                    seen_cwe.add(cwe)
                    cwe_ids.append(cwe)
            if cve_ref.startswith("CVE-"):
                cwe_ids.insert(0, cve_ref)

            # Timestamp.
            try:
                timestamp = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
            except (ValueError, TypeError):
                timestamp = datetime.now(tz=timezone.utc)

            semantic_rule = (
                f"Splunk Alert: {signature} "
                f"| Urgency: {urgency} "
                f"| {description[:200]}"
            )

            return ThreatIndicator(
                id=indicator_id,
                source="Splunk",
                severity=severity,
                cwe=cwe_ids,
                affected_patterns=[signature[:200]] if signature else [],
                stix_pattern=None,
                activation_signature={
                    "vuln_class": cwe_ids[0] if cwe_ids else "unknown",
                    "mean_activation": [],
                    "variance": [],
                    "layer": -1,
                    "source": "Splunk",
                    "event_id": event_id,
                },
                semantic_rule=semantic_rule,
                formal_property=None,
                timestamp=timestamp,
                ttl=43200,  # 12 hours
                description=description[:512],
                cvss_score=0.0,
                affected_packages=[],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Splunk parse error for event %s: %s",
                event.get("event_id", "?"),
                exc,
                exc_info=True,
            )
            return None
