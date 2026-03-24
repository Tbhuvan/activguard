"""
NVD Connector — pulls CVEs from the NIST National Vulnerability Database.

Uses the NVD CVE API 2.0 (https://services.nvd.nist.gov/rest/json/cves/2.0).
Without an API key the endpoint allows 5 requests per 30-second rolling
window; with an API key this rises to 50 requests / 30 s.

Research context:
    The NVD connector is the primary source for code-relevant CVEs.  By
    filtering on keywords that correspond to well-known vulnerability classes
    (SQLi, SSRF, IDOR, etc.) and mapping CVSS scores to severity labels, it
    feeds a curated stream of indicators into all three detection layers.
    Measuring the latency between NVD publication date and ActivGuard
    detection-capability update is a concrete PhD evaluation metric.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Generator

import requests

from core.acp import ACPConnector
from core.threat_indicator import ThreatIndicator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# Default keyword filters for code-relevant vulnerability classes.
DEFAULT_KEYWORD_FILTERS: list[str] = [
    "injection",
    "buffer overflow",
    "use-after-free",
    "IDOR",
    "SSRF",
    "authentication bypass",
    "path traversal",
    "SQL injection",
    "cross-site scripting",
    "deserialization",
]

# CWE keyword mapping for _infer_cwe_from_description
_KEYWORD_CWE_MAP: list[tuple[str, str]] = [
    ("sql injection", "CWE-89"),
    ("sqli", "CWE-89"),
    ("sql", "CWE-89"),
    ("idor", "CWE-639"),
    ("insecure direct object", "CWE-639"),
    ("ssrf", "CWE-918"),
    ("server-side request forgery", "CWE-918"),
    ("path traversal", "CWE-22"),
    ("directory traversal", "CWE-22"),
    ("authentication bypass", "CWE-287"),
    ("improper authentication", "CWE-287"),
    ("buffer overflow", "CWE-120"),
    ("stack overflow", "CWE-121"),
    ("heap overflow", "CWE-122"),
    ("use-after-free", "CWE-416"),
    ("use after free", "CWE-416"),
    ("null pointer", "CWE-476"),
    ("xss", "CWE-79"),
    ("cross-site scripting", "CWE-79"),
    ("open redirect", "CWE-601"),
    ("command injection", "CWE-78"),
    ("code injection", "CWE-94"),
    ("xml injection", "CWE-91"),
    ("xpath injection", "CWE-643"),
    ("ldap injection", "CWE-90"),
    ("deserialization", "CWE-502"),
    ("xxe", "CWE-611"),
    ("xml external entity", "CWE-611"),
    ("race condition", "CWE-362"),
    ("integer overflow", "CWE-190"),
    ("format string", "CWE-134"),
    ("hardcoded credential", "CWE-798"),
    ("hardcoded password", "CWE-259"),
    ("cleartext", "CWE-319"),
    ("insecure randomness", "CWE-338"),
    ("csrf", "CWE-352"),
    ("cross-site request forgery", "CWE-352"),
    ("privilege escalation", "CWE-269"),
    ("improper authoriz", "CWE-285"),
    ("missing authorization", "CWE-862"),
]

# Code-level pattern templates per CWE for Layer 1 / Layer 2 consumption.
_CWE_PATTERN_MAP: dict[str, list[str]] = {
    "CWE-89": [
        "f\"{query}\"",
        "% user_input",
        "+ user_id +",
        "execute(sql_string)",
        "raw_query",
    ],
    "CWE-639": [
        "request.args.get('id')",
        "obj = Model.get(id=user_input)",
        "no ownership check before access",
        "direct object reference without auth",
    ],
    "CWE-918": [
        "requests.get(url)",
        "httpx.get(user_url)",
        "urllib.request.urlopen(user_input)",
        "no allowlist validation for URL",
    ],
    "CWE-22": [
        "open(base_path + user_input)",
        "os.path.join(root, filename)",
        "../ traversal in path",
        "no path sanitization",
    ],
    "CWE-287": [
        "if token == expected",
        "bypass authentication",
        "no session check",
        "is_admin = request.args.get",
    ],
    "CWE-79": [
        "innerHTML = user_input",
        "document.write(data)",
        "no html escaping",
        "render_template_string(user_data)",
    ],
    "CWE-502": [
        "pickle.loads(user_data)",
        "yaml.load(data)",
        "marshal.loads(",
        "jsonpickle.decode(",
    ],
}


class NVDConnector(ACPConnector):
    """Connector for the NIST National Vulnerability Database CVE API 2.0.

    Pulls CVE records, filters for code-relevant vulnerability classes, and
    normalises each CVE to a :class:`~core.ThreatIndicator`.

    The connector respects NVD's rate limits:
    - Without an API key: 5 requests / 30 s → sleep 6 s between requests.
    - With an API key:    50 requests / 30 s → sleep 0.6 s between requests.

    Args:
        api_key: Optional NVD API key.  Register free at
            https://nvd.nist.gov/developers/request-an-api-key.
        keyword_filters: List of keyword strings to search.  Defaults to
            :data:`DEFAULT_KEYWORD_FILTERS`.
        severity_filter: Restrict results to a single severity tier
            ("CRITICAL", "HIGH", "MEDIUM", "LOW").
        results_per_page: Number of CVEs per API page (max 2000).
        days_back: How many days back to search.  Default is 30.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        keyword_filters: list[str] | None = None,
        severity_filter: str | None = None,
        results_per_page: int = 20,
        days_back: int = 30,
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key
        self._keyword_filters: list[str] = (
            keyword_filters if keyword_filters is not None else DEFAULT_KEYWORD_FILTERS
        )
        self._severity_filter = (
            severity_filter.upper() if severity_filter else None
        )
        self._results_per_page = min(results_per_page, 2000)
        self._days_back = days_back
        self._timeout = timeout
        # Rate-limit: 6 s per request without key, 0.6 s with key.
        self._sleep_between_requests: float = 6.0 if not api_key else 0.6
        self._session = requests.Session()
        if api_key:
            self._session.headers.update({"apiKey": api_key})
        self._session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # ACPConnector interface
    # ------------------------------------------------------------------

    def metadata(self) -> dict:
        """Return connector metadata.

        Returns:
            dict: Connector metadata including name, format, update interval,
                threat categories, and version.
        """
        return {
            "name": "NVD",
            "format": "NVD-JSON-5.0",
            "update_interval": 3600,
            "threat_categories": [
                "SQLi", "SSRF", "IDOR", "path_traversal",
                "auth_bypass", "buffer_overflow", "XSS", "deserialization",
            ],
            "version": "1.0.0",
            "base_url": NVD_API_BASE,
            "requires_api_key": False,
        }

    def pull(self) -> list[ThreatIndicator]:
        """Pull code-relevant CVEs from NVD for all configured keyword filters.

        Makes one API request per keyword, respecting rate limits.
        Deduplicates by CVE ID across keywords.

        Returns:
            list[ThreatIndicator]: Normalised CVEs, sorted by CVSS score
                descending.
        """
        seen: dict[str, ThreatIndicator] = {}
        for keyword in self._keyword_filters:
            try:
                raw = self._fetch_cves(keyword)
                vulnerabilities = raw.get("vulnerabilities", [])
                logger.info(
                    "NVD keyword=%r → %d raw CVEs", keyword, len(vulnerabilities)
                )
                for item in vulnerabilities:
                    indicator = self._parse_cve(item)
                    if indicator and indicator.id not in seen:
                        seen[indicator.id] = indicator
                # Respect rate limit between keyword requests.
                time.sleep(self._sleep_between_requests)
            except requests.RequestException as exc:
                logger.warning("NVD fetch failed for keyword=%r: %s", keyword, exc)
        results = sorted(seen.values(), key=lambda i: i.cvss_score, reverse=True)
        logger.info("NVD pull complete — %d unique indicators", len(results))
        return results

    def stream(self) -> Generator[ThreatIndicator, None, None]:
        """Poll NVD on a schedule and yield new CVEs as they appear.

        Runs an infinite loop checking for CVEs published in the last hour.
        The generator sleeps between polls using the configured
        ``update_interval`` (1 hour).

        Yields:
            ThreatIndicator: Newly published CVE indicators.
        """
        seen_ids: set[str] = set()
        while True:
            try:
                indicators = self.pull()
                for indicator in indicators:
                    if indicator.id not in seen_ids:
                        seen_ids.add(indicator.id)
                        yield indicator
            except Exception as exc:  # noqa: BLE001
                logger.warning("NVD stream error: %s", exc)
            time.sleep(3600)  # Poll every hour.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_cves(self, keyword: str, start_index: int = 0) -> dict:
        """Execute a single GET request to the NVD CVE API.

        The NVD CVE API 2.0 supports date-range filters via ``pubStartDate``
        and ``pubEndDate``, but these parameters have a maximum window of 120
        days and may return 404 for certain date combinations.  When date
        filtering fails, the connector automatically retries without the date
        range parameters.

        Args:
            keyword: Keyword to search in CVE descriptions/titles.
            start_index: Pagination offset.

        Returns:
            dict: Parsed JSON response from NVD.

        Raises:
            requests.HTTPError: On 4xx/5xx responses after retry.
            requests.Timeout: If the server does not respond within timeout.
        """
        end_date = datetime.now(tz=timezone.utc)
        start_date = end_date - timedelta(days=min(self._days_back, 120))

        params: dict[str, str | int] = {
            "keywordSearch": keyword,
            "resultsPerPage": self._results_per_page,
            "startIndex": start_index,
        }
        # Add date range only when days_back is set to avoid unbounded queries.
        # NVD requires date range format: YYYY-MM-DDTHH:MM:SS.000 (no timezone offset).
        if self._days_back > 0:
            params["pubStartDate"] = start_date.strftime("%Y-%m-%dT%H:%M:%S.000")
            params["pubEndDate"] = end_date.strftime("%Y-%m-%dT%H:%M:%S.000")
        if self._severity_filter:
            params["cvssV3Severity"] = self._severity_filter

        logger.debug("NVD GET %s params=%s", NVD_API_BASE, params)
        response = self._session.get(
            NVD_API_BASE, params=params, timeout=self._timeout
        )
        # NVD returns 404 for certain invalid date combinations.
        # Retry without date filter as a fallback.
        if response.status_code == 404 and "pubStartDate" in params:
            logger.debug(
                "NVD returned 404 with date filter — retrying without date range."
            )
            params_no_date = {
                k: v for k, v in params.items()
                if k not in ("pubStartDate", "pubEndDate")
            }
            response = self._session.get(
                NVD_API_BASE, params=params_no_date, timeout=self._timeout
            )
        response.raise_for_status()
        return response.json()

    def _parse_cve(self, cve_item: dict) -> ThreatIndicator | None:
        """Parse a single CVE item from NVD JSON 5.0 format.

        Extracts CVE ID, description, CVSS score, CWE identifiers, and
        affected packages.  If parsing fails for any reason the error is
        logged and None is returned (caller skips the item).

        Args:
            cve_item: A single element from the ``vulnerabilities`` array
                in the NVD API response.

        Returns:
            ThreatIndicator | None: Normalised indicator, or None on error.
        """
        try:
            cve = cve_item.get("cve", {})
            cve_id: str = cve.get("id", "")
            if not cve_id:
                return None

            # ---- Description ----
            descriptions = cve.get("descriptions", [])
            description = next(
                (d["value"] for d in descriptions if d.get("lang") == "en"),
                "",
            )

            # ---- CVSS score ----
            cvss_score = 0.0
            metrics = cve.get("metrics", {})
            # Prefer CVSS v3.1, then v3.0, then v2.0
            for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                metric_list = metrics.get(key, [])
                if metric_list:
                    cvss_data = metric_list[0].get("cvssData", {})
                    cvss_score = float(cvss_data.get("baseScore", 0.0))
                    break

            severity = ThreatIndicator.severity_from_cvss(cvss_score)

            # ---- CWE ----
            weaknesses = cve.get("weaknesses", [])
            cwe_ids: list[str] = []
            for weakness in weaknesses:
                for desc in weakness.get("description", []):
                    val: str = desc.get("value", "")
                    if val.startswith("CWE-") and val not in cwe_ids:
                        cwe_ids.append(val)
            # Fallback: infer CWE from description text.
            if not cwe_ids:
                cwe_ids = self._infer_cwe_from_description(description)

            # ---- Affected packages / configurations ----
            affected_packages: list[str] = []
            configurations = cve.get("configurations", [])
            for config in configurations:
                for node in config.get("nodes", []):
                    for cpe_match in node.get("cpeMatch", []):
                        cpe: str = cpe_match.get("criteria", "")
                        if cpe:
                            affected_packages.append(cpe)
            # Cap to 20 entries to keep indicator compact.
            affected_packages = affected_packages[:20]

            # ---- Affected code patterns ----
            affected_patterns = self._build_affected_patterns(cwe_ids, description)

            # ---- Semantic rule ----
            semantic_rule = self._build_semantic_rule(cve_id, cwe_ids, description)

            # ---- Published timestamp ----
            pub_date_str: str = cve.get("published", "")
            try:
                timestamp = datetime.fromisoformat(
                    pub_date_str.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                timestamp = datetime.now(tz=timezone.utc)

            return ThreatIndicator(
                id=cve_id,
                source="NVD",
                severity=severity,
                cwe=cwe_ids,
                affected_patterns=affected_patterns,
                stix_pattern=None,
                activation_signature={
                    "vuln_class": cwe_ids[0] if cwe_ids else "unknown",
                    "mean_activation": [],
                    "variance": [],
                    "layer": -1,
                    "source": "NVD",
                },
                semantic_rule=semantic_rule,
                formal_property=self._build_formal_property(cwe_ids),
                timestamp=timestamp,
                ttl=604800,  # 7 days
                description=description[:512],
                cvss_score=cvss_score,
                affected_packages=affected_packages,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to parse CVE item (id=%s): %s",
                cve_item.get("cve", {}).get("id", "unknown"),
                exc,
                exc_info=True,
            )
            return None

    def _infer_cwe_from_description(self, description: str) -> list[str]:
        """Infer likely CWE identifiers from free-text CVE description.

        Uses keyword matching against a curated map.  Multiple CWEs may be
        returned if the description mentions several vulnerability types.

        Args:
            description: English-language CVE description text.

        Returns:
            list[str]: Inferred CWE identifiers (may be empty).
        """
        if not description:
            return []
        lower = description.lower()
        seen: set[str] = set()
        results: list[str] = []
        for keyword, cwe in _KEYWORD_CWE_MAP:
            if keyword in lower and cwe not in seen:
                seen.add(cwe)
                results.append(cwe)
        return results

    def _build_affected_patterns(
        self, cwe_ids: list[str], description: str
    ) -> list[str]:
        """Build a list of code-level patterns for a given set of CWEs.

        Args:
            cwe_ids: List of CWE identifiers.
            description: CVE description (used as fallback pattern).

        Returns:
            list[str]: Code pattern strings for Layer 2 RAG indexing.
        """
        patterns: list[str] = []
        for cwe in cwe_ids:
            patterns.extend(_CWE_PATTERN_MAP.get(cwe, []))
        if not patterns and description:
            # Use first 120 characters of description as a generic pattern.
            patterns.append(description[:120])
        return list(dict.fromkeys(patterns))  # Deduplicate while preserving order.

    def _build_semantic_rule(
        self, cve_id: str, cwe_ids: list[str], description: str
    ) -> str:
        """Compose a natural-language semantic rule for RAG indexing.

        Args:
            cve_id: The CVE identifier.
            cwe_ids: Inferred or explicit CWE identifiers.
            description: CVE description.

        Returns:
            str: Semantic rule string.
        """
        cwe_str = ", ".join(cwe_ids) if cwe_ids else "unknown CWE"
        snippet = description[:200] if description else "no description"
        return f"{cve_id}: {snippet} [{cwe_str}]"

    def _build_formal_property(self, cwe_ids: list[str]) -> str | None:
        """Select a Nagini/Viper property template for the primary CWE.

        Args:
            cwe_ids: List of CWE identifiers.

        Returns:
            str | None: Formal property string, or None if no template found.
        """
        _cwe_to_property: dict[str, str] = {
            "CWE-89": "Requires(is_parameterized(query))",
            "CWE-639": "Requires(ownership_verified(user, resource_id))",
            "CWE-918": "Requires(is_allowlisted(url))",
            "CWE-22": "Requires(is_sandboxed(path))",
            "CWE-287": "Requires(is_authenticated(session))",
            "CWE-79": "Requires(is_html_escaped(output))",
            "CWE-502": "Requires(is_trusted_source(data))",
        }
        for cwe in cwe_ids:
            prop = _cwe_to_property.get(cwe)
            if prop:
                return prop
        return None

    def query_by_cve_id(self, cve_id: str) -> ThreatIndicator | None:
        """Fetch a specific CVE by its ID.

        Args:
            cve_id: CVE identifier, e.g. "CVE-2024-12345".

        Returns:
            ThreatIndicator | None: Indicator for the CVE, or None if not
                found or on error.
        """
        try:
            params = {"cveId": cve_id}
            response = self._session.get(
                NVD_API_BASE, params=params, timeout=self._timeout
            )
            response.raise_for_status()
            data = response.json()
            vulnerabilities = data.get("vulnerabilities", [])
            if vulnerabilities:
                return self._parse_cve(vulnerabilities[0])
        except requests.RequestException as exc:
            logger.warning("NVD query by CVE ID failed (%s): %s", cve_id, exc)
        return None
