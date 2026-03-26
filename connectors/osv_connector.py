"""
OSV Connector — pulls vulnerabilities from the Open Source Vulnerability database.

Uses the OSV API v1 (https://api.osv.dev/v1/).  OSV aggregates vulnerabilities
from GitHub Security Advisories, PyPI, Go, npm, RubyGems, Maven, crates.io,
and many other ecosystems into a single unified schema.

Research context:
    OSV covers supply-chain vulnerabilities in open-source packages — the
    dominant attack surface for LLM-generated code that references PyPI
    or npm packages.  The OSVConnector provides the package-level threat
    intelligence that complements NVD's CVE-level data.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Generator

import requests

from core.acp import ACPConnector
from core.threat_indicator import ThreatIndicator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OSV_QUERY_URL = "https://api.osv.dev/v1/query"
OSV_QUERYBATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{id}"

# Well-known packages with historical vulnerabilities — used for initial pull.
_WELL_KNOWN_PACKAGES: dict[str, list[tuple[str, str]]] = {
    "PyPI": [
        ("requests", None),
        ("flask", None),
        ("django", None),
        ("pyyaml", None),
        ("pillow", None),
        ("cryptography", None),
        ("urllib3", None),
        ("paramiko", None),
        ("aiohttp", None),
        ("werkzeug", None),
    ],
    "npm": [
        ("express", None),
        ("lodash", None),
        ("axios", None),
        ("moment", None),
        ("minimist", None),
    ],
    "Go": [
        ("github.com/gin-gonic/gin", None),
        ("github.com/gorilla/mux", None),
    ],
}

# OSV severity → ActivGuard severity mapping.
_OSV_SEVERITY_MAP: dict[str, str] = {
    "CRITICAL": "critical",
    "HIGH": "high",
    "MEDIUM": "medium",
    "LOW": "low",
    "NONE": "low",
}

# CWE inference from OSV summary/details keywords.
_KEYWORD_CWE_MAP: list[tuple[str, str]] = [
    ("sql injection", "CWE-89"),
    ("sqli", "CWE-89"),
    ("remote code execution", "CWE-94"),
    ("code execution", "CWE-94"),
    ("path traversal", "CWE-22"),
    ("directory traversal", "CWE-22"),
    ("ssrf", "CWE-918"),
    ("server-side request forgery", "CWE-918"),
    ("idor", "CWE-639"),
    ("authentication bypass", "CWE-287"),
    ("authorization bypass", "CWE-285"),
    ("privilege escalation", "CWE-269"),
    ("xss", "CWE-79"),
    ("cross-site scripting", "CWE-79"),
    ("buffer overflow", "CWE-120"),
    ("use-after-free", "CWE-416"),
    ("null pointer", "CWE-476"),
    ("deserialization", "CWE-502"),
    ("command injection", "CWE-78"),
    ("xxe", "CWE-611"),
    ("xml external entity", "CWE-611"),
    ("open redirect", "CWE-601"),
    ("csrf", "CWE-352"),
    ("hardcoded", "CWE-798"),
    ("denial of service", "CWE-400"),
    ("dos", "CWE-400"),
    ("race condition", "CWE-362"),
    ("integer overflow", "CWE-190"),
    ("format string", "CWE-134"),
]


class OSVConnector(ACPConnector):
    """Connector for the Open Source Vulnerability (OSV) database.

    Supports three query modes:
    1. ``pull()``:          Queries a curated set of well-known packages across
                            configured ecosystems and returns their known vulns.
    2. ``query_package()``: Query a specific package (with optional version)
                            for known vulnerabilities.
    3. ``stream()``:        Polls OSV periodically and yields new indicators.

    Args:
        ecosystems: List of ecosystem identifiers to include.  Defaults to
            ["PyPI", "npm", "Go"].
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        ecosystems: list[str] | None = None,
        timeout: int = 30,
    ) -> None:
        self._ecosystems: list[str] = (
            ecosystems if ecosystems is not None else ["PyPI", "npm", "Go"]
        )
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
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
            "name": "OSV",
            "format": "OSV-Schema-1.0",
            "update_interval": 3600,
            "threat_categories": [
                "supply_chain", "dependency_vuln", "SQLi", "XSS",
                "deserialization", "RCE", "auth_bypass",
            ],
            "version": "1.0.0",
            "base_url": OSV_QUERY_URL,
            "requires_api_key": False,
        }

    def pull(self) -> list[ThreatIndicator]:
        """Pull vulnerabilities for well-known packages in configured ecosystems.

        Queries each package individually (rate: ~10 requests / second is
        safe without API key) and deduplicates by OSV ID.

        Returns:
            list[ThreatIndicator]: Normalised vulnerability indicators.
        """
        seen: dict[str, ThreatIndicator] = {}
        for ecosystem in self._ecosystems:
            packages = _WELL_KNOWN_PACKAGES.get(ecosystem, [])
            for pkg_name, pkg_version in packages:
                try:
                    indicators = self.query_package(pkg_name, ecosystem, pkg_version)
                    for ind in indicators:
                        if ind.id not in seen:
                            seen[ind.id] = ind
                    time.sleep(0.1)  # Polite delay.
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "OSV pull failed for %s/%s: %s", ecosystem, pkg_name, exc
                    )
        results = sorted(seen.values(), key=lambda i: i.cvss_score, reverse=True)
        logger.info("OSV pull complete — %d unique indicators", len(results))
        return results

    def stream(self) -> Generator[ThreatIndicator, None, None]:
        """Poll OSV periodically and yield newly discovered indicators.

        Yields:
            ThreatIndicator: Newly discovered vulnerability indicators.
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
                logger.warning("OSV stream error: %s", exc)
            time.sleep(3600)  # Poll every hour.

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def query_package(
        self,
        name: str,
        ecosystem: str,
        version: str | None = None,
    ) -> list[ThreatIndicator]:
        """Query OSV for known vulnerabilities affecting a specific package.

        Args:
            name: Package name (e.g. "requests", "lodash").
            ecosystem: Ecosystem identifier (e.g. "PyPI", "npm", "Go").
            version: Specific version string.  If None, returns all known
                vulnerabilities for the package regardless of version.

        Returns:
            list[ThreatIndicator]: Vulnerabilities affecting this package.

        Raises:
            requests.RequestException: On network / HTTP errors.
        """
        if not name:
            raise ValueError("Package name must be a non-empty string.")
        if not ecosystem:
            raise ValueError("Ecosystem must be a non-empty string.")

        body: dict = {"package": {"name": name, "ecosystem": ecosystem}}
        if version:
            body["version"] = version

        logger.debug("OSV query: %s/%s@%s", ecosystem, name, version or "any")
        response = self._session.post(
            OSV_QUERY_URL, json=body, timeout=self._timeout
        )
        response.raise_for_status()
        data = response.json()

        vulns = data.get("vulns", [])
        indicators: list[ThreatIndicator] = []
        for vuln in vulns:
            indicator = self._parse_osv_vuln(vuln, name, ecosystem)
            if indicator:
                indicators.append(indicator)
        return indicators

    def query_batch(
        self, queries: list[dict]
    ) -> list[list[ThreatIndicator]]:
        """Submit multiple package queries in a single batch request.

        Args:
            queries: List of query dicts, each with ``package``
                (containing ``name`` and ``ecosystem``) and optional
                ``version``.

        Returns:
            list[list[ThreatIndicator]]: One inner list per query, in the
                same order as the input.

        Raises:
            requests.RequestException: On network / HTTP errors.
        """
        if not queries:
            return []
        body = {"queries": queries}
        response = self._session.post(
            OSV_QUERYBATCH_URL, json=body, timeout=self._timeout
        )
        response.raise_for_status()
        data = response.json()
        results: list[list[ThreatIndicator]] = []
        for result_item, query in zip(data.get("results", []), queries):
            pkg_name = query.get("package", {}).get("name", "unknown")
            ecosystem = query.get("package", {}).get("ecosystem", "unknown")
            indicators: list[ThreatIndicator] = []
            for vuln in result_item.get("vulns", []):
                indicator = self._parse_osv_vuln(vuln, pkg_name, ecosystem)
                if indicator:
                    indicators.append(indicator)
            results.append(indicators)
        return results

    def get_vuln_by_id(self, osv_id: str) -> ThreatIndicator | None:
        """Fetch full details for a specific OSV vulnerability.

        Args:
            osv_id: OSV vulnerability ID (e.g. "GHSA-xxxx-xxxx-xxxx",
                "PYSEC-2023-000").

        Returns:
            ThreatIndicator | None: Full indicator, or None on failure.
        """
        if not osv_id:
            raise ValueError("osv_id must be a non-empty string.")
        url = OSV_VULN_URL.format(id=osv_id)
        try:
            response = self._session.get(url, timeout=self._timeout)
            response.raise_for_status()
            return self._parse_osv_vuln(response.json(), "unknown", "unknown")
        except requests.RequestException as exc:
            logger.warning("OSV get vuln by id failed (%s): %s", osv_id, exc)
            return None

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_osv_vuln(
        self, vuln: dict, pkg_name: str, ecosystem: str
    ) -> ThreatIndicator | None:
        """Parse a single OSV vulnerability record to a ThreatIndicator.

        Args:
            vuln: Raw OSV vulnerability dict.
            pkg_name: Package name (used as fallback in affected_packages).
            ecosystem: Package ecosystem.

        Returns:
            ThreatIndicator | None: Normalised indicator, or None on error.
        """
        try:
            osv_id: str = vuln.get("id", "")
            if not osv_id:
                return None

            summary: str = vuln.get("summary", "")
            details: str = vuln.get("details", "")
            description = (summary or details)[:512]

            # ---- Severity / CVSS ----
            cvss_score = 0.0
            osv_severity_label = "unknown"
            severity_list = vuln.get("severity", [])
            for sev in severity_list:
                sev_type = sev.get("type", "")
                sev_score = sev.get("score", "")
                if sev_type in ("CVSS_V3", "CVSS_V2") and sev_score:
                    # OSV severity score is raw CVSS vector or numeric string.
                    cvss_score = self._extract_cvss_score(sev_score)
                    osv_severity_label = sev_type
                    break
            # Also check database-specific severity fields.
            if cvss_score == 0.0:
                db_specific = vuln.get("database_specific", {})
                sev_str = db_specific.get("severity", "")
                if sev_str.upper() in _OSV_SEVERITY_MAP:
                    osv_severity_label = sev_str.upper()
            severity = (
                ThreatIndicator.severity_from_cvss(cvss_score)
                if cvss_score > 0.0
                else _OSV_SEVERITY_MAP.get(osv_severity_label, "unknown")
            )

            # ---- CWE ----
            cwe_ids = self._extract_cwe(vuln, description)

            # ---- Affected packages ----
            affected_packages: list[str] = []
            for affected in vuln.get("affected", []):
                a_pkg = affected.get("package", {})
                a_name = a_pkg.get("name", pkg_name)
                a_eco = a_pkg.get("ecosystem", ecosystem)
                versions = affected.get("versions", [])
                if versions:
                    for v in versions[:5]:
                        affected_packages.append(f"{a_eco}/{a_name}@{v}")
                else:
                    affected_packages.append(f"{a_eco}/{a_name}")

            # ---- Aliases (CVE IDs) ----
            aliases: list[str] = vuln.get("aliases", [])
            cve_alias = next(
                (a for a in aliases if a.startswith("CVE-")), None
            )
            # Prefer CVE ID as the indicator ID if available.
            indicator_id = cve_alias or osv_id

            # ---- Fetch timestamp (use now, not OSV publish date, so TTL is from ingestion) ----
            timestamp = datetime.now(tz=timezone.utc)

            # ---- Code patterns ----
            affected_patterns = self._build_affected_patterns(cwe_ids, description)
            semantic_rule = self._build_semantic_rule(
                indicator_id, osv_id, cwe_ids, description
            )

            return ThreatIndicator(
                id=indicator_id,
                source="OSV",
                severity=severity,
                cwe=cwe_ids,
                affected_patterns=affected_patterns,
                stix_pattern=None,
                activation_signature={
                    "vuln_class": cwe_ids[0] if cwe_ids else "unknown",
                    "mean_activation": [],
                    "variance": [],
                    "layer": -1,
                    "source": "OSV",
                    "osv_id": osv_id,
                },
                semantic_rule=semantic_rule,
                formal_property=self._build_formal_property(cwe_ids),
                timestamp=timestamp,
                ttl=604800,  # 7 days
                description=description,
                cvss_score=cvss_score,
                affected_packages=affected_packages[:20],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to parse OSV vuln (id=%s): %s",
                vuln.get("id", "unknown"),
                exc,
                exc_info=True,
            )
            return None

    def _extract_cvss_score(self, score_value: str | float) -> float:
        """Extract a numeric CVSS score from the OSV severity field.

        OSV may store a full CVSS vector string or a plain numeric string.

        Args:
            score_value: Raw score value from OSV severity array.

        Returns:
            float: Numeric CVSS score, or 0.0 if extraction fails.
        """
        if isinstance(score_value, (int, float)):
            return float(score_value)
        # Try plain numeric string.
        try:
            return float(score_value)
        except (ValueError, TypeError):
            pass
        # Try to extract base score from CVSS vector, e.g.
        # "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H" — numeric score
        # is not embedded in the vector itself; we cannot parse without
        # a CVSS library.  Return 0.0 to trigger keyword-based severity.
        return 0.0

    def _extract_cwe(self, vuln: dict, description: str) -> list[str]:
        """Extract CWE identifiers from an OSV vulnerability record.

        Checks ``database_specific.cwe_ids`` and ``database_specific.cwes``
        fields, then falls back to keyword inference from the description.

        Args:
            vuln: Raw OSV vulnerability dict.
            description: Combined summary/details text.

        Returns:
            list[str]: CWE identifiers.
        """
        cwe_ids: list[str] = []
        db_specific = vuln.get("database_specific", {})
        # GitHub Advisory format.
        for field_name in ("cwe_ids", "cwes"):
            raw = db_specific.get(field_name, [])
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, str) and item.startswith("CWE-"):
                        if item not in cwe_ids:
                            cwe_ids.append(item)
                    elif isinstance(item, dict):
                        cwe_val = item.get("cweId", item.get("id", ""))
                        if cwe_val.startswith("CWE-") and cwe_val not in cwe_ids:
                            cwe_ids.append(cwe_val)
        if not cwe_ids:
            # Keyword-based inference from description.
            lower = description.lower()
            seen: set[str] = set()
            for keyword, cwe in _KEYWORD_CWE_MAP:
                if keyword in lower and cwe not in seen:
                    seen.add(cwe)
                    cwe_ids.append(cwe)
        return cwe_ids

    def _build_affected_patterns(
        self, cwe_ids: list[str], description: str
    ) -> list[str]:
        """Build code-level patterns for RAG indexing from CWEs.

        Args:
            cwe_ids: CWE identifiers.
            description: Vulnerability description.

        Returns:
            list[str]: Code pattern strings.
        """
        _cwe_patterns: dict[str, list[str]] = {
            "CWE-89": ["f\"{query}\"", "execute(sql_string)", "raw_query"],
            "CWE-94": ["exec(user_input)", "eval(data)", "compile(user_code)"],
            "CWE-22": ["os.path.join(base, user_input)", "open(user_path)"],
            "CWE-918": ["requests.get(url)", "urllib.request.urlopen(user_url)"],
            "CWE-639": ["Model.get(id=user_id)", "no ownership check"],
            "CWE-502": ["pickle.loads(data)", "yaml.load(stream)"],
            "CWE-79": ["innerHTML = user_input", "render_template_string(user)"],
            "CWE-287": ["no session check", "bypass authentication"],
            "CWE-78": ["subprocess.call(user_input, shell=True)"],
            "CWE-611": ["etree.parse(user_xml)", "lxml with external entities"],
        }
        patterns: list[str] = []
        for cwe in cwe_ids:
            patterns.extend(_cwe_patterns.get(cwe, []))
        if not patterns and description:
            patterns.append(description[:120])
        return list(dict.fromkeys(patterns))

    def _build_semantic_rule(
        self,
        indicator_id: str,
        osv_id: str,
        cwe_ids: list[str],
        description: str,
    ) -> str:
        """Compose semantic rule string for RAG indexing.

        Args:
            indicator_id: Primary indicator identifier.
            osv_id: Original OSV identifier (may differ from indicator_id).
            cwe_ids: CWE identifiers.
            description: Vulnerability description.

        Returns:
            str: Semantic rule string.
        """
        cwe_str = ", ".join(cwe_ids) if cwe_ids else "unknown CWE"
        suffix = f" (OSV: {osv_id})" if osv_id != indicator_id else ""
        snippet = description[:200] if description else "no description"
        return f"{indicator_id}{suffix}: {snippet} [{cwe_str}]"

    def _build_formal_property(self, cwe_ids: list[str]) -> str | None:
        """Select a formal property template for the primary CWE.

        Args:
            cwe_ids: CWE identifiers.

        Returns:
            str | None: Nagini precondition string, or None.
        """
        _cwe_to_property: dict[str, str] = {
            "CWE-89": "Requires(is_parameterized(query))",
            "CWE-639": "Requires(ownership_verified(user, resource_id))",
            "CWE-918": "Requires(is_allowlisted(url))",
            "CWE-22": "Requires(is_sandboxed(path))",
            "CWE-287": "Requires(is_authenticated(session))",
            "CWE-94": "Requires(is_trusted_source(code))",
            "CWE-502": "Requires(is_trusted_source(data))",
            "CWE-79": "Requires(is_html_escaped(output))",
            "CWE-78": "Requires(is_allowlisted(command))",
        }
        for cwe in cwe_ids:
            prop = _cwe_to_property.get(cwe)
            if prop:
                return prop
        return None
