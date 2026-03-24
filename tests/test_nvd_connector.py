"""
Tests for NVDConnector.

These tests make real HTTP calls to the NVD API.  They are integration tests
and require network access.  To run them in isolation without a network,
set the environment variable ACTIVGUARD_OFFLINE=1, which will skip tests
marked with ``@pytest.mark.integration``.

Run all tests: pytest tests/test_nvd_connector.py -v
Run offline only: pytest tests/test_nvd_connector.py -v -m "not integration"
"""

from __future__ import annotations

import os
import time

import pytest

from connectors.nvd_connector import NVDConnector
from core.threat_indicator import ThreatIndicator

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

OFFLINE = os.environ.get("ACTIVGUARD_OFFLINE", "0") == "1"
skip_if_offline = pytest.mark.skipif(OFFLINE, reason="ACTIVGUARD_OFFLINE=1 set")


# ---------------------------------------------------------------------------
# Unit tests (no network)
# ---------------------------------------------------------------------------

class TestNVDConnectorInit:
    """Test NVDConnector initialisation and metadata."""

    def test_metadata_keys(self):
        """metadata() must return all required keys."""
        connector = NVDConnector()
        meta = connector.metadata()
        required_keys = {"name", "format", "update_interval", "threat_categories", "version"}
        assert required_keys.issubset(set(meta.keys()))

    def test_metadata_name(self):
        """Connector name must be 'NVD'."""
        connector = NVDConnector()
        assert connector.metadata()["name"] == "NVD"

    def test_metadata_format(self):
        """Format must be NVD JSON 5.0."""
        connector = NVDConnector()
        assert connector.metadata()["format"] == "NVD-JSON-5.0"

    def test_init_with_api_key(self):
        """Initialisation with API key must not raise."""
        connector = NVDConnector(api_key="test-key-12345")
        assert connector._api_key == "test-key-12345"

    def test_init_custom_keyword_filters(self):
        """Custom keyword filters must be stored correctly."""
        filters = ["XSS", "deserialization"]
        connector = NVDConnector(keyword_filters=filters)
        assert connector._keyword_filters == filters

    def test_init_severity_filter_uppercase(self):
        """Severity filter must be normalised to uppercase."""
        connector = NVDConnector(severity_filter="high")
        assert connector._severity_filter == "HIGH"

    def test_init_results_per_page_capped(self):
        """results_per_page must be capped at 2000."""
        connector = NVDConnector(results_per_page=99999)
        assert connector._results_per_page == 2000

    def test_rate_limit_without_key(self):
        """Without API key, sleep between requests should be 6.0s."""
        connector = NVDConnector()
        assert connector._sleep_between_requests == 6.0

    def test_rate_limit_with_key(self):
        """With API key, sleep between requests should be 0.6s."""
        connector = NVDConnector(api_key="fake-key")
        assert connector._sleep_between_requests == 0.6


class TestNVDParseLogic:
    """Test internal parsing logic without network calls."""

    def test_infer_cwe_sql_injection(self):
        """'sql injection' in description → CWE-89."""
        connector = NVDConnector()
        cwes = connector._infer_cwe_from_description(
            "A SQL injection vulnerability allows remote code execution."
        )
        assert "CWE-89" in cwes

    def test_infer_cwe_ssrf(self):
        """'SSRF' in description → CWE-918."""
        connector = NVDConnector()
        cwes = connector._infer_cwe_from_description(
            "Server-side request forgery (SSRF) in the webhook endpoint."
        )
        assert "CWE-918" in cwes

    def test_infer_cwe_path_traversal(self):
        """'path traversal' in description → CWE-22."""
        connector = NVDConnector()
        cwes = connector._infer_cwe_from_description(
            "Path traversal in file download allows reading arbitrary files."
        )
        assert "CWE-22" in cwes

    def test_infer_cwe_idor(self):
        """'IDOR' in description → CWE-639."""
        connector = NVDConnector()
        cwes = connector._infer_cwe_from_description(
            "Insecure direct object reference (IDOR) allows unauthorized access."
        )
        assert "CWE-639" in cwes

    def test_infer_cwe_auth_bypass(self):
        """'authentication bypass' → CWE-287."""
        connector = NVDConnector()
        cwes = connector._infer_cwe_from_description(
            "Authentication bypass allows unauthenticated users to access admin panel."
        )
        assert "CWE-287" in cwes

    def test_infer_cwe_empty_description(self):
        """Empty description → empty CWE list."""
        connector = NVDConnector()
        cwes = connector._infer_cwe_from_description("")
        assert cwes == []

    def test_infer_cwe_none_description(self):
        """None description → empty CWE list."""
        connector = NVDConnector()
        cwes = connector._infer_cwe_from_description(None)  # type: ignore[arg-type]
        assert cwes == []

    def test_severity_from_cvss_critical(self):
        """CVSS ≥ 9.0 → critical."""
        assert ThreatIndicator.severity_from_cvss(9.8) == "critical"
        assert ThreatIndicator.severity_from_cvss(9.0) == "critical"

    def test_severity_from_cvss_high(self):
        """7.0 ≤ CVSS < 9.0 → high."""
        assert ThreatIndicator.severity_from_cvss(7.5) == "high"
        assert ThreatIndicator.severity_from_cvss(7.0) == "high"

    def test_severity_from_cvss_medium(self):
        """4.0 ≤ CVSS < 7.0 → medium."""
        assert ThreatIndicator.severity_from_cvss(5.5) == "medium"
        assert ThreatIndicator.severity_from_cvss(4.0) == "medium"

    def test_severity_from_cvss_low(self):
        """0 < CVSS < 4.0 → low."""
        assert ThreatIndicator.severity_from_cvss(2.0) == "low"

    def test_severity_from_cvss_zero(self):
        """CVSS == 0.0 → unknown."""
        assert ThreatIndicator.severity_from_cvss(0.0) == "unknown"

    def test_severity_from_cvss_invalid(self):
        """CVSS out of [0,10] → ValueError."""
        with pytest.raises(ValueError):
            ThreatIndicator.severity_from_cvss(11.0)

    def test_parse_cve_minimal(self):
        """_parse_cve should return None for item missing 'cve.id'."""
        connector = NVDConnector()
        result = connector._parse_cve({"cve": {}})
        assert result is None

    def test_parse_cve_valid_item(self):
        """_parse_cve should return a ThreatIndicator for valid NVD JSON."""
        connector = NVDConnector()
        item = {
            "cve": {
                "id": "CVE-2024-99999",
                "published": "2024-01-15T12:00:00.000",
                "descriptions": [
                    {"lang": "en", "value": "SQL injection vulnerability in MyApp."}
                ],
                "metrics": {
                    "cvssMetricV31": [
                        {"cvssData": {"baseScore": 9.8}}
                    ]
                },
                "weaknesses": [
                    {"description": [{"lang": "en", "value": "CWE-89"}]}
                ],
            }
        }
        indicator = connector._parse_cve(item)
        assert indicator is not None
        assert indicator.id == "CVE-2024-99999"
        assert indicator.cvss_score == 9.8
        assert indicator.severity == "critical"
        assert "CWE-89" in indicator.cwe
        assert indicator.source == "NVD"

    def test_build_formal_property_sqli(self):
        """SQLi CWE should produce parameterised query property."""
        connector = NVDConnector()
        prop = connector._build_formal_property(["CWE-89"])
        assert prop == "Requires(is_parameterized(query))"

    def test_build_formal_property_unknown(self):
        """Unknown CWE should return None."""
        connector = NVDConnector()
        prop = connector._build_formal_property(["CWE-9999"])
        assert prop is None

    def test_build_affected_patterns_sqli(self):
        """CWE-89 should produce SQL-related patterns."""
        connector = NVDConnector()
        patterns = connector._build_affected_patterns(["CWE-89"], "")
        assert any("execute" in p.lower() or "query" in p.lower() for p in patterns)


# ---------------------------------------------------------------------------
# Integration tests (require network)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestNVDConnectorIntegration:
    """Integration tests that call the real NVD API."""

    @skip_if_offline
    def test_nvd_pull_returns_indicators(self):
        """pull() with 'injection' keyword should return ThreatIndicator objects."""
        connector = NVDConnector(
            keyword_filters=["injection"],
            results_per_page=5,
            days_back=365,
        )
        indicators = connector.pull()
        assert len(indicators) > 0, (
            "NVD returned 0 indicators for 'injection'.  "
            "Check network connectivity and NVD API status."
        )
        assert all(isinstance(i, ThreatIndicator) for i in indicators)

    @skip_if_offline
    def test_nvd_indicator_fields(self):
        """Returned indicators must have valid CVE IDs, severity, and CVSS."""
        connector = NVDConnector(
            keyword_filters=["SQL injection"],
            results_per_page=5,
            days_back=365,
        )
        indicators = connector.pull()
        if indicators:
            i = indicators[0]
            assert i.id.startswith("CVE-"), f"Expected CVE-xxx, got {i.id}"
            assert i.severity in {"critical", "high", "medium", "low", "unknown"}
            assert isinstance(i.cvss_score, float)
            assert 0.0 <= i.cvss_score <= 10.0
            assert i.source == "NVD"
            assert isinstance(i.cwe, list)
            assert isinstance(i.affected_patterns, list)
            assert isinstance(i.timestamp.year, int)

    @skip_if_offline
    def test_nvd_rate_limiting(self):
        """Three consecutive pulls should not raise rate-limit errors."""
        connector = NVDConnector(
            keyword_filters=["buffer overflow"],
            results_per_page=3,
            days_back=180,
        )
        for i in range(3):
            result = connector.pull()
            assert isinstance(result, list), f"pull() call {i+1} returned non-list"
            # Respect rate limit between test iterations.
            if i < 2:
                time.sleep(7)

    @skip_if_offline
    def test_nvd_idor_keyword(self):
        """IDOR keyword search should return results or empty list (not raise)."""
        connector = NVDConnector(
            keyword_filters=["IDOR"],
            results_per_page=5,
            days_back=730,
        )
        indicators = connector.pull()
        # IDOR may return 0 results in NVD; that is valid.
        assert isinstance(indicators, list)

    @skip_if_offline
    def test_nvd_high_severity_filter(self):
        """HIGH severity filter should return only high/critical CVEs."""
        connector = NVDConnector(
            keyword_filters=["authentication bypass"],
            severity_filter="HIGH",
            results_per_page=5,
            days_back=365,
        )
        indicators = connector.pull()
        for ind in indicators:
            assert ind.severity in {"high", "critical"}, (
                f"Expected high/critical, got {ind.severity} for {ind.id}"
            )

    @skip_if_offline
    def test_nvd_indicator_to_semantic_pattern(self):
        """to_semantic_pattern() should return a non-empty string."""
        connector = NVDConnector(
            keyword_filters=["path traversal"],
            results_per_page=3,
            days_back=365,
        )
        indicators = connector.pull()
        if indicators:
            pattern = indicators[0].to_semantic_pattern()
            assert isinstance(pattern, str)
            assert len(pattern) > 20

    @skip_if_offline
    def test_nvd_indicator_not_expired(self):
        """Freshly pulled indicators should not be expired."""
        connector = NVDConnector(
            keyword_filters=["injection"],
            results_per_page=3,
            days_back=30,
        )
        indicators = connector.pull()
        for ind in indicators:
            assert not ind.is_expired(), (
                f"Indicator {ind.id} is unexpectedly expired immediately after pull."
            )

    @skip_if_offline
    def test_nvd_multiple_keywords(self):
        """Pull with multiple keywords should return deduplicated results."""
        connector = NVDConnector(
            keyword_filters=["SQL injection", "SSRF"],
            results_per_page=5,
            days_back=365,
        )
        indicators = connector.pull()
        ids = [i.id for i in indicators]
        assert len(ids) == len(set(ids)), "Duplicate CVE IDs in pull result."
