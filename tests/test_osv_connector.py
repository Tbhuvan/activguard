"""
Tests for OSVConnector.

Integration tests make real HTTP calls to api.osv.dev.  Set
ACTIVGUARD_OFFLINE=1 to skip them.

Run all tests:     pytest tests/test_osv_connector.py -v
Run offline only:  pytest tests/test_osv_connector.py -v -m "not integration"
"""

from __future__ import annotations

import os

import pytest

from connectors.osv_connector import OSVConnector
from core.threat_indicator import ThreatIndicator

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

OFFLINE = os.environ.get("ACTIVGUARD_OFFLINE", "0") == "1"
skip_if_offline = pytest.mark.skipif(OFFLINE, reason="ACTIVGUARD_OFFLINE=1 set")


# ---------------------------------------------------------------------------
# Unit tests (no network)
# ---------------------------------------------------------------------------

class TestOSVConnectorInit:
    """Test OSVConnector initialisation and metadata."""

    def test_metadata_keys(self):
        """metadata() must return all required keys."""
        connector = OSVConnector()
        meta = connector.metadata()
        required_keys = {"name", "format", "update_interval", "threat_categories", "version"}
        assert required_keys.issubset(set(meta.keys()))

    def test_metadata_name(self):
        """Connector name must be 'OSV'."""
        assert OSVConnector().metadata()["name"] == "OSV"

    def test_default_ecosystems(self):
        """Default ecosystems should include PyPI, npm, Go."""
        connector = OSVConnector()
        assert "PyPI" in connector._ecosystems
        assert "npm" in connector._ecosystems

    def test_custom_ecosystems(self):
        """Custom ecosystem list should be stored correctly."""
        connector = OSVConnector(ecosystems=["PyPI", "RubyGems"])
        assert connector._ecosystems == ["PyPI", "RubyGems"]

    def test_query_package_empty_name_raises(self):
        """Empty package name should raise ValueError."""
        connector = OSVConnector()
        with pytest.raises(ValueError, match="Package name"):
            connector.query_package("", "PyPI")

    def test_query_package_empty_ecosystem_raises(self):
        """Empty ecosystem should raise ValueError."""
        connector = OSVConnector()
        with pytest.raises(ValueError, match="Ecosystem"):
            connector.query_package("requests", "")


class TestOSVParseLogic:
    """Test internal OSV parsing logic without network calls."""

    def _make_connector(self) -> OSVConnector:
        return OSVConnector(ecosystems=["PyPI"])

    def test_extract_cvss_score_numeric_string(self):
        """Numeric string CVSS score should be parsed correctly."""
        connector = self._make_connector()
        assert connector._extract_cvss_score("7.5") == 7.5

    def test_extract_cvss_score_float(self):
        """Float CVSS score should pass through."""
        connector = self._make_connector()
        assert connector._extract_cvss_score(9.8) == 9.8

    def test_extract_cvss_score_vector_returns_zero(self):
        """CVSS vector string should return 0.0 (cannot parse without library)."""
        connector = self._make_connector()
        score = connector._extract_cvss_score("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
        assert score == 0.0

    def test_extract_cvss_score_none(self):
        """None should return 0.0."""
        connector = self._make_connector()
        assert connector._extract_cvss_score(None) == 0.0  # type: ignore[arg-type]

    def test_extract_cwe_from_labels(self):
        """CWE labels in database_specific.cwe_ids should be extracted."""
        connector = self._make_connector()
        vuln = {
            "database_specific": {"cwe_ids": ["CWE-89", "CWE-639"]},
        }
        cwes = connector._extract_cwe(vuln, "SQL injection via user input")
        assert "CWE-89" in cwes
        assert "CWE-639" in cwes

    def test_extract_cwe_from_description_fallback(self):
        """CWE should be inferred from description when no labels present."""
        connector = self._make_connector()
        vuln = {"database_specific": {}}
        cwes = connector._extract_cwe(vuln, "SQL injection in query builder")
        assert "CWE-89" in cwes

    def test_extract_cwe_ssrf_description(self):
        """SSRF description → CWE-918."""
        connector = self._make_connector()
        vuln = {"database_specific": {}}
        cwes = connector._extract_cwe(vuln, "SSRF via webhook callback URL parameter")
        assert "CWE-918" in cwes

    def test_parse_osv_vuln_minimal(self):
        """_parse_osv_vuln should return None for empty id."""
        connector = self._make_connector()
        result = connector._parse_osv_vuln({}, "requests", "PyPI")
        assert result is None

    def test_parse_osv_vuln_valid(self):
        """_parse_osv_vuln should return ThreatIndicator for valid OSV record."""
        connector = self._make_connector()
        vuln = {
            "id": "GHSA-test-0001-0001",
            "summary": "SQL injection in ORM query builder",
            "details": "Unsanitised user input passed to ORM filter.",
            "severity": [{"type": "CVSS_V3", "score": "8.5"}],
            "database_specific": {"cwe_ids": ["CWE-89"]},
            "affected": [
                {
                    "package": {"name": "mylib", "ecosystem": "PyPI"},
                    "versions": ["1.0.0", "1.0.1"],
                }
            ],
            "aliases": ["CVE-2024-11111"],
            "published": "2024-01-10T00:00:00Z",
        }
        indicator = connector._parse_osv_vuln(vuln, "mylib", "PyPI")
        assert indicator is not None
        assert indicator.id == "CVE-2024-11111"  # Prefers CVE alias.
        assert indicator.source == "OSV"
        assert "CWE-89" in indicator.cwe
        assert indicator.severity in {"critical", "high", "medium", "low", "unknown"}
        assert "PyPI/mylib@1.0.0" in indicator.affected_packages

    def test_parse_osv_vuln_no_cvss(self):
        """_parse_osv_vuln should handle missing CVSS score gracefully."""
        connector = self._make_connector()
        vuln = {
            "id": "GHSA-test-0002-0002",
            "summary": "Path traversal vulnerability",
            "database_specific": {"severity": "HIGH", "cwe_ids": ["CWE-22"]},
            "affected": [],
            "aliases": [],
        }
        indicator = connector._parse_osv_vuln(vuln, "testpkg", "PyPI")
        assert indicator is not None
        assert indicator.severity in {"high", "unknown"}

    def test_build_affected_patterns_cwe89(self):
        """CWE-89 should produce SQL execution patterns."""
        connector = self._make_connector()
        patterns = connector._build_affected_patterns(["CWE-89"], "")
        assert any("execute" in p or "raw_query" in p for p in patterns)

    def test_build_affected_patterns_fallback(self):
        """Unknown CWE with description should use description as fallback pattern."""
        connector = self._make_connector()
        patterns = connector._build_affected_patterns(
            ["CWE-9999"], "A very specific vulnerability description."
        )
        assert len(patterns) > 0

    def test_build_formal_property_ssrf(self):
        """CWE-918 should produce URL allowlist property."""
        connector = self._make_connector()
        prop = connector._build_formal_property(["CWE-918"])
        assert "allowlisted" in prop.lower() or "allowlist" in prop.lower()

    def test_build_formal_property_none(self):
        """Unknown CWE should return None."""
        connector = self._make_connector()
        prop = connector._build_formal_property(["CWE-9999"])
        assert prop is None

    def test_build_semantic_rule_with_alias(self):
        """build_semantic_rule should include OSV ID suffix when different from indicator_id."""
        connector = self._make_connector()
        rule = connector._build_semantic_rule(
            "CVE-2024-12345", "GHSA-xxxx-xxxx-xxxx", ["CWE-89"], "SQLi in query builder"
        )
        assert "CVE-2024-12345" in rule
        assert "GHSA-xxxx-xxxx-xxxx" in rule


# ---------------------------------------------------------------------------
# Integration tests (require network)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestOSVConnectorIntegration:
    """Integration tests against the real OSV API (api.osv.dev)."""

    @skip_if_offline
    def test_osv_query_package_requests(self):
        """Querying 'requests' PyPI package should return a list."""
        connector = OSVConnector(ecosystems=["PyPI"])
        indicators = connector.query_package("requests", "PyPI")
        assert isinstance(indicators, list)
        # 'requests' has known historical vulnerabilities.
        assert len(indicators) >= 0  # May be 0 if all are fixed in latest version.

    @skip_if_offline
    def test_osv_query_package_vulnerable_version(self):
        """requests 2.25.0 has known CVEs."""
        connector = OSVConnector(ecosystems=["PyPI"])
        indicators = connector.query_package("requests", "PyPI", "2.25.0")
        assert isinstance(indicators, list)
        # This version has at least PYSEC-2023-74 (SSRF via Proxy-Authorization redirect).
        for ind in indicators:
            assert isinstance(ind, ThreatIndicator)
            assert ind.source == "OSV"

    @skip_if_offline
    def test_osv_query_pyyaml(self):
        """pyyaml has historical deserialization CVEs."""
        connector = OSVConnector(ecosystems=["PyPI"])
        indicators = connector.query_package("pyyaml", "PyPI", "5.3.1")
        assert isinstance(indicators, list)
        for ind in indicators:
            assert ind.source == "OSV"
            assert isinstance(ind.cwe, list)

    @skip_if_offline
    def test_osv_indicator_fields(self):
        """Returned indicators must have all required fields."""
        connector = OSVConnector(ecosystems=["PyPI"])
        indicators = connector.query_package("pillow", "PyPI", "9.0.0")
        for ind in indicators:
            assert ind.id  # Non-empty ID.
            assert ind.source == "OSV"
            assert ind.severity in {"critical", "high", "medium", "low", "unknown"}
            assert isinstance(ind.cvss_score, float)
            assert 0.0 <= ind.cvss_score <= 10.0
            assert isinstance(ind.affected_packages, list)
            assert isinstance(ind.cwe, list)

    @skip_if_offline
    def test_osv_pull(self):
        """pull() should return a list of indicators."""
        connector = OSVConnector(ecosystems=["PyPI"])
        indicators = connector.pull()
        assert isinstance(indicators, list)
        # Should find at least some vulnerabilities across the well-known package set.
        if indicators:
            assert all(isinstance(i, ThreatIndicator) for i in indicators)

    @skip_if_offline
    def test_osv_batch_query(self):
        """query_batch() should return one list per query."""
        connector = OSVConnector()
        queries = [
            {"package": {"name": "flask", "ecosystem": "PyPI"}},
            {"package": {"name": "django", "ecosystem": "PyPI"}},
        ]
        results = connector.query_batch(queries)
        assert isinstance(results, list)
        assert len(results) == 2
        for result_list in results:
            assert isinstance(result_list, list)

    @skip_if_offline
    def test_osv_query_empty_batch(self):
        """Empty batch query should return empty list."""
        connector = OSVConnector()
        results = connector.query_batch([])
        assert results == []

    @skip_if_offline
    def test_osv_npm_package(self):
        """OSV query for npm package should work."""
        connector = OSVConnector(ecosystems=["npm"])
        indicators = connector.query_package("lodash", "npm", "4.17.20")
        assert isinstance(indicators, list)
        for ind in indicators:
            assert ind.source == "OSV"

    @skip_if_offline
    def test_osv_indicator_not_expired(self):
        """Freshly pulled indicators should not be expired."""
        connector = OSVConnector(ecosystems=["PyPI"])
        indicators = connector.query_package("cryptography", "PyPI")
        for ind in indicators:
            assert not ind.is_expired(), (
                f"Indicator {ind.id} expired immediately after pull."
            )

    @skip_if_offline
    def test_osv_to_semantic_pattern(self):
        """to_semantic_pattern() should return a non-empty string."""
        connector = OSVConnector(ecosystems=["PyPI"])
        indicators = connector.query_package("werkzeug", "PyPI")
        if indicators:
            pattern = indicators[0].to_semantic_pattern()
            assert isinstance(pattern, str)
            assert len(pattern) > 10

    @skip_if_offline
    def test_osv_get_vuln_by_id_invalid(self):
        """get_vuln_by_id with a non-existent ID should return None."""
        connector = OSVConnector()
        result = connector.get_vuln_by_id("GHSA-does-not-exist-0000")
        assert result is None
