"""
Tests for ThreatIndicator dataclass.

These tests are pure unit tests with no network calls.

Run: pytest tests/test_threat_indicator.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from core.threat_indicator import ThreatIndicator


def make_indicator(**overrides) -> ThreatIndicator:
    """Helper: construct a valid ThreatIndicator with sensible defaults."""
    defaults = dict(
        id="CVE-2024-12345",
        source="NVD",
        severity="high",
        cwe=["CWE-89"],
        affected_patterns=["execute(sql_string)", "f\"{query}\""],
        stix_pattern=None,
        activation_signature={"layer": 16, "mean_activation": [0.1, 0.2]},
        semantic_rule="SQL injection via f-string [CWE-89]",
        formal_property="Requires(is_parameterized(query))",
        timestamp=datetime.now(tz=timezone.utc),
        ttl=86400,
        description="A SQL injection vulnerability in MyApp.",
        cvss_score=8.5,
        affected_packages=["PyPI/myapp@1.0.0"],
    )
    defaults.update(overrides)
    return ThreatIndicator(**defaults)


class TestThreatIndicatorCreation:
    """Test ThreatIndicator construction and validation."""

    def test_basic_creation(self):
        """Valid indicator should be created without error."""
        ind = make_indicator()
        assert ind.id == "CVE-2024-12345"
        assert ind.source == "NVD"
        assert ind.severity == "high"

    def test_invalid_severity_raises(self):
        """Invalid severity should raise ValueError."""
        with pytest.raises(ValueError, match="severity"):
            make_indicator(severity="extreme")

    def test_empty_id_raises(self):
        """Empty ID should raise ValueError."""
        with pytest.raises(ValueError, match="id"):
            make_indicator(id="")

    def test_invalid_cvss_score_high(self):
        """CVSS > 10.0 should raise ValueError."""
        with pytest.raises(ValueError, match="cvss_score"):
            make_indicator(cvss_score=10.1)

    def test_invalid_cvss_score_negative(self):
        """CVSS < 0.0 should raise ValueError."""
        with pytest.raises(ValueError, match="cvss_score"):
            make_indicator(cvss_score=-0.1)

    def test_invalid_ttl_raises(self):
        """Negative TTL should raise ValueError."""
        with pytest.raises(ValueError, match="ttl"):
            make_indicator(ttl=-1)

    def test_non_numeric_cvss_raises(self):
        """Non-numeric CVSS score should raise TypeError."""
        with pytest.raises((TypeError, ValueError)):
            make_indicator(cvss_score="high")  # type: ignore[arg-type]

    def test_unknown_severity_valid(self):
        """'unknown' is a valid severity (used when CVSS is 0.0)."""
        ind = make_indicator(severity="unknown", cvss_score=0.0)
        assert ind.severity == "unknown"

    def test_all_valid_severities(self):
        """All expected severity labels should be accepted."""
        for sev in ("critical", "high", "medium", "low", "unknown"):
            ind = make_indicator(severity=sev)
            assert ind.severity == sev

    def test_zero_ttl_is_valid(self):
        """TTL of 0 is valid (means never expires)."""
        ind = make_indicator(ttl=0)
        assert ind.ttl == 0

    def test_default_description_empty(self):
        """description defaults to empty string."""
        ind = ThreatIndicator(
            id="TEST-001",
            source="internal",
            severity="low",
            cwe=[],
            affected_patterns=[],
            stix_pattern=None,
            activation_signature={},
            semantic_rule="test rule",
            formal_property=None,
            timestamp=datetime.now(tz=timezone.utc),
            ttl=3600,
        )
        assert ind.description == ""

    def test_default_cvss_zero(self):
        """cvss_score defaults to 0.0."""
        ind = ThreatIndicator(
            id="TEST-002",
            source="internal",
            severity="unknown",
            cwe=[],
            affected_patterns=[],
            stix_pattern=None,
            activation_signature={},
            semantic_rule="test rule",
            formal_property=None,
            timestamp=datetime.now(tz=timezone.utc),
            ttl=3600,
        )
        assert ind.cvss_score == 0.0

    def test_default_affected_packages_empty_list(self):
        """affected_packages should default to an empty list (not shared mutable)."""
        ind1 = ThreatIndicator(
            id="TEST-003",
            source="internal",
            severity="low",
            cwe=[],
            affected_patterns=[],
            stix_pattern=None,
            activation_signature={},
            semantic_rule="",
            formal_property=None,
            timestamp=datetime.now(tz=timezone.utc),
            ttl=3600,
        )
        ind2 = ThreatIndicator(
            id="TEST-004",
            source="internal",
            severity="low",
            cwe=[],
            affected_patterns=[],
            stix_pattern=None,
            activation_signature={},
            semantic_rule="",
            formal_property=None,
            timestamp=datetime.now(tz=timezone.utc),
            ttl=3600,
        )
        ind1.affected_packages.append("pkg")
        assert ind2.affected_packages == [], "Default affected_packages list is shared!"


class TestThreatIndicatorExpiry:
    """Test ThreatIndicator.is_expired() behaviour."""

    def test_fresh_indicator_not_expired(self):
        """Indicator created now with 1-day TTL should not be expired."""
        ind = make_indicator(ttl=86400)
        assert not ind.is_expired()

    def test_old_indicator_is_expired(self):
        """Indicator with timestamp 2 days ago and 1-day TTL should be expired."""
        old_ts = datetime.now(tz=timezone.utc) - timedelta(days=2)
        ind = make_indicator(timestamp=old_ts, ttl=86400)
        assert ind.is_expired()

    def test_zero_ttl_never_expires(self):
        """TTL of 0 means the indicator never expires."""
        old_ts = datetime.now(tz=timezone.utc) - timedelta(days=3650)
        ind = make_indicator(timestamp=old_ts, ttl=0)
        assert not ind.is_expired()

    def test_just_at_ttl_boundary(self):
        """Indicator at exactly TTL seconds old should be expired."""
        boundary_ts = datetime.now(tz=timezone.utc) - timedelta(seconds=86401)
        ind = make_indicator(timestamp=boundary_ts, ttl=86400)
        assert ind.is_expired()

    def test_naive_timestamp_treated_as_utc(self):
        """Naive timestamp (no tzinfo) should be treated as UTC."""
        old_ts = datetime.utcnow() - timedelta(days=2)
        assert old_ts.tzinfo is None
        ind = make_indicator(timestamp=old_ts, ttl=86400)
        assert ind.is_expired()


class TestSeverityFromCVSS:
    """Test ThreatIndicator.severity_from_cvss() classmethod."""

    def test_critical(self):
        assert ThreatIndicator.severity_from_cvss(10.0) == "critical"
        assert ThreatIndicator.severity_from_cvss(9.0) == "critical"

    def test_high(self):
        assert ThreatIndicator.severity_from_cvss(8.9) == "high"
        assert ThreatIndicator.severity_from_cvss(7.0) == "high"

    def test_medium(self):
        assert ThreatIndicator.severity_from_cvss(6.9) == "medium"
        assert ThreatIndicator.severity_from_cvss(4.0) == "medium"

    def test_low(self):
        assert ThreatIndicator.severity_from_cvss(3.9) == "low"
        assert ThreatIndicator.severity_from_cvss(0.1) == "low"

    def test_unknown(self):
        assert ThreatIndicator.severity_from_cvss(0.0) == "unknown"

    def test_boundary_9(self):
        assert ThreatIndicator.severity_from_cvss(9.0) == "critical"

    def test_boundary_7(self):
        assert ThreatIndicator.severity_from_cvss(7.0) == "high"

    def test_boundary_4(self):
        assert ThreatIndicator.severity_from_cvss(4.0) == "medium"

    @pytest.mark.parametrize("invalid", [-0.1, 10.1, 100.0])
    def test_out_of_range_raises(self, invalid):
        with pytest.raises(ValueError):
            ThreatIndicator.severity_from_cvss(invalid)


class TestToSemanticPattern:
    """Test ThreatIndicator.to_semantic_pattern()."""

    def test_contains_id(self):
        """Semantic pattern must contain the indicator ID."""
        ind = make_indicator()
        pattern = ind.to_semantic_pattern()
        assert "CVE-2024-12345" in pattern

    def test_contains_severity(self):
        """Semantic pattern must contain severity."""
        ind = make_indicator()
        pattern = ind.to_semantic_pattern()
        assert "high" in pattern

    def test_contains_cwe(self):
        """Semantic pattern must contain CWE."""
        ind = make_indicator(cwe=["CWE-89", "CWE-639"])
        pattern = ind.to_semantic_pattern()
        assert "CWE-89" in pattern
        assert "CWE-639" in pattern

    def test_contains_source(self):
        """Semantic pattern must contain source."""
        ind = make_indicator(source="OSV")
        pattern = ind.to_semantic_pattern()
        assert "OSV" in pattern

    def test_empty_cwe_handled(self):
        """Empty CWE list should produce valid pattern without crash."""
        ind = make_indicator(cwe=[])
        pattern = ind.to_semantic_pattern()
        assert isinstance(pattern, str)
        assert "CWE-unknown" in pattern

    def test_empty_patterns_handled(self):
        """Empty affected_patterns should produce valid pattern."""
        ind = make_indicator(affected_patterns=[])
        pattern = ind.to_semantic_pattern()
        assert "N/A" in pattern


class TestSerialisation:
    """Test ThreatIndicator.to_dict() and from_dict()."""

    def test_round_trip(self):
        """to_dict() + from_dict() should produce an equivalent indicator."""
        original = make_indicator()
        data = original.to_dict()
        restored = ThreatIndicator.from_dict(data)
        assert restored.id == original.id
        assert restored.source == original.source
        assert restored.severity == original.severity
        assert restored.cvss_score == original.cvss_score
        assert restored.cwe == original.cwe
        assert restored.affected_packages == original.affected_packages

    def test_to_dict_timestamp_is_string(self):
        """to_dict() should serialise timestamp as ISO 8601 string."""
        ind = make_indicator()
        data = ind.to_dict()
        assert isinstance(data["timestamp"], str)
        # Must be parseable.
        datetime.fromisoformat(data["timestamp"])

    def test_from_dict_missing_optional_fields(self):
        """from_dict() should handle missing optional fields with defaults."""
        data = {
            "id": "CVE-2024-XXXX",
            "source": "NVD",
            "severity": "medium",
            "cwe": [],
            "affected_patterns": [],
            "stix_pattern": None,
            "activation_signature": {},
            "semantic_rule": "test",
            "formal_property": None,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "ttl": 3600,
        }
        ind = ThreatIndicator.from_dict(data)
        assert ind.description == ""
        assert ind.cvss_score == 0.0
        assert ind.affected_packages == []

    def test_to_dict_contains_all_fields(self):
        """to_dict() output must contain all expected keys."""
        ind = make_indicator()
        data = ind.to_dict()
        expected_keys = {
            "id", "source", "severity", "cwe", "affected_patterns",
            "stix_pattern", "activation_signature", "semantic_rule",
            "formal_property", "timestamp", "ttl", "description",
            "cvss_score", "affected_packages",
        }
        assert expected_keys.issubset(set(data.keys()))

    def test_repr_is_informative(self):
        """__repr__ should include id, source, severity, and cvss."""
        ind = make_indicator()
        r = repr(ind)
        assert "CVE-2024-12345" in r
        assert "NVD" in r
        assert "high" in r
        assert "8.5" in r


class TestThreatIndicatorEdgeCases:
    """Edge cases and boundary conditions."""

    def test_multiple_cwes(self):
        """Multiple CWEs should all be stored."""
        cwes = ["CWE-89", "CWE-639", "CWE-918"]
        ind = make_indicator(cwe=cwes)
        assert ind.cwe == cwes

    def test_stix_pattern_optional(self):
        """stix_pattern can be None."""
        ind = make_indicator(stix_pattern=None)
        assert ind.stix_pattern is None

    def test_stix_pattern_string(self):
        """stix_pattern can be a STIX 2.1 pattern string."""
        pattern = "[file:content MATCHES '.*pickle.loads.*']"
        ind = make_indicator(stix_pattern=pattern)
        assert ind.stix_pattern == pattern

    def test_activation_signature_is_dict(self):
        """activation_signature must be stored as-is."""
        sig = {"layer": 16, "mean_activation": [0.1, 0.2], "variance": [0.01]}
        ind = make_indicator(activation_signature=sig)
        assert ind.activation_signature == sig

    def test_formal_property_optional(self):
        """formal_property can be None."""
        ind = make_indicator(formal_property=None)
        assert ind.formal_property is None

    def test_long_description_stored(self):
        """Long descriptions should be stored fully."""
        long_desc = "A" * 1000
        ind = make_indicator(description=long_desc)
        assert len(ind.description) == 1000

    def test_empty_affected_packages(self):
        """Empty affected_packages list is valid."""
        ind = make_indicator(affected_packages=[])
        assert ind.affected_packages == []

    def test_many_affected_packages(self):
        """Large affected_packages list should be stored."""
        pkgs = [f"pkg_{i}" for i in range(100)]
        ind = make_indicator(affected_packages=pkgs)
        assert len(ind.affected_packages) == 100
