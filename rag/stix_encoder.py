"""
STIX Encoder — translates STIX 2.1 threat indicators to semantic anti-patterns.

Converts the machine-readable STIX 2.1 pattern language into natural-language
security rules suitable for ChromaDB embedding and RAG retrieval.  Also
generates Nagini/Viper formal property strings for Layer 3.

Research contribution:
    This is (to our knowledge) the first automated STIX-to-natural-language
    encoder specifically designed for *code security* contexts.  Existing
    STIX tooling targets network indicators (IPs, domains, hashes).  Our
    encoder handles code-relevant STIX patterns and maps them to both
    semantic rules (for RAG) and formal properties (for verification).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# CWE identifier pattern.
_CWE_RE = re.compile(r"CWE-\d+")

# STIX kill-chain phase → CWE mapping.
_KILL_CHAIN_CWE_MAP: dict[str, list[str]] = {
    "initial-access": ["CWE-287", "CWE-89"],
    "execution": ["CWE-94", "CWE-78"],
    "persistence": ["CWE-502", "CWE-287"],
    "privilege-escalation": ["CWE-269", "CWE-284"],
    "defense-evasion": ["CWE-284", "CWE-693"],
    "credential-access": ["CWE-255", "CWE-798"],
    "discovery": ["CWE-200", "CWE-639"],
    "lateral-movement": ["CWE-287", "CWE-918"],
    "collection": ["CWE-200", "CWE-22"],
    "exfiltration": ["CWE-200", "CWE-918"],
    "impact": ["CWE-400", "CWE-502"],
}

# STIX pattern keyword → CWE + natural-language template.
_PATTERN_RULES: list[tuple[str, str, str]] = [
    # (keyword, cwe, rule_template)
    ("sql", "CWE-89", "Code constructs a SQL query using string operations — SQL injection risk"),
    ("execute", "CWE-89", "Direct execution of potentially user-influenced query — injection risk"),
    ("httpx.get", "CWE-918", "Function performs HTTP GET using potentially user-supplied URL — SSRF risk"),
    ("requests.get", "CWE-918", "Function performs HTTP GET using potentially user-supplied URL — SSRF risk"),
    ("urlopen", "CWE-918", "urllib URL open with user-supplied input — SSRF risk (CWE-918)"),
    ("path.join", "CWE-22", "File path constructed using path.join without canonicalisation — path traversal risk"),
    ("os.path", "CWE-22", "File system access with user-influenced path — path traversal risk"),
    ("pickle.loads", "CWE-502", "Deserialisation of potentially untrusted pickle data — RCE risk"),
    ("yaml.load", "CWE-502", "YAML load without SafeLoader — unsafe deserialisation risk"),
    ("subprocess", "CWE-78", "Subprocess execution with potentially user-controlled argument — command injection"),
    ("eval(", "CWE-94", "Dynamic code evaluation with user-controlled input — code injection risk"),
    ("exec(", "CWE-94", "Dynamic code execution with user-controlled input — code injection risk"),
    ("innerHTML", "CWE-79", "Direct DOM innerHTML assignment from user data — XSS risk"),
    ("document.write", "CWE-79", "document.write with user-controlled data — XSS risk"),
    ("is_admin", "CWE-287", "Admin privilege check from user-controlled source — auth bypass risk"),
    ("request.args", "CWE-284", "Security decision based on client-supplied query parameter"),
    ("etree.parse", "CWE-611", "XML parsing with potentially external entity references — XXE risk"),
    ("open(", "CWE-22", "File open operation with potentially user-influenced path"),
]

# Formal property templates per CWE (Nagini/Viper syntax).
_CWE_FORMAL_TEMPLATES: dict[str, dict[str, str]] = {
    "CWE-89": {
        "precondition": "Requires(is_parameterized(query))",
        "postcondition": "Ensures(no_sql_injection(result))",
        "invariant": "Invariant(all_queries_parameterized)",
    },
    "CWE-639": {
        "precondition": "Requires(ownership_verified(user_id, resource_id))",
        "postcondition": "Ensures(authorized_access(result, user_id))",
        "invariant": "Invariant(resource_ownership_enforced)",
    },
    "CWE-918": {
        "precondition": "Requires(is_allowlisted(url) and is_not_private_ip(url))",
        "postcondition": "Ensures(external_service_only(response))",
        "invariant": "Invariant(no_ssrf_possible)",
    },
    "CWE-22": {
        "precondition": "Requires(is_sandboxed(path, BASE_DIR))",
        "postcondition": "Ensures(within_sandbox(result_path, BASE_DIR))",
        "invariant": "Invariant(path_traversal_impossible)",
    },
    "CWE-287": {
        "precondition": "Requires(is_authenticated(session) and session_valid(session))",
        "postcondition": "Ensures(privileged_operation_authorized(result))",
        "invariant": "Invariant(authentication_enforced)",
    },
    "CWE-94": {
        "precondition": "Requires(is_trusted_source(code_input))",
        "postcondition": "Ensures(no_code_injection(result))",
        "invariant": "Invariant(eval_exec_restricted)",
    },
    "CWE-502": {
        "precondition": "Requires(is_trusted_source(serialized_data))",
        "postcondition": "Ensures(safe_deserialization(result))",
        "invariant": "Invariant(untrusted_deserialization_impossible)",
    },
    "CWE-79": {
        "precondition": "Requires(is_html_escaped(user_content))",
        "postcondition": "Ensures(no_xss_in_output(rendered))",
        "invariant": "Invariant(output_always_escaped)",
    },
    "CWE-78": {
        "precondition": "Requires(is_allowlisted(command) and no_shell_metacharacters(args))",
        "postcondition": "Ensures(command_injection_impossible(result))",
        "invariant": "Invariant(subprocess_restricted)",
    },
    "CWE-611": {
        "precondition": "Requires(external_entities_disabled(xml_parser))",
        "postcondition": "Ensures(no_xxe_in_parsed(result))",
        "invariant": "Invariant(xml_parser_hardened)",
    },
}


class STIXEncoder:
    """Converts STIX 2.1 threat indicators to semantic rules and formal properties.

    The encoder handles two conversion paths:
    1. STIX indicator → natural-language security rule (for Layer 2 RAG).
    2. STIX indicator → Nagini/Viper formal property (for Layer 3 verification).

    Both paths start from the STIX pattern string and the indicator's kill-chain
    and label metadata.  The encoder uses rule-based pattern matching rather
    than an LLM to ensure deterministic, reproducible output without API calls.

    Research note:
        Future work will replace the rule-based encoder with a fine-tuned
        model trained on (STIX pattern, natural-language rule) pairs, which
        should improve coverage for novel STIX patterns.
    """

    def __init__(self) -> None:
        self._rule_cache: dict[str, str] = {}

    def encode_indicator(self, stix_indicator: dict) -> str:
        """Convert a STIX 2.1 indicator object to a natural-language security rule.

        The output is intended for embedding in ChromaDB.  It combines:
        - The indicator name/description.
        - A rule inferred from the STIX pattern string.
        - CWE identifiers extracted from kill-chain phases and labels.

        Args:
            stix_indicator: STIX 2.1 Indicator object dict.

        Returns:
            str: Natural-language security rule for RAG indexing.

        Example::

            stix = {
                "type": "indicator",
                "name": "SSRF via user URL",
                "pattern": "[file:content MATCHES '.*httpx\\.get\\(.*request\\..*\\).*']",
                "labels": ["ssrf", "CWE-918"],
            }
            encoder.encode_indicator(stix)
            # → "Function performs HTTP GET using potentially user-supplied URL
            #    — SSRF risk (CWE-918) | Labels: ssrf, CWE-918"
        """
        if not isinstance(stix_indicator, dict):
            raise TypeError("stix_indicator must be a dict.")
        stix_id = stix_indicator.get("id", "unknown")
        # Check cache.
        if stix_id in self._rule_cache:
            return self._rule_cache[stix_id]

        name: str = stix_indicator.get("name", "")
        description: str = stix_indicator.get("description", "")
        pattern: str = stix_indicator.get("pattern", "")
        labels: list[str] = stix_indicator.get("labels", [])
        kill_chain: list[dict] = stix_indicator.get("kill_chain_phases", [])

        # Extract CWEs.
        cwe_ids = self.extract_cwe(stix_indicator)
        cwe_str = ", ".join(cwe_ids) if cwe_ids else "unknown CWE"

        # Infer rule from pattern.
        rule = self._infer_rule_from_pattern(pattern)
        if not rule:
            rule = name or description[:200] or f"STIX indicator {stix_id}"

        # Add CWE annotation.
        rule = f"{rule} ({cwe_str})"

        # Append labels for additional retrieval context.
        if labels:
            labels_str = ", ".join(labels[:5])
            rule = f"{rule} | Labels: {labels_str}"

        # Append kill-chain phases.
        if kill_chain:
            phases = [p.get("phase_name", "") for p in kill_chain if p.get("phase_name")]
            if phases:
                rule = f"{rule} | Kill-chain: {', '.join(phases)}"

        self._rule_cache[stix_id] = rule
        return rule

    def extract_cwe(self, stix_indicator: dict) -> list[str]:
        """Extract CWE identifiers from a STIX 2.1 indicator.

        Checks:
        1. ``labels`` field for "CWE-" prefixed strings.
        2. Kill-chain phase names mapped to CWEs via :data:`_KILL_CHAIN_CWE_MAP`.
        3. Pattern string content via keyword rules.

        Args:
            stix_indicator: STIX 2.1 Indicator object dict.

        Returns:
            list[str]: Deduplicated CWE identifiers.
        """
        cwe_ids: list[str] = []
        seen: set[str] = set()

        # From labels.
        for label in stix_indicator.get("labels", []):
            if label.startswith("CWE-") and label not in seen:
                cwe_ids.append(label)
                seen.add(label)

        # From kill-chain phases.
        for phase in stix_indicator.get("kill_chain_phases", []):
            phase_name = phase.get("phase_name", "").lower()
            for cwe in _KILL_CHAIN_CWE_MAP.get(phase_name, []):
                if cwe not in seen:
                    cwe_ids.append(cwe)
                    seen.add(cwe)

        # From pattern content.
        pattern = stix_indicator.get("pattern", "").lower()
        for keyword, cwe, _ in _PATTERN_RULES:
            if keyword in pattern and cwe not in seen:
                cwe_ids.append(cwe)
                seen.add(cwe)

        # From description/name via regex.
        for text in (
            stix_indicator.get("description", ""),
            stix_indicator.get("name", ""),
        ):
            for match in _CWE_RE.finditer(text):
                cwe = match.group()
                if cwe not in seen:
                    cwe_ids.append(cwe)
                    seen.add(cwe)

        return cwe_ids

    def stix_to_formal_property(
        self, stix_indicator: dict, vuln_class: str | None = None
    ) -> str:
        """Convert a STIX indicator to a Nagini/Viper formal property.

        Selects the appropriate property template based on the extracted CWE
        identifiers or the provided ``vuln_class``.

        Args:
            stix_indicator: STIX 2.1 Indicator object dict.
            vuln_class: Optional vulnerability class override.

        Returns:
            str: Nagini precondition string.

        Example::

            stix = {"pattern": "[...httpx.get...]", "labels": ["CWE-918"]}
            encoder.stix_to_formal_property(stix)
            # → "Requires(is_allowlisted(url) and is_not_private_ip(url))"
        """
        cwe_ids = self.extract_cwe(stix_indicator)
        # Map vuln_class hint to a CWE if provided.
        if vuln_class:
            vuln_cwe_map = {
                "IDOR": "CWE-639",
                "SQLi": "CWE-89",
                "SSRF": "CWE-918",
                "path_traversal": "CWE-22",
                "auth_bypass": "CWE-287",
                "XSS": "CWE-79",
                "command_injection": "CWE-78",
                "deserialization": "CWE-502",
            }
            hint_cwe = vuln_cwe_map.get(vuln_class)
            if hint_cwe and hint_cwe not in cwe_ids:
                cwe_ids.insert(0, hint_cwe)

        for cwe in cwe_ids:
            template = _CWE_FORMAL_TEMPLATES.get(cwe)
            if template:
                return template["precondition"]

        # Fallback: generic property.
        return "Requires(input_is_validated(user_input))"

    def encode_batch(self, stix_indicators: list[dict]) -> list[str]:
        """Encode a batch of STIX indicators to natural-language rules.

        Args:
            stix_indicators: List of STIX 2.1 Indicator dicts.

        Returns:
            list[str]: One rule string per indicator.
        """
        return [self.encode_indicator(ind) for ind in stix_indicators]

    def get_formal_templates_for_cwe(self, cwe: str) -> dict | None:
        """Return the full formal property template for a CWE.

        Args:
            cwe: CWE identifier (e.g. "CWE-89").

        Returns:
            dict | None: Template dict with precondition, postcondition,
                and invariant keys, or None if not found.
        """
        return _CWE_FORMAL_TEMPLATES.get(cwe)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_rule_from_pattern(self, pattern: str) -> Optional[str]:
        """Infer a natural-language rule from a STIX pattern string.

        Args:
            pattern: STIX 2.1 pattern string.

        Returns:
            str | None: Inferred rule, or None if no rule matches.
        """
        if not pattern:
            return None
        pattern_lower = pattern.lower()
        for keyword, _cwe, rule_template in _PATTERN_RULES:
            if keyword in pattern_lower:
                return rule_template
        return None
