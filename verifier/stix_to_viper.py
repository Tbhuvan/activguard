"""
STIX-to-Viper Generator — auto-generates Nagini/Viper verification specs
from STIX 2.1 threat indicators.

This module implements what we believe to be the first automated pipeline
from STIX 2.1 threat intelligence to formal verification specifications.
The pipeline has three stages:
1. Extract CWE identifiers from the STIX indicator.
2. Map each CWE to a formal property template.
3. Instantiate the template with indicator-specific values.

Research contribution:
    The ability to auto-generate formal specs from live threat intel means
    that when a new CVE is published and ingested via the NVD or TAXII
    connector, Layer 3's verification coverage is automatically extended —
    without any manual property engineering.  This is the "zero-day" update
    path for formal verification.

    The key research claim is that a meaningful fraction of code-relevant
    CVEs (those associated with IDOR, SQLi, SSRF, path traversal, auth bypass)
    have formal properties that can be auto-generated from their CWE
    classification.  Our evaluation quantifies this fraction.
"""

from __future__ import annotations

import logging
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.threat_indicator import ThreatIndicator

logger = logging.getLogger(__name__)

# CWE → formal precondition.
_CWE_PRECONDITION: dict[str, str] = {
    "CWE-89":  "  requires query.is_parameterized",
    "CWE-639": "  requires ownership_check(caller, resource_id)",
    "CWE-918": "  requires allowlist_check(url) && !is_private_range(resolve(url))",
    "CWE-22":  "  requires realpath(path).startswith(BASE_DIR)",
    "CWE-287": "  requires session.is_authenticated && !session.is_expired",
    "CWE-284": "  requires permission_check(caller, required_permission)",
    "CWE-94":  "  requires is_trusted_source(code_input)",
    "CWE-502": "  requires is_trusted_source(data) || has_valid_signature(data, KEY)",
    "CWE-79":  "  requires html_escaped(user_content)",
    "CWE-78":  "  requires is_allowlisted(command) && !has_shell_metacharacters(args)",
    "CWE-611": "  requires external_entities_disabled(xml_parser)",
    "CWE-352": "  requires csrf_token_valid(request)",
    "CWE-798": "  requires !is_hardcoded_credential(credential)",
    "CWE-400": "  requires rate_limited(request) || resource_bounded(input_size)",
    "CWE-269": "  requires principle_of_least_privilege(operation, caller)",
    "CWE-601": "  requires is_relative_redirect(url) || is_allowlisted(url)",
}

# CWE → formal postcondition.
_CWE_POSTCONDITION: dict[str, str] = {
    "CWE-89":  "  ensures !result.contains_injection_artifact",
    "CWE-639": "  ensures result.owner == caller || explicit_permission(caller, result)",
    "CWE-918": "  ensures response.source_domain in ALLOWED_DOMAINS",
    "CWE-22":  "  ensures realpath(result_path).startswith(BASE_DIR)",
    "CWE-287": "  ensures result.is_authorized",
    "CWE-284": "  ensures result.is_authorized",
    "CWE-94":  "  ensures !result.contains_injected_code",
    "CWE-502": "  ensures result.is_safe_object",
    "CWE-79":  "  ensures !rendered.contains_unescaped_user_data",
    "CWE-78":  "  ensures result.was_not_injected",
    "CWE-611": "  ensures !result.resolved_external_entities",
    "CWE-352": "  ensures request.is_csrf_protected",
    "CWE-798": "  ensures credential.is_from_secure_store",
    "CWE-400": "  ensures result.size <= MAX_RESPONSE_SIZE",
    "CWE-269": "  ensures operation.performed_with_minimum_privilege",
    "CWE-601": "  ensures redirect.is_safe",
}

# CWE → method signature template (Viper-like pseudocode).
_CWE_METHOD_TEMPLATE: dict[str, str] = {
    "CWE-89":  "method execute_query(query: QueryObject, params: List[Param]) returns (result: ResultSet)",
    "CWE-639": "method get_resource(caller: User, resource_id: ResourceId) returns (result: Resource)",
    "CWE-918": "method fetch_url(url: String) returns (response: HTTPResponse)",
    "CWE-22":  "method open_file(path: String, BASE_DIR: String) returns (handle: FileHandle)",
    "CWE-287": "method privileged_action(session: Session) returns (result: ActionResult)",
    "CWE-284": "method access_resource(caller: Principal, required_permission: Permission) returns (result: Resource)",
    "CWE-94":  "method evaluate_code(code_input: String, source: Source) returns (result: Value)",
    "CWE-502": "method deserialize(data: Bytes, source: Source, KEY: SigningKey) returns (result: Object)",
    "CWE-79":  "method render_template(template: String, user_content: String) returns (rendered: HTML)",
    "CWE-78":  "method run_command(command: String, args: List[String]) returns (result: ProcessResult)",
    "CWE-611": "method parse_xml(xml_data: Bytes, xml_parser: Parser) returns (result: XMLDocument)",
    "CWE-352": "method process_form(request: HTTPRequest) returns (result: FormResult)",
    "CWE-798": "method authenticate(credential: Credential) returns (result: AuthToken)",
    "CWE-400": "method process_input(request: HTTPRequest, input_size: Int) returns (result: Response)",
    "CWE-269": "method perform_operation(operation: Operation, caller: Principal) returns (result: OperationResult)",
    "CWE-601": "method redirect(url: String) returns (response: HTTPResponse)",
}


class STIXToViperGenerator:
    """Auto-generates Nagini/Viper verification specs from STIX 2.1 indicators.

    Each generated spec consists of:
    - A method signature inferred from the vulnerability class.
    - Preconditions derived from the CWE property template.
    - Postconditions ensuring the vulnerability cannot occur.
    - A provenance comment linking back to the originating CVE/STIX ID.

    The generator is invoked by the Layer 4 orchestrator whenever a new
    ThreatIndicator is ingested, automatically extending Layer 3's
    verification coverage.
    """

    def __init__(self) -> None:
        self._generated_specs: dict[str, str] = {}

    def generate(self, indicator: "ThreatIndicator") -> str:
        """Generate a Viper-style verification spec from a ThreatIndicator.

        Args:
            indicator: A :class:`~core.ThreatIndicator` instance.

        Returns:
            str: Generated specification string (Viper/Nagini pseudocode).
        """
        from core.threat_indicator import ThreatIndicator
        if not isinstance(indicator, ThreatIndicator):
            raise TypeError("Expected ThreatIndicator instance.")

        spec_lines = [
            f"// ActivGuard — Auto-generated Viper spec",
            f"// Source: {indicator.source}",
            f"// Indicator ID: {indicator.id}",
            f"// Severity: {indicator.severity} (CVSS: {indicator.cvss_score})",
            f"// CWEs: {', '.join(indicator.cwe)}",
            f"// Generated: {datetime.now(tz=timezone.utc).isoformat()}",
            f"// Description: {indicator.description[:200]}",
            "",
        ]

        # Generate a spec for each CWE in the indicator.
        if not indicator.cwe:
            spec_lines.append("// No CWE mapping available — generic spec.")
            spec_lines.extend(self._generate_generic_spec(indicator))
        else:
            for cwe in indicator.cwe:
                spec = self._generate_cwe_spec(cwe, indicator)
                spec_lines.extend(spec)
                spec_lines.append("")

        full_spec = "\n".join(spec_lines)
        self._generated_specs[indicator.id] = full_spec
        logger.info(
            "Generated Viper spec for %s (CWEs: %s)",
            indicator.id,
            indicator.cwe,
        )
        return full_spec

    def _generate_cwe_spec(
        self, cwe: str, indicator: "ThreatIndicator"
    ) -> list[str]:
        """Generate Viper method spec lines for a single CWE.

        Args:
            cwe: CWE identifier.
            indicator: Source indicator for provenance.

        Returns:
            list[str]: Spec lines.
        """
        method_sig = _CWE_METHOD_TEMPLATE.get(cwe, f"method vulnerability_check_{cwe}()")
        precondition = _CWE_PRECONDITION.get(cwe, "  requires input_validated()")
        postcondition = _CWE_POSTCONDITION.get(cwe, "  ensures no_vulnerability()")
        return [
            f"// {cwe} — {self._cwe_description(cwe)}",
            method_sig,
            precondition,
            postcondition,
            "{",
            f"  // Implementation must satisfy: {indicator.semantic_rule[:200]}",
            "}",
        ]

    def _generate_generic_spec(
        self, indicator: "ThreatIndicator"
    ) -> list[str]:
        """Generate a generic spec when no CWE is available.

        Args:
            indicator: Source indicator.

        Returns:
            list[str]: Spec lines.
        """
        return [
            f"method security_check_{indicator.id.replace('-', '_')}()",
            f"  requires input_validated(user_input)",
            f"  ensures no_vulnerability_introduced(result)",
            "{",
            f"  // Semantic rule: {indicator.semantic_rule[:200]}",
            "}",
        ]

    def _map_cwe_to_property(self, cwe: str) -> str:
        """Return the precondition string for a CWE identifier.

        Args:
            cwe: CWE identifier.

        Returns:
            str: Precondition string, or a generic fallback.
        """
        return _CWE_PRECONDITION.get(cwe, "  requires input_validated(user_input)")

    def _generate_precondition(self, vuln_class: str) -> str:
        """Return the precondition for a vulnerability class name.

        Args:
            vuln_class: Vulnerability class (e.g. "IDOR", "SQLi").

        Returns:
            str: Precondition string.
        """
        class_cwe_map = {
            "IDOR": "CWE-639",
            "SQLi": "CWE-89",
            "SSRF": "CWE-918",
            "path_traversal": "CWE-22",
            "auth_bypass": "CWE-287",
            "XSS": "CWE-79",
            "command_injection": "CWE-78",
            "deserialization": "CWE-502",
        }
        cwe = class_cwe_map.get(vuln_class, "")
        return _CWE_PRECONDITION.get(cwe, "  requires input_validated(user_input)")

    def _generate_postcondition(self, vuln_class: str) -> str:
        """Return the postcondition for a vulnerability class name.

        Args:
            vuln_class: Vulnerability class.

        Returns:
            str: Postcondition string.
        """
        class_cwe_map = {
            "IDOR": "CWE-639",
            "SQLi": "CWE-89",
            "SSRF": "CWE-918",
            "path_traversal": "CWE-22",
            "auth_bypass": "CWE-287",
            "XSS": "CWE-79",
            "command_injection": "CWE-78",
            "deserialization": "CWE-502",
        }
        cwe = class_cwe_map.get(vuln_class, "")
        return _CWE_POSTCONDITION.get(cwe, "  ensures no_vulnerability(result)")

    def save_spec(self, spec: str, path: str) -> None:
        """Save a generated spec to a file.

        Args:
            spec: Specification string.
            path: Output file path (will create parent directories).
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(spec, encoding="utf-8")
        logger.info("Saved Viper spec to %s", path)

    def save_all(self, directory: str) -> list[str]:
        """Save all generated specs to separate files in a directory.

        Args:
            directory: Target directory.

        Returns:
            list[str]: Paths of written files.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for indicator_id, spec in self._generated_specs.items():
            safe_id = indicator_id.replace("/", "_").replace(":", "_")
            file_path = dir_path / f"{safe_id}.vpr"
            file_path.write_text(spec, encoding="utf-8")
            written.append(str(file_path))
        logger.info("Saved %d specs to %s", len(written), directory)
        return written

    @staticmethod
    def _cwe_description(cwe: str) -> str:
        """Return a short description for a CWE identifier.

        Args:
            cwe: CWE identifier.

        Returns:
            str: Short description.
        """
        descriptions = {
            "CWE-89": "SQL Injection",
            "CWE-639": "IDOR — Authorization Through User-Controlled Key",
            "CWE-918": "SSRF — Server-Side Request Forgery",
            "CWE-22": "Path Traversal — Improper Limitation of Pathname",
            "CWE-287": "Improper Authentication",
            "CWE-284": "Improper Access Control",
            "CWE-94": "Code Injection",
            "CWE-502": "Deserialization of Untrusted Data",
            "CWE-79": "Cross-site Scripting (XSS)",
            "CWE-78": "OS Command Injection",
            "CWE-611": "XML External Entity (XXE)",
            "CWE-352": "CSRF — Cross-Site Request Forgery",
            "CWE-798": "Use of Hard-coded Credentials",
            "CWE-400": "Uncontrolled Resource Consumption",
            "CWE-269": "Improper Privilege Management",
            "CWE-601": "URL Redirection to Untrusted Site",
        }
        return descriptions.get(cwe, f"Vulnerability class {cwe}")

    def __repr__(self) -> str:
        return (
            f"STIXToViperGenerator(specs_generated={len(self._generated_specs)})"
        )
