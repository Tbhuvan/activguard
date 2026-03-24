"""
Property Templates — formal verification specifications per vulnerability class.

Each template encodes the security property that must hold for a given
vulnerability class, expressed in three complementary forms:
1. Natural-language description (for human reviewers).
2. Nagini-compatible Python annotation (for direct use with Nagini verifier).
3. Viper specification (for use with the Viper verification infrastructure).

Additionally, each template names an AST checker function that
:class:`~verifier.formal_check.FormalChecker` uses for fast static analysis.

Research context:
    The property templates are the *formal contract* of the system.  By making
    security requirements explicit and machine-checkable, the verifier can
    provide *proof* (not just a warning) that a code snippet satisfies or
    violates a security invariant.  This is a qualitative advance over
    heuristic-based scanners.

    The Nagini integration enables the PhD's formal methods contribution:
    demonstrating that a subset of code-level IDOR/SQLi/SSRF properties can
    be verified in polynomial time under reasonable assumptions about the
    code structure.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Master property template dictionary
# ---------------------------------------------------------------------------

PROPERTY_TEMPLATES: dict[str, dict[str, str]] = {
    "IDOR": {
        "description": (
            "Every resource access must include an ownership or permission "
            "verification step before returning the resource.  Specifically: "
            "for any call to an ORM method (get, filter, find_by_id) the "
            "result must be checked against the requesting user's identity "
            "before being returned to the caller."
        ),
        "nagini_annotation": (
            "Requires(ownership_verified(user, resource_id))\n"
            "Ensures(result.owner == user or has_permission(user, result))"
        ),
        "viper_spec": (
            "method get_resource(user: Ref, resource_id: Int) returns (r: Ref)\n"
            "  requires ownership_check(user, resource_id)\n"
            "  ensures r.owner_id == user.id || has_explicit_permission(user, r)\n"
            "  ensures acc(r.data)"
        ),
        "check_function": "has_ownership_check",
        "cwe": "CWE-639",
        "owasp": "A01:2021 Broken Access Control",
    },
    "SQLi": {
        "description": (
            "All database queries must use parameterised statements or an ORM "
            "that prevents direct string interpolation of user input into SQL.  "
            "No raw string concatenation, f-strings, or %-formatting of "
            "user-controlled values into SQL query strings is permitted."
        ),
        "nagini_annotation": (
            "Requires(is_parameterized(query) or uses_orm(query))\n"
            "Ensures(no_injection_possible(result))"
        ),
        "viper_spec": (
            "method execute_query(query: QueryObject) returns (result: ResultSet)\n"
            "  requires query.is_parameterized\n"
            "  requires forall p: Param :: query.params.contains(p) ==> p.is_escaped\n"
            "  ensures result.is_safe"
        ),
        "check_function": "has_parameterized_query",
        "cwe": "CWE-89",
        "owasp": "A03:2021 Injection",
    },
    "SSRF": {
        "description": (
            "All outbound URL fetch operations must validate the target URL "
            "against an allowlist of approved domains/IP ranges before "
            "establishing the connection.  Private IP ranges (RFC-1918, "
            "loopback) must be explicitly blocked."
        ),
        "nagini_annotation": (
            "Requires(is_allowlisted(url) and not is_private_ip(url))\n"
            "Ensures(response.is_from_external_service)"
        ),
        "viper_spec": (
            "method fetch_url(url: String) returns (response: HTTPResponse)\n"
            "  requires allowlist_check(url)\n"
            "  requires !is_private_range(resolve_hostname(url))\n"
            "  ensures response.source_domain in ALLOWED_DOMAINS"
        ),
        "check_function": "has_url_allowlist_check",
        "cwe": "CWE-918",
        "owasp": "A10:2021 Server-Side Request Forgery",
    },
    "path_traversal": {
        "description": (
            "All file-system operations must restrict the resolved path to an "
            "approved base directory.  The canonical path (after resolving "
            "symlinks and '..' components) must start with the approved base "
            "directory prefix."
        ),
        "nagini_annotation": (
            "Requires(is_sandboxed(path, BASE_DIR))\n"
            "Ensures(realpath(result_path).startswith(BASE_DIR))"
        ),
        "viper_spec": (
            "method open_file(path: String, base_dir: String) returns (handle: FileHandle)\n"
            "  requires realpath(path).starts_with(base_dir)\n"
            "  requires !contains_traversal_sequence(path)\n"
            "  ensures handle.path.starts_with(base_dir)"
        ),
        "check_function": "has_path_sandbox_check",
        "cwe": "CWE-22",
        "owasp": "A01:2021 Broken Access Control",
    },
    "auth_bypass": {
        "description": (
            "All operations requiring elevated privileges must verify the "
            "requestor's identity and permissions via a server-side session "
            "or token mechanism.  Client-supplied parameters (query strings, "
            "headers, cookies) must never be the *sole* basis for a "
            "privilege grant."
        ),
        "nagini_annotation": (
            "Requires(is_authenticated(session) and session_not_expired(session))\n"
            "Ensures(operation_authorized(result, session.user))"
        ),
        "viper_spec": (
            "method privileged_action(session: Session) returns (result: Result)\n"
            "  requires session.is_authenticated\n"
            "  requires !session.is_expired\n"
            "  requires session.user.has_permission(required_permission)\n"
            "  ensures result.is_authorized"
        ),
        "check_function": "has_auth_check",
        "cwe": "CWE-287",
        "owasp": "A07:2021 Identification and Authentication Failures",
    },
    "XSS": {
        "description": (
            "All user-controlled data rendered in an HTML context must be "
            "HTML-escaped.  Template engines must use auto-escaping; "
            "raw/safe filters must not be applied to user-controlled data."
        ),
        "nagini_annotation": (
            "Requires(is_html_escaped(user_content) or template_auto_escapes)\n"
            "Ensures(no_script_injection(rendered_html))"
        ),
        "viper_spec": (
            "method render_content(content: String) returns (html: String)\n"
            "  requires html_escaped(content)\n"
            "  ensures !contains_unescaped_user_data(html)"
        ),
        "check_function": "has_html_escaping",
        "cwe": "CWE-79",
        "owasp": "A03:2021 Injection",
    },
    "deserialization": {
        "description": (
            "Deserialisation of objects from untrusted sources must use a safe "
            "deserialiser (e.g. json, not pickle/yaml.load/marshal) OR must "
            "validate the source via a cryptographic signature before "
            "deserialising."
        ),
        "nagini_annotation": (
            "Requires(is_trusted_source(data) or is_signed(data, SIGNING_KEY))\n"
            "Ensures(no_rce_via_deserialization(result))"
        ),
        "viper_spec": (
            "method deserialize(data: Bytes, source: Source) returns (obj: Object)\n"
            "  requires source.is_trusted || data.has_valid_signature(SIGNING_KEY)\n"
            "  ensures obj.is_safe"
        ),
        "check_function": "has_safe_deserializer",
        "cwe": "CWE-502",
        "owasp": "A08:2021 Software and Data Integrity Failures",
    },
    "command_injection": {
        "description": (
            "Subprocess and shell execution calls must not accept user-supplied "
            "strings as command arguments with shell=True.  If shell execution "
            "is required, arguments must be allowlisted and shell metacharacters "
            "stripped."
        ),
        "nagini_annotation": (
            "Requires(not shell_injection_possible(cmd_args))\n"
            "Ensures(result.exit_code_is_expected)"
        ),
        "viper_spec": (
            "method run_command(cmd: List[String]) returns (result: ProcessResult)\n"
            "  requires forall arg: String :: cmd.contains(arg) ==> is_allowlisted(arg)\n"
            "  requires !cmd.uses_shell_metacharacters\n"
            "  ensures !result.was_injected"
        ),
        "check_function": "has_command_allowlist",
        "cwe": "CWE-78",
        "owasp": "A03:2021 Injection",
    },
}


class PropertyTemplateEngine:
    """Manages formal verification property templates for vulnerability classes.

    Provides instantiation of templates with context-specific values and
    dynamic addition of templates from STIX indicators.

    Args:
        custom_templates: Optional dict of additional templates to merge
            with the built-in :data:`PROPERTY_TEMPLATES`.
    """

    def __init__(
        self,
        custom_templates: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self._templates: dict[str, dict[str, str]] = dict(PROPERTY_TEMPLATES)
        if custom_templates:
            for key, tmpl in custom_templates.items():
                self._templates[key] = tmpl
            logger.info(
                "PropertyTemplateEngine: merged %d custom templates.",
                len(custom_templates),
            )

    def get_template(self, vuln_class: str) -> dict:
        """Retrieve the full template for a vulnerability class.

        Args:
            vuln_class: Vulnerability class name (e.g. "IDOR", "SQLi").

        Returns:
            dict: Full template dict.

        Raises:
            KeyError: If no template exists for ``vuln_class``.
        """
        if vuln_class not in self._templates:
            raise KeyError(
                f"No property template for '{vuln_class}'.  "
                f"Available: {list(self._templates.keys())}"
            )
        return self._templates[vuln_class]

    def get_nagini_annotation(self, vuln_class: str) -> str:
        """Return the Nagini annotation string for a vulnerability class.

        Args:
            vuln_class: Vulnerability class name.

        Returns:
            str: Nagini-compatible annotation string.

        Raises:
            KeyError: If no template exists for ``vuln_class``.
        """
        return self.get_template(vuln_class)["nagini_annotation"]

    def get_viper_spec(self, vuln_class: str) -> str:
        """Return the Viper specification string for a vulnerability class.

        Args:
            vuln_class: Vulnerability class name.

        Returns:
            str: Viper specification string.

        Raises:
            KeyError: If no template exists for ``vuln_class``.
        """
        return self.get_template(vuln_class)["viper_spec"]

    def instantiate(self, vuln_class: str, context: dict) -> str:
        """Instantiate a Nagini annotation with context-specific values.

        Replaces placeholder tokens in the template with values from ``context``.
        Supported placeholders: ``{function_name}``, ``{resource_model}``,
        ``{url_var}``, ``{path_var}``.

        Args:
            vuln_class: Vulnerability class.
            context: Dict of placeholder replacements.

        Returns:
            str: Instantiated annotation string.
        """
        template = self.get_nagini_annotation(vuln_class)
        try:
            return template.format(**context)
        except KeyError:
            return template  # Return uninstantiated if context is incomplete.

    def add_from_stix(self, stix_indicator: dict) -> str:
        """Derive and register a new property template from a STIX indicator.

        Uses the STIXEncoder to extract CWE identifiers and build property
        strings, then registers the result under the CWE-derived class name.

        Args:
            stix_indicator: STIX 2.1 Indicator object dict.

        Returns:
            str: The new template key (e.g. "CWE-639-custom").
        """
        from rag.stix_encoder import STIXEncoder
        encoder = STIXEncoder()
        cwe_ids = encoder.extract_cwe(stix_indicator)
        primary_cwe = cwe_ids[0] if cwe_ids else "unknown"
        formal_property = encoder.stix_to_formal_property(stix_indicator)
        template_key = f"{primary_cwe}-stix"
        description = stix_indicator.get("description", stix_indicator.get("name", ""))
        self._templates[template_key] = {
            "description": description[:500],
            "nagini_annotation": formal_property,
            "viper_spec": f"// Auto-generated from STIX: {stix_indicator.get('id', '')}",
            "check_function": "has_generic_security_check",
            "cwe": primary_cwe,
            "owasp": "N/A",
        }
        logger.info(
            "Added STIX-derived template '%s' for CWE %s", template_key, primary_cwe
        )
        return template_key

    def list_classes(self) -> list[str]:
        """Return all registered vulnerability class names.

        Returns:
            list[str]: Sorted list of class names.
        """
        return sorted(self._templates.keys())

    def get_check_function_name(self, vuln_class: str) -> Optional[str]:
        """Return the AST check function name for a vulnerability class.

        Args:
            vuln_class: Vulnerability class name.

        Returns:
            str | None: Check function name, or None if not in template.
        """
        tmpl = self._templates.get(vuln_class)
        if tmpl:
            return tmpl.get("check_function")
        return None

    def __len__(self) -> int:
        return len(self._templates)

    def __repr__(self) -> str:
        return f"PropertyTemplateEngine(templates={list(self._templates.keys())})"
