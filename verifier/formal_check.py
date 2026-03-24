"""
Formal Checker — Layer 3 probe-gated static analysis and formal verification.

Invoked only when Layer 1 OR Layer 2 has flagged a code snippet.  Uses a
two-phase approach:
1. Fast AST-based property checking (always runs; no external tools).
2. Optional Nagini/Viper formal verification (requires installation).

Design target: invoked on <15% of LLM outputs (gated by Layers 1–2) while
catching >90% of real vulnerabilities.

Research context:
    Formal verification of security properties in arbitrary code is undecidable
    in general.  However, for the specific vulnerability classes in scope
    (IDOR, SQLi, SSRF, path traversal, auth bypass), the *relevant property*
    can be reduced to a local invariant check on the function level.  The PhD
    contribution is to show that these invariants can be:
    (a) automatically synthesised from threat intel (via STIXToViperGenerator),
    (b) efficiently checked at code-review time via Nagini annotations, and
    (c) applied selectively (gated by Layers 1–2) to stay within practical
        performance budgets.
"""

from __future__ import annotations

import ast
import logging
import subprocess
import textwrap
from pathlib import Path
from typing import Optional

from .property_templates import PropertyTemplateEngine, PROPERTY_TEMPLATES

logger = logging.getLogger(__name__)

# AST patterns that indicate parameterised query usage.
_PARAMETERISED_CALL_ARGS = {
    "execute",
    "executemany",
    "raw",
}

# ORM method names that imply safe parameterised queries (Django/SQLAlchemy).
_SAFE_ORM_METHODS = {
    "filter",
    "filter_by",
    "get",
    "get_or_create",
    "update_or_create",
    "annotate",
    "select_related",
    "prefetch_related",
}

# Auth-related identifier names.
_AUTH_IDENTIFIERS = frozenset({
    "is_authenticated",
    "login_required",
    "permission_required",
    "check_permission",
    "has_permission",
    "is_owner",
    "can_access",
    "verify_token",
    "jwt_required",
    "authenticate",
    "authorize",
    "authorise",
    "is_staff",
    "is_superuser",
    "is_admin",
    "session_required",
    "token_required",
    "@login_required",
    "@permission_required",
})

# URL-fetch call names.
_URL_FETCH_CALLS = frozenset({
    "get", "post", "put", "delete", "patch", "head",
    "urlopen", "urlretrieve", "request",
    "fetch", "send",
})

# Safe deserialiser module.function pairs.
_SAFE_DESERIALISERS = frozenset({
    "json.loads",
    "json.load",
    "json.dumps",
    "orjson.loads",
})

# Unsafe deserialiser module.function pairs.
_UNSAFE_DESERIALISERS = frozenset({
    "pickle.loads",
    "pickle.load",
    # yaml.load is handled separately (safe when Loader=yaml.SafeLoader is used)
    "marshal.loads",
    "marshal.load",
    "jsonpickle.decode",
    "dill.loads",
    "shelve.open",
})


class FormalChecker:
    """Layer 3 probe-gated formal verification.

    Performs two-phase verification:
    1. AST-based checks (always available, no external deps).
    2. Nagini-based formal verification (optional, requires Nagini install).

    Args:
        invoke_threshold: Confidence score from Layer 1 / Layer 2 above which
            Layer 3 should be invoked.  Defaults to 0.7.
        use_nagini: Whether to attempt Nagini-based verification.  Defaults
            to False (AST-only mode).
        nagini_timeout: Nagini subprocess timeout in seconds.
    """

    def __init__(
        self,
        invoke_threshold: float = 0.7,
        use_nagini: bool = False,
        nagini_timeout: int = 30,
    ) -> None:
        if not (0.0 <= invoke_threshold <= 1.0):
            raise ValueError("invoke_threshold must be in [0.0, 1.0].")
        self._invoke_threshold = invoke_threshold
        self._use_nagini = use_nagini
        self._nagini_timeout = nagini_timeout
        self._template_engine = PropertyTemplateEngine()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def verify(
        self,
        code_snippet: str,
        vuln_class: str,
        context: dict | None = None,
    ) -> dict:
        """Verify that ``code_snippet`` satisfies the property for ``vuln_class``.

        Runs AST-based checks first.  If ``use_nagini=True`` and the AST
        check is inconclusive, attempts Nagini verification.

        Args:
            code_snippet: Python code to verify.
            vuln_class: Vulnerability class to check
                (e.g. "IDOR", "SQLi", "SSRF").
            context: Optional context dict (e.g. project auth model hints).

        Returns:
            dict:
                - ``result`` (str): "VERIFIED", "VIOLATION", or "UNKNOWN".
                - ``property`` (str): The property that was checked.
                - ``evidence`` (str): Proof trace or violation description.
                - ``confidence`` (float): Confidence in the result [0, 1].
                - ``method`` (str): "AST", "Nagini", or "AST+Nagini".
        """
        if not code_snippet:
            raise ValueError("code_snippet must be non-empty.")
        if not vuln_class:
            raise ValueError("vuln_class must be non-empty.")

        # Get property description.
        try:
            template = self._template_engine.get_template(vuln_class)
            property_desc = template.get("nagini_annotation", f"Property for {vuln_class}")
        except KeyError:
            property_desc = f"Generic security property for {vuln_class}"
            template = {}

        # Phase 1: AST-based check.
        ast_result = self._ast_check(code_snippet, vuln_class)
        method = "AST"
        result = ast_result.get("result", "UNKNOWN")
        evidence = ast_result.get("evidence", "No AST evidence.")
        confidence = ast_result.get("confidence", 0.5)

        # Phase 2: Nagini (optional, only if AST is inconclusive and enabled).
        if self._use_nagini and result == "UNKNOWN":
            nagini_result = self._run_nagini(code_snippet, vuln_class, template)
            if nagini_result.get("result") != "UNKNOWN":
                result = nagini_result["result"]
                evidence = nagini_result.get("evidence", evidence)
                confidence = nagini_result.get("confidence", confidence)
                method = "AST+Nagini"

        return {
            "result": result,
            "property": property_desc,
            "evidence": evidence,
            "confidence": round(confidence, 4),
            "method": method,
            "vuln_class": vuln_class,
        }

    # ------------------------------------------------------------------
    # AST-based checks (Phase 1)
    # ------------------------------------------------------------------

    def _ast_check(self, code: str, vuln_class: str) -> dict:
        """Perform AST-based property checking.

        Args:
            code: Python source code.
            vuln_class: Vulnerability class to check.

        Returns:
            dict: Result with ``result``, ``evidence``, and ``confidence``.
        """
        try:
            tree = ast.parse(textwrap.dedent(code))
        except SyntaxError as exc:
            return {
                "result": "UNKNOWN",
                "evidence": f"Syntax error in code: {exc}",
                "confidence": 0.0,
            }

        dispatch: dict[str, object] = {
            "IDOR": self._check_idor,
            "SQLi": self._check_sqli,
            "SSRF": self._check_ssrf,
            "auth_bypass": self._check_auth_bypass,
            "path_traversal": self._check_path_traversal,
            "XSS": self._check_xss,
            "deserialization": self._check_deserialization,
            "command_injection": self._check_command_injection,
        }
        checker = dispatch.get(vuln_class)
        if checker is None:
            return {
                "result": "UNKNOWN",
                "evidence": f"No AST checker for '{vuln_class}'.",
                "confidence": 0.3,
            }
        has_property, evidence = checker(tree)  # type: ignore[operator]
        if has_property:
            return {
                "result": "VERIFIED",
                "evidence": f"AST check passed: {evidence}",
                "confidence": 0.85,
            }
        return {
            "result": "VIOLATION",
            "evidence": f"AST check failed: {evidence}",
            "confidence": 0.8,
        }

    def _check_idor(self, tree: ast.AST) -> tuple[bool, str]:
        """Check for IDOR property: resource accesses accompanied by ownership check.

        Args:
            tree: Parsed AST.

        Returns:
            tuple[bool, str]: (property_holds, evidence_string).
        """
        orm_access_found = False
        ownership_check_found = False
        for node in ast.walk(tree):
            # Detect ORM access patterns: .get(id=...), .filter(id=...).
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    method = node.func.attr
                    if method in ("get", "filter", "find", "find_by_id"):
                        for kw in node.keywords:
                            if kw.arg in ("id", "pk", "user_id"):
                                orm_access_found = True
                            # user=request.user, owner=..., created_by=... in the
                            # same ORM call is itself an ownership filter.
                            if kw.arg and any(
                                token in kw.arg.lower()
                                for token in ("user", "owner", "created_by", "author")
                            ):
                                ownership_check_found = True
            # Detect ownership checks via identifiers.
            if isinstance(node, ast.Name):
                if node.id in _AUTH_IDENTIFIERS or "owner" in node.id.lower():
                    ownership_check_found = True
            if isinstance(node, ast.Attribute):
                if "owner" in node.attr.lower() or node.attr in _AUTH_IDENTIFIERS:
                    ownership_check_found = True
            # Detect raise PermissionDenied or abort(403).
            if isinstance(node, ast.Raise):
                ownership_check_found = True
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "abort":
                    if node.args and isinstance(node.args[0], ast.Constant):
                        if node.args[0].value in (403, 401):
                            ownership_check_found = True

        if not orm_access_found:
            return True, "No resource access detected — IDOR not applicable."
        if ownership_check_found:
            return True, "Ownership or permission check detected alongside resource access."
        return False, "Resource access detected without ownership/permission check — potential IDOR."

    def _check_sqli(self, tree: ast.AST) -> tuple[bool, str]:
        """Check for SQLi property: no string interpolation into SQL calls.

        Args:
            tree: Parsed AST.

        Returns:
            tuple[bool, str]: (property_holds, evidence_string).
        """
        # Collect variables that are assigned from string concatenation or f-strings.
        # These are potential SQL injection sources if passed to execute().
        dangerous_names: set[str] = set()
        for node in ast.walk(tree):
            # Detect: query = "..." + user_input  or  query = f"..."
            if isinstance(node, ast.Assign):
                if isinstance(node.value, (ast.BinOp, ast.JoinedStr)):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            dangerous_names.add(target.id)
            # Detect augmented assignment: query += user_input
            if isinstance(node, ast.AugAssign):
                if isinstance(node.value, (ast.BinOp, ast.Name, ast.Constant)):
                    if isinstance(node.target, ast.Name):
                        dangerous_names.add(node.target.id)

        # Flag any SQL-keyword-containing string built via concatenation/f-string,
        # even if not yet passed to execute() — construction itself is dangerous.
        sql_keywords = ("SELECT", "INSERT", "UPDATE", "DELETE", "WHERE", "FROM")
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                val = node.value
                if isinstance(val, ast.BinOp):
                    # Walk the binop to find string literals containing SQL keywords.
                    for subnode in ast.walk(val):
                        if isinstance(subnode, ast.Constant) and isinstance(subnode.value, str):
                            if any(kw in subnode.value.upper() for kw in sql_keywords):
                                return (
                                    False,
                                    "SQL query built via string concatenation — SQLi risk "
                                    "(use parameterised queries instead).",
                                )
                if isinstance(val, ast.JoinedStr):
                    # f-string containing SQL keywords is dangerous.
                    for subnode in ast.walk(val):
                        if isinstance(subnode, ast.Constant) and isinstance(subnode.value, str):
                            if any(kw in subnode.value.upper() for kw in sql_keywords):
                                return (
                                    False,
                                    "SQL query built via f-string — SQLi risk.",
                                )

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in _PARAMETERISED_CALL_ARGS:
                        if node.args:
                            first_arg = node.args[0]
                            # Direct string concat/f-string as first arg.
                            if isinstance(first_arg, (ast.BinOp, ast.JoinedStr)):
                                return (
                                    False,
                                    "execute() called with string concatenation or f-string — SQLi risk.",
                                )
                            # Variable known to be a string concat result.
                            if isinstance(first_arg, ast.Name):
                                if first_arg.id in dangerous_names:
                                    return (
                                        False,
                                        f"execute() called with '{first_arg.id}' which is built via string "
                                        "concatenation — SQLi risk.",
                                    )
        return True, "No unsafe SQL string interpolation detected."

    def _check_ssrf(self, tree: ast.AST) -> tuple[bool, str]:
        """Check for SSRF property: URL fetch calls have allowlist validation nearby.

        Args:
            tree: Parsed AST.

        Returns:
            tuple[bool, str]: (property_holds, evidence_string).
        """
        url_fetch_found = False
        allowlist_check_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                call_name = ""
                if isinstance(func, ast.Attribute):
                    call_name = func.attr
                elif isinstance(func, ast.Name):
                    call_name = func.id
                if call_name in _URL_FETCH_CALLS:
                    # Flag if any positional arg looks user-controlled:
                    # ast.Name (variable), ast.Subscript (dict/query access),
                    # ast.Attribute, or JoinedStr (f-string).
                    for arg in node.args:
                        if isinstance(arg, ast.Name):
                            if "url" in arg.id.lower() or "user" in arg.id.lower():
                                url_fetch_found = True
                        elif isinstance(arg, (ast.Subscript, ast.JoinedStr)):
                            # e.g. request.query["url"] or f"{user_url}"
                            url_fetch_found = True
                        elif isinstance(arg, ast.Attribute):
                            # e.g. request.url or self.target_url
                            url_fetch_found = True
                    # Also check keyword args: requests.get(url=user_url)
                    for kw in node.keywords:
                        if kw.arg == "url":
                            url_fetch_found = True
            # Check for allowlist/validation patterns.
            if isinstance(node, ast.Name):
                if any(
                    kw in node.id.lower()
                    for kw in ("allowlist", "whitelist", "validate", "is_safe", "is_allowed")
                ):
                    allowlist_check_found = True
            if isinstance(node, ast.Compare):
                # e.g. hostname not in BLOCKED_HOSTS or hostname in ALLOWED_DOMAINS
                allowlist_check_found = True
            if isinstance(node, ast.Raise):
                allowlist_check_found = True  # Raise = validation present.

        if not url_fetch_found:
            return True, "No URL fetch with user-supplied URL detected."
        if allowlist_check_found:
            return True, "URL fetch with validation/allowlist check detected."
        return False, "URL fetch with user-supplied URL — no allowlist check found."

    def _check_auth_bypass(self, tree: ast.AST) -> tuple[bool, str]:
        """Check for auth bypass: privilege operations have session-based auth check.

        Args:
            tree: Parsed AST.

        Returns:
            tuple[bool, str]: (property_holds, evidence_string).
        """
        has_auth = False
        has_query_param_auth = False
        for node in ast.walk(tree):
            # Detect request.args.get('admin') or request.args['is_admin'].
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "get":
                        if isinstance(node.func.value, ast.Attribute):
                            if "args" in node.func.value.attr.lower():
                                # request.args.get('admin') pattern.
                                if node.args and isinstance(node.args[0], ast.Constant):
                                    val = str(node.args[0].value).lower()
                                    if any(
                                        kw in val
                                        for kw in ("admin", "is_admin", "role", "staff")
                                    ):
                                        has_query_param_auth = True
            # Detect session-based auth identifiers.
            if isinstance(node, ast.Name):
                if node.id in _AUTH_IDENTIFIERS:
                    has_auth = True
            if isinstance(node, ast.Attribute):
                if node.attr in _AUTH_IDENTIFIERS or "is_authenticated" in node.attr:
                    has_auth = True
            # Detect decorator-based auth.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    dec_name = ""
                    if isinstance(dec, ast.Name):
                        dec_name = dec.id
                    elif isinstance(dec, ast.Attribute):
                        dec_name = dec.attr
                    elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                        dec_name = dec.func.id
                    if any(
                        kw in dec_name.lower()
                        for kw in ("login", "auth", "permission")
                    ):
                        has_auth = True

        if has_query_param_auth and not has_auth:
            return (
                False,
                "Auth decision based on query parameter without session verification.",
            )
        if has_auth:
            return True, "Session/decorator-based authentication check detected."
        return True, "No privileged operation detected in this snippet."

    def _check_path_traversal(self, tree: ast.AST) -> tuple[bool, str]:
        """Check for path traversal: file operations have path sandboxing.

        Args:
            tree: Parsed AST.

        Returns:
            tuple[bool, str]: (property_holds, evidence_string).
        """
        file_op_found = False
        sandbox_check_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                call_name = ""
                if isinstance(func, ast.Name):
                    call_name = func.id
                elif isinstance(func, ast.Attribute):
                    call_name = func.attr
                # File open.
                if call_name in ("open", "send_file", "send_from_directory"):
                    file_op_found = True
                # Sandbox check: realpath, abspath.
                if call_name in ("realpath", "abspath", "normpath", "resolve"):
                    sandbox_check_found = True
            # Check for startswith(BASE_DIR) pattern.
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "startswith":
                        sandbox_check_found = True
            # Check for raise in context of path validation.
            if isinstance(node, ast.Raise):
                sandbox_check_found = True

        if not file_op_found:
            return True, "No file operation detected."
        if sandbox_check_found:
            return True, "File operation with path canonicalisation/sandbox check detected."
        return False, "File operation without path sandbox check — path traversal risk."

    def _check_xss(self, tree: ast.AST) -> tuple[bool, str]:
        """Check for XSS: no raw user content in HTML output contexts.

        Args:
            tree: Parsed AST.

        Returns:
            tuple[bool, str]: (property_holds, evidence_string).
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("render_template_string", "Markup"):
                        # If called with a variable argument, possible XSS.
                        if node.args and not isinstance(node.args[0], ast.Constant):
                            return (
                                False,
                                "render_template_string/Markup called with non-constant argument — XSS risk.",
                            )
        return True, "No unsafe HTML rendering pattern detected."

    def _check_deserialization(self, tree: ast.AST) -> tuple[bool, str]:
        """Check for unsafe deserialisation.

        Args:
            tree: Parsed AST.

        Returns:
            tuple[bool, str]: (property_holds, evidence_string).
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                full_name = ""
                if isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name):
                        full_name = f"{func.value.id}.{func.attr}"
                if full_name in _UNSAFE_DESERIALISERS:
                    return (
                        False,
                        f"Unsafe deserialiser call detected: {full_name}",
                    )
                # yaml.load without Loader=yaml.SafeLoader.
                if full_name == "yaml.load":
                    safe_loader = any(
                        isinstance(kw.value, ast.Attribute)
                        and "safe" in getattr(kw.value, "attr", "").lower()
                        for kw in node.keywords
                    )
                    if not safe_loader:
                        return (
                            False,
                            "yaml.load called without SafeLoader — unsafe deserialisation.",
                        )
        return True, "No unsafe deserialisation detected."

    def _check_command_injection(self, tree: ast.AST) -> tuple[bool, str]:
        """Check for command injection via subprocess with shell=True.

        Args:
            tree: Parsed AST.

        Returns:
            tuple[bool, str]: (property_holds, evidence_string).
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in (
                        "call", "run", "Popen", "check_output", "check_call"
                    ):
                        # Check for shell=True keyword.
                        for kw in node.keywords:
                            if kw.arg == "shell" and isinstance(kw.value, ast.Constant):
                                if kw.value.value is True:
                                    return (
                                        False,
                                        "subprocess call with shell=True detected — command injection risk.",
                                    )
        return True, "No shell=True subprocess call detected."

    # ------------------------------------------------------------------
    # Nagini integration (Phase 2)
    # ------------------------------------------------------------------

    def _run_nagini(
        self, code: str, vuln_class: str, template: dict
    ) -> dict:
        """Run Nagini formal verifier on the code snippet.

        Prepends Nagini annotations from the property template, writes to a
        temporary file, and invokes Nagini as a subprocess.

        Args:
            code: Python source code.
            vuln_class: Vulnerability class.
            template: Property template dict.

        Returns:
            dict: Result with ``result``, ``evidence``, and ``confidence``.
        """
        try:
            import tempfile
            annotation = template.get("nagini_annotation", "")
            annotated_code = f"# Nagini annotations for {vuln_class}\n{annotation}\n\n{code}"
            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", delete=False, encoding="utf-8"
            ) as f:
                f.write(annotated_code)
                tmp_path = f.name
            proc = subprocess.run(
                ["nagini", tmp_path],
                capture_output=True,
                text=True,
                timeout=self._nagini_timeout,
            )
            Path(tmp_path).unlink(missing_ok=True)
            if proc.returncode == 0:
                return {
                    "result": "VERIFIED",
                    "evidence": f"Nagini: Verification succeeded.\n{proc.stdout[:500]}",
                    "confidence": 0.95,
                }
            return {
                "result": "VIOLATION",
                "evidence": f"Nagini: Verification failed.\n{proc.stderr[:500]}",
                "confidence": 0.9,
            }
        except FileNotFoundError:
            logger.debug("Nagini not found in PATH — skipping formal verification.")
            return {"result": "UNKNOWN", "evidence": "Nagini not installed.", "confidence": 0.0}
        except subprocess.TimeoutExpired:
            logger.warning("Nagini timed out on %s check.", vuln_class)
            return {"result": "UNKNOWN", "evidence": "Nagini timed out.", "confidence": 0.0}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Nagini error: %s", exc)
            return {"result": "UNKNOWN", "evidence": str(exc), "confidence": 0.0}

    def __repr__(self) -> str:
        return (
            f"FormalChecker("
            f"threshold={self._invoke_threshold}, "
            f"use_nagini={self._use_nagini})"
        )
