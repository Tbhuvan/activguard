"""
Auth Extractor — extracts the authentication and authorisation model from a codebase.

Builds a structured representation of the project's access-control model
(who can access what, under what conditions) so that Layer 2 RAG can reason
about whether a given LLM-generated code snippet violates the project's
existing security invariants.

Research rationale:
    Generic vulnerability detectors (e.g. CodeQL) know about CWE patterns
    but do not understand the *project-specific* access model.  An IDOR
    vulnerability is only an IDOR if the object is accessible to users who
    should not have access — and that depends on the project's own
    authorisation logic.  By extracting the auth model, ActivGuard can make
    *context-specific* vulnerability judgements, reducing false positives
    dramatically compared to project-agnostic scanners.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Identifiers considered auth-related.
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
    "require_auth",
    "authentication_classes",
    "permission_classes",
    "is_staff",
    "is_superuser",
    "is_admin",
    "session_required",
    "token_required",
    "requires_roles",
})

# ORM access patterns that should be checked for ownership.
_RESOURCE_ACCESS_PATTERNS = frozenset({
    "objects.get",
    "objects.filter",
    "objects.all",
    "query.get",
    "query.filter",
    "session.get",
    "find_by_id",
    "get_by_id",
    "find_one",
    "fetchone",
    "fetchall",
})


@dataclass
class AuthFunction:
    """Represents an authentication/authorisation function found in the codebase.

    Attributes:
        name: Function name.
        file_path: Source file containing this function.
        line_number: Line number of the function definition.
        checks: List of auth-related identifiers found in the function body.
        resource_accesses: ORM/DB access patterns found in the body.
        is_decorator: Whether this function is used as a decorator.
        source_snippet: First 400 characters of the function source.
    """

    name: str
    file_path: str
    line_number: int
    checks: list[str] = field(default_factory=list)
    resource_accesses: list[str] = field(default_factory=list)
    is_decorator: bool = False
    source_snippet: str = ""


@dataclass
class AuthModel:
    """Structured representation of a codebase's authentication/access model.

    Attributes:
        project_path: Root path of the analysed project.
        auth_functions: List of extracted auth/permission functions.
        decorators: Names of functions used as auth decorators.
        protected_endpoints: Paths/view names that have auth decorators.
        unprotected_resource_accesses: Functions that access resources without
            any detected auth check (potential IDOR risk).
        auth_middleware: Detected auth middleware classes/functions.
    """

    project_path: str
    auth_functions: list[AuthFunction] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    protected_endpoints: list[str] = field(default_factory=list)
    unprotected_resource_accesses: list[str] = field(default_factory=list)
    auth_middleware: list[str] = field(default_factory=list)

    def to_rag_documents(self) -> list[str]:
        """Convert the auth model to natural-language documents for RAG indexing.

        Returns:
            list[str]: One document per auth function plus a summary document.
        """
        docs: list[str] = []
        # Summary document.
        docs.append(
            f"Project auth model: {len(self.auth_functions)} auth functions, "
            f"{len(self.decorators)} auth decorators, "
            f"{len(self.protected_endpoints)} protected endpoints, "
            f"{len(self.unprotected_resource_accesses)} potentially unprotected "
            f"resource accesses."
        )
        # Per-function documents.
        for af in self.auth_functions:
            checks_str = ", ".join(af.checks) if af.checks else "none"
            accesses_str = ", ".join(af.resource_accesses) if af.resource_accesses else "none"
            docs.append(
                f"Auth function '{af.name}' in {Path(af.file_path).name}:"
                f" checks=[{checks_str}]"
                f" resource_accesses=[{accesses_str}]"
                f" snippet={af.source_snippet[:200]}"
            )
        # Unprotected access warnings.
        for func_name in self.unprotected_resource_accesses:
            docs.append(
                f"WARNING: Function '{func_name}' accesses a resource without "
                f"detectable ownership check — potential IDOR (CWE-639)."
            )
        return docs


class AuthExtractor:
    """Extracts the authentication and authorisation model from a Python codebase.

    Uses Python's ``ast`` module to parse source files without executing them.
    Identifies:
    - Functions that perform auth checks.
    - Functions that access resources (ORM queries).
    - Functions that access resources WITHOUT auth checks (IDOR candidates).
    - Auth decorators and the endpoints they protect.

    Args:
        project_path: Path to the Python project root.
    """

    def __init__(self, project_path: str) -> None:
        if not project_path:
            raise ValueError("project_path must be a non-empty string.")
        self._project_path = Path(project_path)
        if not self._project_path.exists():
            raise FileNotFoundError(
                f"Project path does not exist: {project_path}"
            )

    def extract(self) -> AuthModel:
        """Analyse the codebase and return a structured AuthModel.

        Returns:
            AuthModel: Extracted access-control model.
        """
        model = AuthModel(project_path=str(self._project_path))
        python_files = list(self._project_path.rglob("*.py"))
        logger.info(
            "AuthExtractor: scanning %d Python files in %s",
            len(python_files),
            self._project_path,
        )
        for py_file in python_files:
            self._analyse_file(py_file, model)
        # Detect unprotected resource accesses.
        auth_func_names = {af.name for af in model.auth_functions}
        for af in model.auth_functions:
            if af.resource_accesses and not af.checks:
                # Resource access with no auth check.
                model.unprotected_resource_accesses.append(af.name)
        logger.info(
            "AuthExtractor complete: %d auth functions, %d unprotected accesses",
            len(model.auth_functions),
            len(model.unprotected_resource_accesses),
        )
        return model

    def _analyse_file(self, py_file: Path, model: AuthModel) -> None:
        """Analyse a single Python file and update the auth model.

        Args:
            py_file: Path to the Python source file.
            model: AuthModel to update in-place.
        """
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError) as exc:
            logger.debug("Skipping %s: %s", py_file, exc)
            return

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            auth_func = self._analyse_function(node, source, str(py_file))
            if auth_func:
                model.auth_functions.append(auth_func)
            # Check for auth decorators.
            for decorator in node.decorator_list:
                dec_name = self._get_decorator_name(decorator)
                if dec_name and any(kw in dec_name.lower() for kw in _AUTH_IDENTIFIERS):
                    if dec_name not in model.decorators:
                        model.decorators.append(dec_name)
                    model.protected_endpoints.append(node.name)

        # Detect middleware classes.
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_lower = node.name.lower()
                if any(kw in class_lower for kw in ("auth", "middleware", "permission")):
                    model.auth_middleware.append(
                        f"{py_file.stem}.{node.name}"
                    )

    def _analyse_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source: str,
        file_path: str,
    ) -> Optional[AuthFunction]:
        """Analyse a single function node for auth and resource access patterns.

        Args:
            node: AST function node.
            source: Full source of the file.
            file_path: Path to the file (for metadata).

        Returns:
            AuthFunction | None: Populated AuthFunction, or None if not
                auth-related and has no resource accesses.
        """
        func_source = ast.get_source_segment(source, node) or ""
        func_lower = func_source.lower()
        checks_found: list[str] = []
        accesses_found: list[str] = []

        # Walk the function AST for identifier usage.
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if child.id in _AUTH_IDENTIFIERS:
                    if child.id not in checks_found:
                        checks_found.append(child.id)
            elif isinstance(child, ast.Attribute):
                attr_key = f"{self._get_attr_name(child)}"
                for pattern in _RESOURCE_ACCESS_PATTERNS:
                    if pattern in attr_key:
                        if pattern not in accesses_found:
                            accesses_found.append(pattern)

        if not checks_found and not accesses_found:
            return None

        is_decorator = any(
            isinstance(dec, (ast.Name, ast.Attribute))
            for dec in getattr(node, "decorator_list", [])
        )
        return AuthFunction(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            checks=checks_found,
            resource_accesses=accesses_found,
            is_decorator=is_decorator,
            source_snippet=func_source[:400],
        )

    @staticmethod
    def _get_decorator_name(decorator: ast.expr) -> str | None:
        """Extract the name string from a decorator AST node.

        Args:
            decorator: AST decorator node.

        Returns:
            str | None: Decorator name, or None if not extractable.
        """
        if isinstance(decorator, ast.Name):
            return decorator.id
        if isinstance(decorator, ast.Attribute):
            return decorator.attr
        if isinstance(decorator, ast.Call):
            return AuthExtractor._get_decorator_name(decorator.func)
        return None

    @staticmethod
    def _get_attr_name(node: ast.Attribute) -> str:
        """Reconstruct dotted attribute name from an AST Attribute node.

        Args:
            node: AST Attribute node.

        Returns:
            str: Dotted name string.
        """
        parts: list[str] = [node.attr]
        current = node.value
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
