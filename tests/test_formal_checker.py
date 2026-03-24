"""
Tests for FormalChecker, PropertyTemplateEngine, and STIXToViperGenerator.

Pure unit tests — no network, no external tools.

Run: pytest tests/test_formal_checker.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from verifier.formal_check import FormalChecker
from verifier.property_templates import PropertyTemplateEngine, PROPERTY_TEMPLATES
from verifier.stix_to_viper import STIXToViperGenerator
from core.threat_indicator import ThreatIndicator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_indicator(**overrides) -> ThreatIndicator:
    defaults = dict(
        id="CVE-2024-99999",
        source="NVD",
        severity="high",
        cwe=["CWE-89"],
        affected_patterns=["execute(sql_string)"],
        stix_pattern=None,
        activation_signature={},
        semantic_rule="SQLi in query builder",
        formal_property="Requires(is_parameterized(query))",
        timestamp=datetime.now(tz=timezone.utc),
        ttl=86400,
        description="SQL injection via f-string.",
        cvss_score=8.5,
        affected_packages=[],
    )
    defaults.update(overrides)
    return ThreatIndicator(**defaults)


# ---------------------------------------------------------------------------
# FormalChecker — init and basic
# ---------------------------------------------------------------------------

class TestFormalCheckerInit:

    def test_default_init(self):
        checker = FormalChecker()
        assert checker._invoke_threshold == 0.7
        assert checker._use_nagini is False

    def test_custom_threshold(self):
        checker = FormalChecker(invoke_threshold=0.5)
        assert checker._invoke_threshold == 0.5

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            FormalChecker(invoke_threshold=1.5)

    def test_verify_empty_code_raises(self):
        checker = FormalChecker()
        with pytest.raises(ValueError, match="code_snippet"):
            checker.verify("", "IDOR")

    def test_verify_empty_vuln_class_raises(self):
        checker = FormalChecker()
        with pytest.raises(ValueError, match="vuln_class"):
            checker.verify("def foo(): pass", "")

    def test_repr(self):
        checker = FormalChecker(invoke_threshold=0.6)
        assert "0.6" in repr(checker)


# ---------------------------------------------------------------------------
# FormalChecker — IDOR checks
# ---------------------------------------------------------------------------

class TestFormalCheckerIDOR:

    def setup_method(self):
        self.checker = FormalChecker()

    def test_idor_vulnerable_no_ownership_check(self):
        code = """
def get_invoice(invoice_id):
    return Invoice.objects.get(id=invoice_id)
"""
        result = self.checker.verify(code, "IDOR")
        assert result["result"] == "VIOLATION"

    def test_idor_safe_with_ownership_check(self):
        code = """
def get_invoice(invoice_id, user):
    invoice = Invoice.objects.get(id=invoice_id)
    if invoice.owner != user:
        raise PermissionDenied
    return invoice
"""
        result = self.checker.verify(code, "IDOR")
        assert result["result"] == "VERIFIED"

    def test_idor_safe_with_abort(self):
        code = """
def get_resource(resource_id, user):
    resource = Resource.objects.get(pk=resource_id)
    if not user.can_access(resource):
        abort(403)
    return resource
"""
        result = self.checker.verify(code, "IDOR")
        assert result["result"] == "VERIFIED"

    def test_idor_safe_no_resource_access(self):
        code = """
def greet(name):
    return f"Hello, {name}"
"""
        result = self.checker.verify(code, "IDOR")
        assert result["result"] == "VERIFIED"

    def test_idor_result_has_required_keys(self):
        code = "def foo(): pass"
        result = self.checker.verify(code, "IDOR")
        assert set(result.keys()) >= {"result", "property", "evidence", "confidence", "method"}

    def test_idor_confidence_in_range(self):
        code = "def get_item(id): return Item.objects.get(id=id)"
        result = self.checker.verify(code, "IDOR")
        assert 0.0 <= result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# FormalChecker — SQLi checks
# ---------------------------------------------------------------------------

class TestFormalCheckerSQLi:

    def setup_method(self):
        self.checker = FormalChecker()

    def test_sqli_vulnerable_string_concat(self):
        code = """
def get_user(username):
    query = "SELECT * FROM users WHERE name='" + username + "'"
    cursor.execute(query)
"""
        result = self.checker.verify(code, "SQLi")
        assert result["result"] == "VIOLATION"

    def test_sqli_vulnerable_fstring(self):
        code = """
def get_user(name):
    cursor.execute(f"SELECT * FROM users WHERE name='{name}'")
"""
        result = self.checker.verify(code, "SQLi")
        assert result["result"] == "VIOLATION"

    def test_sqli_safe_parameterised(self):
        code = """
def get_user(username):
    cursor.execute('SELECT * FROM users WHERE name = %s', (username,))
    return cursor.fetchone()
"""
        result = self.checker.verify(code, "SQLi")
        assert result["result"] == "VERIFIED"

    def test_sqli_method_is_ast(self):
        code = "def foo(): cursor.execute('SELECT 1')"
        result = self.checker.verify(code, "SQLi")
        assert "AST" in result["method"]


# ---------------------------------------------------------------------------
# FormalChecker — SSRF checks
# ---------------------------------------------------------------------------

class TestFormalCheckerSSRF:

    def setup_method(self):
        self.checker = FormalChecker()

    def test_ssrf_vulnerable_plain_get(self):
        code = """
def fetch(url):
    return requests.get(url).text
"""
        result = self.checker.verify(code, "SSRF")
        assert result["result"] == "VIOLATION"

    def test_ssrf_safe_with_allowlist(self):
        code = """
ALLOWED = ['https://api.example.com']
def fetch(url):
    if url not in ALLOWED:
        raise ValueError
    return requests.get(url).text
"""
        result = self.checker.verify(code, "SSRF")
        assert result["result"] == "VERIFIED"

    def test_ssrf_safe_no_url_fetch(self):
        code = """
def add(a, b):
    return a + b
"""
        result = self.checker.verify(code, "SSRF")
        assert result["result"] == "VERIFIED"


# ---------------------------------------------------------------------------
# FormalChecker — path traversal checks
# ---------------------------------------------------------------------------

class TestFormalCheckerPathTraversal:

    def setup_method(self):
        self.checker = FormalChecker()

    def test_path_traversal_vulnerable(self):
        code = """
def read_file(filename):
    with open('/var/data/' + filename) as f:
        return f.read()
"""
        result = self.checker.verify(code, "path_traversal")
        assert result["result"] == "VIOLATION"

    def test_path_traversal_safe_realpath(self):
        code = """
import os
BASE = '/var/data'
def read_file(filename):
    safe = os.path.realpath(os.path.join(BASE, filename))
    if not safe.startswith(BASE):
        raise ValueError
    with open(safe) as f:
        return f.read()
"""
        result = self.checker.verify(code, "path_traversal")
        assert result["result"] == "VERIFIED"

    def test_path_traversal_no_file_op(self):
        code = "def greet(): return 'hello'"
        result = self.checker.verify(code, "path_traversal")
        assert result["result"] == "VERIFIED"


# ---------------------------------------------------------------------------
# FormalChecker — auth bypass checks
# ---------------------------------------------------------------------------

class TestFormalCheckerAuthBypass:

    def setup_method(self):
        self.checker = FormalChecker()

    def test_auth_bypass_query_param(self):
        code = """
def admin_view(request):
    if request.args.get('is_admin') == '1':
        return render_admin()
"""
        result = self.checker.verify(code, "auth_bypass")
        assert result["result"] == "VIOLATION"

    def test_auth_bypass_safe_session_check(self):
        code = """
def admin_view(request):
    if not request.user.is_authenticated:
        abort(403)
    return render_admin()
"""
        result = self.checker.verify(code, "auth_bypass")
        assert result["result"] == "VERIFIED"

    def test_auth_bypass_safe_decorator(self):
        code = """
@login_required
def protected_view(request):
    return render_template('secret.html')
"""
        result = self.checker.verify(code, "auth_bypass")
        assert result["result"] == "VERIFIED"


# ---------------------------------------------------------------------------
# FormalChecker — deserialization checks
# ---------------------------------------------------------------------------

class TestFormalCheckerDeserialization:

    def setup_method(self):
        self.checker = FormalChecker()

    def test_unsafe_pickle(self):
        code = """
import pickle
def load_obj(data):
    return pickle.loads(data)
"""
        result = self.checker.verify(code, "deserialization")
        assert result["result"] == "VIOLATION"
        assert "pickle" in result["evidence"]

    def test_unsafe_yaml_no_loader(self):
        code = """
import yaml
def load_config(stream):
    return yaml.load(stream)
"""
        result = self.checker.verify(code, "deserialization")
        assert result["result"] == "VIOLATION"

    def test_safe_json(self):
        code = """
import json
def load_data(text):
    return json.loads(text)
"""
        result = self.checker.verify(code, "deserialization")
        assert result["result"] == "VERIFIED"

    def test_safe_yaml_with_safeloader(self):
        code = """
import yaml
def load_config(stream):
    return yaml.load(stream, Loader=yaml.SafeLoader)
"""
        result = self.checker.verify(code, "deserialization")
        assert result["result"] == "VERIFIED"


# ---------------------------------------------------------------------------
# FormalChecker — command injection checks
# ---------------------------------------------------------------------------

class TestFormalCheckerCommandInjection:

    def setup_method(self):
        self.checker = FormalChecker()

    def test_command_injection_shell_true(self):
        code = """
import subprocess
def run(cmd):
    subprocess.call(cmd, shell=True)
"""
        result = self.checker.verify(code, "command_injection")
        assert result["result"] == "VIOLATION"

    def test_command_injection_safe_no_shell(self):
        code = """
import subprocess
def run(cmd_list):
    subprocess.run(cmd_list)
"""
        result = self.checker.verify(code, "command_injection")
        assert result["result"] == "VERIFIED"


# ---------------------------------------------------------------------------
# FormalChecker — unknown vuln class
# ---------------------------------------------------------------------------

class TestFormalCheckerUnknownClass:

    def test_unknown_vuln_class_returns_unknown(self):
        checker = FormalChecker()
        result = checker.verify("def foo(): pass", "unknown_vuln_class_xyz")
        assert result["result"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# FormalChecker — syntax error
# ---------------------------------------------------------------------------

class TestFormalCheckerSyntaxError:

    def test_syntax_error_returns_unknown(self):
        checker = FormalChecker()
        result = checker.verify("def foo(: pass", "IDOR")
        assert result["result"] == "UNKNOWN"
        assert "Syntax error" in result["evidence"]


# ---------------------------------------------------------------------------
# PropertyTemplateEngine
# ---------------------------------------------------------------------------

class TestPropertyTemplateEngine:

    def test_get_template_idor(self):
        engine = PropertyTemplateEngine()
        tmpl = engine.get_template("IDOR")
        assert "CWE-639" in tmpl["cwe"]
        assert "ownership" in tmpl["nagini_annotation"].lower()

    def test_get_template_sqli(self):
        engine = PropertyTemplateEngine()
        tmpl = engine.get_template("SQLi")
        assert "CWE-89" in tmpl["cwe"]

    def test_get_template_unknown_raises(self):
        engine = PropertyTemplateEngine()
        with pytest.raises(KeyError):
            engine.get_template("NonExistentVulnClass")

    def test_list_classes_includes_all(self):
        engine = PropertyTemplateEngine()
        classes = engine.list_classes()
        for expected in ("IDOR", "SQLi", "SSRF", "path_traversal", "auth_bypass"):
            assert expected in classes

    def test_get_nagini_annotation(self):
        engine = PropertyTemplateEngine()
        ann = engine.get_nagini_annotation("IDOR")
        assert "Requires" in ann

    def test_get_viper_spec(self):
        engine = PropertyTemplateEngine()
        spec = engine.get_viper_spec("SSRF")
        assert "requires" in spec.lower()

    def test_len(self):
        engine = PropertyTemplateEngine()
        assert len(engine) == len(PROPERTY_TEMPLATES)

    def test_custom_templates_merged(self):
        custom = {"my_vuln": {"description": "Custom", "nagini_annotation": "Requires(x)", "viper_spec": "requires x", "check_function": "check_x", "cwe": "CWE-999", "owasp": "N/A"}}
        engine = PropertyTemplateEngine(custom_templates=custom)
        assert "my_vuln" in engine.list_classes()

    def test_get_check_function_name(self):
        engine = PropertyTemplateEngine()
        fn = engine.get_check_function_name("IDOR")
        assert fn == "has_ownership_check"

    def test_get_check_function_unknown(self):
        engine = PropertyTemplateEngine()
        fn = engine.get_check_function_name("nonexistent")
        assert fn is None

    def test_repr(self):
        engine = PropertyTemplateEngine()
        r = repr(engine)
        assert "IDOR" in r

    def test_all_templates_have_required_fields(self):
        required = {"description", "nagini_annotation", "viper_spec", "check_function", "cwe", "owasp"}
        for cls, tmpl in PROPERTY_TEMPLATES.items():
            missing = required - set(tmpl.keys())
            assert not missing, f"Template '{cls}' missing fields: {missing}"


# ---------------------------------------------------------------------------
# STIXToViperGenerator
# ---------------------------------------------------------------------------

class TestSTIXToViperGenerator:

    def test_generate_from_indicator(self):
        gen = STIXToViperGenerator()
        ind = make_indicator(cwe=["CWE-89", "CWE-639"])
        spec = gen.generate(ind)
        assert "CVE-2024-99999" in spec
        assert "CWE-89" in spec
        assert "CWE-639" in spec
        assert "requires" in spec.lower()

    def test_generate_stores_spec(self):
        gen = STIXToViperGenerator()
        ind = make_indicator()
        gen.generate(ind)
        assert "CVE-2024-99999" in gen._generated_specs

    def test_generate_type_check(self):
        gen = STIXToViperGenerator()
        with pytest.raises(TypeError):
            gen.generate("not an indicator")  # type: ignore[arg-type]

    def test_map_cwe_to_property_known(self):
        gen = STIXToViperGenerator()
        prop = gen._map_cwe_to_property("CWE-89")
        assert "parameterized" in prop

    def test_map_cwe_to_property_unknown(self):
        gen = STIXToViperGenerator()
        prop = gen._map_cwe_to_property("CWE-9999")
        assert "input_validated" in prop

    def test_generate_precondition_sqli(self):
        gen = STIXToViperGenerator()
        pre = gen._generate_precondition("SQLi")
        assert "parameterized" in pre

    def test_generate_precondition_ssrf(self):
        gen = STIXToViperGenerator()
        pre = gen._generate_precondition("SSRF")
        assert "allowlist" in pre.lower()

    def test_generate_postcondition_idor(self):
        gen = STIXToViperGenerator()
        post = gen._generate_postcondition("IDOR")
        assert "owner" in post.lower() or "authorized" in post.lower()

    def test_generate_no_cwe_graceful(self):
        gen = STIXToViperGenerator()
        ind = make_indicator(cwe=[])
        spec = gen.generate(ind)
        assert isinstance(spec, str)
        assert len(spec) > 0

    def test_cwe_description_known(self):
        assert "SQL Injection" in STIXToViperGenerator._cwe_description("CWE-89")

    def test_cwe_description_unknown(self):
        desc = STIXToViperGenerator._cwe_description("CWE-9999")
        assert "CWE-9999" in desc

    def test_save_spec(self, tmp_path):
        gen = STIXToViperGenerator()
        ind = make_indicator()
        spec = gen.generate(ind)
        out_path = str(tmp_path / "test.vpr")
        gen.save_spec(spec, out_path)
        import os
        assert os.path.exists(out_path)
        with open(out_path) as f:
            content = f.read()
        assert "CVE-2024-99999" in content

    def test_save_all(self, tmp_path):
        gen = STIXToViperGenerator()
        gen.generate(make_indicator(id="CVE-2024-00001"))
        gen.generate(make_indicator(id="CVE-2024-00002"))
        written = gen.save_all(str(tmp_path))
        assert len(written) == 2

    def test_repr(self):
        gen = STIXToViperGenerator()
        assert "STIXToViperGenerator" in repr(gen)
