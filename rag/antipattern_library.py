"""
Anti-Pattern Library — pre-seeded CVE/OWASP-sourced vulnerability patterns.

Each AntiPattern record captures a vulnerability class with a concrete
vulnerable code example, its safe counterpart, and metadata linking it to
CWE, OWASP, or a specific CVE.  The library is pre-seeded with 25 real
patterns across 5 vulnerability classes (5 per class).

Research rationale:
    The anti-pattern library is the *static knowledge base* of Layer 2.
    Unlike live threat intel (which changes), these patterns encode
    time-stable vulnerability knowledge derived from authoritative sources
    (CWE, OWASP Top 10, HackerOne disclosures).  The library can be
    exported to JSONL for use as training data for Layer 1 probes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AntiPattern:
    """A single vulnerability anti-pattern record.

    Attributes:
        pattern_id: Unique identifier (e.g. "IDOR-001", "SQLi-003").
        description: One-sentence description of the vulnerability.
        anti_pattern: The problematic code construct to search for.
        safe_alternative: The correct/safe replacement.
        severity: "critical", "high", "medium", or "low".
        cwe: CWE identifier (e.g. "CWE-639").
        source: Authoritative source — "NVD", "OWASP", "HackerOne", "CWE".
        stix_id: Optional STIX 2.1 indicator ID if this pattern was encoded
            from a STIX feed.
        example_vulnerable: Short vulnerable code example (Python).
        example_safe: Short safe code example (Python).
    """

    pattern_id: str
    description: str
    anti_pattern: str
    safe_alternative: str
    severity: str
    cwe: str
    source: str
    stix_id: Optional[str] = None
    example_vulnerable: str = ""
    example_safe: str = ""


# ---------------------------------------------------------------------------
# Pre-seeded patterns — 5 per class
# ---------------------------------------------------------------------------

IDOR_PATTERNS: list[AntiPattern] = [
    AntiPattern(
        pattern_id="IDOR-001",
        description="Direct ORM lookup by user-supplied ID without ownership check (CWE-639).",
        anti_pattern="Model.objects.get(id=request_param)",
        safe_alternative="Model.objects.get(id=request_param, owner=request.user)",
        severity="high",
        cwe="CWE-639",
        source="OWASP",
        example_vulnerable=(
            "def get_invoice(request, invoice_id):\n"
            "    return Invoice.objects.get(id=invoice_id)"
        ),
        example_safe=(
            "def get_invoice(request, invoice_id):\n"
            "    invoice = Invoice.objects.get(id=invoice_id)\n"
            "    if invoice.owner != request.user:\n"
            "        raise PermissionDenied('Access denied')\n"
            "    return invoice"
        ),
    ),
    AntiPattern(
        pattern_id="IDOR-002",
        description="Sequential integer ID exposed in URL allows enumeration of resources.",
        anti_pattern="GET /api/orders/{integer_id} — no ownership check",
        safe_alternative="Use UUID resource identifiers AND server-side ownership verification.",
        severity="high",
        cwe="CWE-639",
        source="HackerOne",
        example_vulnerable=(
            "@app.route('/orders/<int:order_id>')\n"
            "def order_detail(order_id):\n"
            "    return Order.query.get(order_id)"
        ),
        example_safe=(
            "@app.route('/orders/<uuid:order_uuid>')\n"
            "@login_required\n"
            "def order_detail(order_uuid):\n"
            "    order = Order.query.filter_by(uuid=order_uuid).first_or_404()\n"
            "    if order.user_id != current_user.id:\n"
            "        abort(403)\n"
            "    return order"
        ),
    ),
    AntiPattern(
        pattern_id="IDOR-003",
        description="File download endpoint returns files by filename without user-scope check.",
        anti_pattern="open(UPLOAD_DIR + request.args['filename'])",
        safe_alternative="Verify that the file record belongs to the requesting user before serving.",
        severity="high",
        cwe="CWE-639",
        source="OWASP",
        example_vulnerable=(
            "def download(filename):\n"
            "    path = os.path.join(UPLOAD_DIR, filename)\n"
            "    return send_file(path)"
        ),
        example_safe=(
            "def download(filename):\n"
            "    record = FileRecord.query.filter_by(\n"
            "        filename=filename, owner_id=current_user.id\n"
            "    ).first_or_404()\n"
            "    return send_file(record.full_path)"
        ),
    ),
    AntiPattern(
        pattern_id="IDOR-004",
        description="API endpoint updates a resource using client-supplied ID without scoping to session.",
        anti_pattern="UPDATE SET ... WHERE id = user_supplied_id",
        safe_alternative="Include session user ID in the WHERE clause: WHERE id = ? AND user_id = ?",
        severity="high",
        cwe="CWE-639",
        source="CWE",
        example_vulnerable=(
            "def update_profile(user_id, data):\n"
            "    cursor.execute('UPDATE users SET email=%s WHERE id=%s',\n"
            "                   (data['email'], user_id))"
        ),
        example_safe=(
            "def update_profile(user_id, data, requesting_user_id):\n"
            "    cursor.execute(\n"
            "        'UPDATE users SET email=%s WHERE id=%s AND id=%s',\n"
            "        (data['email'], user_id, requesting_user_id)\n"
            "    )"
        ),
    ),
    AntiPattern(
        pattern_id="IDOR-005",
        description="GraphQL resolver returns node by global ID without authorization guard.",
        anti_pattern="node(id: ID!) without ownership policy check",
        safe_alternative="Implement a NodeAuthorizationMixin that checks field-level permissions.",
        severity="medium",
        cwe="CWE-639",
        source="HackerOne",
        example_vulnerable=(
            "class Query(graphene.ObjectType):\n"
            "    node = graphene.relay.Node.Field()"
        ),
        example_safe=(
            "class AuthorizedNode(graphene.relay.Node):\n"
            "    @staticmethod\n"
            "    def get_node_from_global_id(info, global_id, only_type=None):\n"
            "        node = super().get_node_from_global_id(info, global_id, only_type)\n"
            "        if not info.context.user.can_access(node):\n"
            "            raise GraphQLError('Forbidden')\n"
            "        return node"
        ),
    ),
]

SQLI_PATTERNS: list[AntiPattern] = [
    AntiPattern(
        pattern_id="SQLi-001",
        description="String concatenation to build SQL query — classic SQL injection (CWE-89).",
        anti_pattern="'SELECT * FROM users WHERE name = \\'' + username + '\\''",
        safe_alternative="Parameterised query: cursor.execute('SELECT * FROM users WHERE name = %s', (username,))",
        severity="critical",
        cwe="CWE-89",
        source="OWASP",
        example_vulnerable=(
            "def get_user(username):\n"
            "    query = \"SELECT * FROM users WHERE name='\" + username + \"'\"\n"
            "    cursor.execute(query)\n"
            "    return cursor.fetchone()"
        ),
        example_safe=(
            "def get_user(username):\n"
            "    cursor.execute(\n"
            "        'SELECT * FROM users WHERE name = %s', (username,)\n"
            "    )\n"
            "    return cursor.fetchone()"
        ),
    ),
    AntiPattern(
        pattern_id="SQLi-002",
        description="Python f-string used to interpolate user input into SQL query.",
        anti_pattern="f\"SELECT * FROM {table} WHERE id={user_id}\"",
        safe_alternative="Use SQLAlchemy's text() with bindparams or a parameterised driver.",
        severity="critical",
        cwe="CWE-89",
        source="NVD",
        example_vulnerable=(
            "def search(table, field, value):\n"
            "    query = f\"SELECT * FROM {table} WHERE {field}='{value}'\"\n"
            "    return db.execute(query)"
        ),
        example_safe=(
            "from sqlalchemy import text\n"
            "def search(field, value):\n"
            "    stmt = text('SELECT * FROM items WHERE :field = :value')\n"
            "    return db.execute(stmt, {'field': field, 'value': value})"
        ),
    ),
    AntiPattern(
        pattern_id="SQLi-003",
        description="Django raw() query with unsanitised user input.",
        anti_pattern="Model.objects.raw(f'SELECT * FROM app_model WHERE name={name}')",
        safe_alternative="Model.objects.raw('SELECT * FROM app_model WHERE name=%s', [name])",
        severity="critical",
        cwe="CWE-89",
        source="OWASP",
        example_vulnerable=(
            "def search_users(name):\n"
            "    return User.objects.raw(\n"
            "        f\"SELECT * FROM auth_user WHERE username='{name}'\"\n"
            "    )"
        ),
        example_safe=(
            "def search_users(name):\n"
            "    return User.objects.raw(\n"
            "        'SELECT * FROM auth_user WHERE username=%s', [name]\n"
            "    )"
        ),
    ),
    AntiPattern(
        pattern_id="SQLi-004",
        description="LIKE query with user-controlled pattern — SQLi via wildcard injection.",
        anti_pattern="'SELECT * FROM products WHERE name LIKE %' + search_term + '%'",
        safe_alternative="Use parameterised LIKE: cursor.execute('... LIKE %s', ('%' + search_term + '%',))",
        severity="high",
        cwe="CWE-89",
        source="CWE",
        example_vulnerable=(
            "def search_products(term):\n"
            "    query = \"SELECT * FROM products WHERE name LIKE '%\" + term + \"%'\"\n"
            "    cursor.execute(query)"
        ),
        example_safe=(
            "def search_products(term):\n"
            "    cursor.execute(\n"
            "        'SELECT * FROM products WHERE name LIKE %s',\n"
            "        ('%' + term + '%',)\n"
            "    )"
        ),
    ),
    AntiPattern(
        pattern_id="SQLi-005",
        description="ORDER BY clause constructed from user input — second-order SQLi.",
        anti_pattern="f'SELECT * FROM orders ORDER BY {sort_field} {sort_dir}'",
        safe_alternative="Validate sort_field against a whitelist; never interpolate column names.",
        severity="high",
        cwe="CWE-89",
        source="HackerOne",
        example_vulnerable=(
            "def list_orders(sort_field, sort_dir):\n"
            "    query = f'SELECT * FROM orders ORDER BY {sort_field} {sort_dir}'\n"
            "    return db.execute(query)"
        ),
        example_safe=(
            "ALLOWED_SORT_FIELDS = {'created_at', 'total', 'status'}\n"
            "ALLOWED_DIRECTIONS = {'ASC', 'DESC'}\n"
            "def list_orders(sort_field, sort_dir):\n"
            "    if sort_field not in ALLOWED_SORT_FIELDS:\n"
            "        raise ValueError('Invalid sort field')\n"
            "    if sort_dir.upper() not in ALLOWED_DIRECTIONS:\n"
            "        raise ValueError('Invalid sort direction')\n"
            "    query = f'SELECT * FROM orders ORDER BY {sort_field} {sort_dir}'\n"
            "    return db.execute(query)"
        ),
    ),
]

SSRF_PATTERNS: list[AntiPattern] = [
    AntiPattern(
        pattern_id="SSRF-001",
        description="HTTP fetch using user-supplied URL without allowlist validation (CWE-918).",
        anti_pattern="requests.get(user_url)",
        safe_alternative="Validate URL against an allowlist before fetching.",
        severity="high",
        cwe="CWE-918",
        source="OWASP",
        example_vulnerable=(
            "def fetch_url(url):\n"
            "    response = requests.get(url)\n"
            "    return response.text"
        ),
        example_safe=(
            "ALLOWED_DOMAINS = ['api.example.com', 'cdn.example.com']\n"
            "from urllib.parse import urlparse\n"
            "def fetch_url(url):\n"
            "    parsed = urlparse(url)\n"
            "    if parsed.hostname not in ALLOWED_DOMAINS:\n"
            "        raise ValueError('Domain not in allowlist')\n"
            "    response = requests.get(url, timeout=10)\n"
            "    return response.text"
        ),
    ),
    AntiPattern(
        pattern_id="SSRF-002",
        description="Webhook endpoint fetches user-supplied callback URL — SSRF to internal services.",
        anti_pattern="httpx.post(request.json['callback_url'], data=payload)",
        safe_alternative="Validate callback URL scheme, host, and port; block RFC-1918 ranges.",
        severity="high",
        cwe="CWE-918",
        source="HackerOne",
        example_vulnerable=(
            "def trigger_webhook(request):\n"
            "    callback = request.json['callback_url']\n"
            "    httpx.post(callback, json={'event': 'completed'})"
        ),
        example_safe=(
            "import ipaddress\n"
            "from urllib.parse import urlparse\n"
            "def is_safe_url(url):\n"
            "    parsed = urlparse(url)\n"
            "    if parsed.scheme not in ('http', 'https'):\n"
            "        return False\n"
            "    try:\n"
            "        ip = ipaddress.ip_address(parsed.hostname)\n"
            "        if ip.is_private or ip.is_loopback:\n"
            "            return False\n"
            "    except ValueError:\n"
            "        pass  # hostname, not IP — check allowlist\n"
            "    return True"
        ),
    ),
    AntiPattern(
        pattern_id="SSRF-003",
        description="Image proxy endpoint fetches arbitrary URLs — SSRF via image upload.",
        anti_pattern="urllib.request.urlopen(image_url).read()",
        safe_alternative="Proxy only pre-signed URLs from known CDN domains; reject all others.",
        severity="high",
        cwe="CWE-918",
        source="NVD",
        example_vulnerable=(
            "def proxy_image(image_url):\n"
            "    data = urllib.request.urlopen(image_url).read()\n"
            "    return Response(data, mimetype='image/jpeg')"
        ),
        example_safe=(
            "CDN_PATTERN = re.compile(r'^https://cdn\\.example\\.com/')\n"
            "def proxy_image(image_url):\n"
            "    if not CDN_PATTERN.match(image_url):\n"
            "        abort(400, 'Image URL not from approved CDN')\n"
            "    data = requests.get(image_url, timeout=5).content\n"
            "    return Response(data, mimetype='image/jpeg')"
        ),
    ),
    AntiPattern(
        pattern_id="SSRF-004",
        description="XML external entity (XXE) via DTD references — a form of SSRF (CWE-611).",
        anti_pattern="etree.parse(user_xml_file)",
        safe_alternative="Disable external entities: use defusedxml or explicitly set parser options.",
        severity="high",
        cwe="CWE-918",
        source="OWASP",
        example_vulnerable=(
            "from lxml import etree\n"
            "def parse_config(xml_data):\n"
            "    tree = etree.fromstring(xml_data)\n"
            "    return tree"
        ),
        example_safe=(
            "import defusedxml.ElementTree as ET\n"
            "def parse_config(xml_data):\n"
            "    # defusedxml blocks XXE, billion laughs, and quadratic blowup\n"
            "    tree = ET.fromstring(xml_data)\n"
            "    return tree"
        ),
    ),
    AntiPattern(
        pattern_id="SSRF-005",
        description="DNS rebinding: SSRF bypass via DNS resolution timing.",
        anti_pattern="requests.get(url) after only resolving hostname once",
        safe_alternative="Resolve hostname to IP, validate IP is not RFC-1918, then connect to IP directly.",
        severity="medium",
        cwe="CWE-918",
        source="HackerOne",
        example_vulnerable=(
            "# Vulnerable to DNS rebinding — hostname resolves to public IP at\n"
            "# check time, then rebinds to 192.168.x.x at request time.\n"
            "def safe_fetch(url):\n"
            "    host = urlparse(url).hostname\n"
            "    if is_private_ip(socket.gethostbyname(host)):\n"
            "        raise ValueError('Private IP')\n"
            "    return requests.get(url)  # Second DNS resolution!"
        ),
        example_safe=(
            "import socket\n"
            "import ipaddress\n"
            "def safe_fetch(url):\n"
            "    parsed = urlparse(url)\n"
            "    ip_str = socket.gethostbyname(parsed.hostname)\n"
            "    ip = ipaddress.ip_address(ip_str)\n"
            "    if ip.is_private or ip.is_loopback:\n"
            "        raise ValueError('Disallowed IP range')\n"
            "    # Connect directly to resolved IP to prevent rebinding.\n"
            "    direct_url = url.replace(parsed.hostname, ip_str)\n"
            "    return requests.get(direct_url, headers={'Host': parsed.hostname})"
        ),
    ),
]

AUTH_BYPASS_PATTERNS: list[AntiPattern] = [
    AntiPattern(
        pattern_id="AUTH-001",
        description="Admin check via URL parameter rather than session role (CWE-287).",
        anti_pattern="if request.args.get('admin') == 'true': render_admin()",
        safe_alternative="Check server-side session: if request.user.is_staff: render_admin()",
        severity="critical",
        cwe="CWE-287",
        source="OWASP",
        example_vulnerable=(
            "def admin_view(request):\n"
            "    if request.args.get('is_admin') == '1':\n"
            "        return render_template('admin.html')\n"
            "    return abort(403)"
        ),
        example_safe=(
            "from functools import wraps\n"
            "def admin_required(f):\n"
            "    @wraps(f)\n"
            "    def decorated(*args, **kwargs):\n"
            "        if not current_user.is_authenticated or not current_user.is_admin:\n"
            "            abort(403)\n"
            "        return f(*args, **kwargs)\n"
            "    return decorated"
        ),
    ),
    AntiPattern(
        pattern_id="AUTH-002",
        description="JWT signature not verified — algorithm confusion attack (CVE-2015-9235).",
        anti_pattern="jwt.decode(token, options={'verify_signature': False})",
        safe_alternative="Always verify signature with explicit algorithm list.",
        severity="critical",
        cwe="CWE-347",
        source="NVD",
        example_vulnerable=(
            "def get_current_user(token):\n"
            "    payload = jwt.decode(\n"
            "        token, options={'verify_signature': False}\n"
            "    )\n"
            "    return User.get(payload['user_id'])"
        ),
        example_safe=(
            "SECRET = os.environ['JWT_SECRET']\n"
            "def get_current_user(token):\n"
            "    payload = jwt.decode(\n"
            "        token,\n"
            "        SECRET,\n"
            "        algorithms=['HS256'],  # Explicit allowlist\n"
            "    )\n"
            "    return User.get(payload['user_id'])"
        ),
    ),
    AntiPattern(
        pattern_id="AUTH-003",
        description="Password reset token not invalidated after use — replay attack.",
        anti_pattern="user.reset_token == provided_token (without marking token as used)",
        safe_alternative="Mark token as used/expired immediately on successful password reset.",
        severity="high",
        cwe="CWE-287",
        source="HackerOne",
        example_vulnerable=(
            "def reset_password(token, new_password):\n"
            "    user = User.query.filter_by(reset_token=token).first()\n"
            "    if user:\n"
            "        user.password = hash_password(new_password)\n"
            "        db.session.commit()"
        ),
        example_safe=(
            "def reset_password(token, new_password):\n"
            "    user = User.query.filter_by(\n"
            "        reset_token=token,\n"
            "        token_expires_at > datetime.utcnow()\n"
            "    ).first()\n"
            "    if user:\n"
            "        user.password = hash_password(new_password)\n"
            "        user.reset_token = None  # Invalidate immediately\n"
            "        user.token_expires_at = None\n"
            "        db.session.commit()"
        ),
    ),
    AntiPattern(
        pattern_id="AUTH-004",
        description="HTTP method override allows bypassing method-based access control.",
        anti_pattern="method = request.headers.get('X-HTTP-Method-Override', request.method)",
        safe_alternative="Never accept method overrides from untrusted headers for authenticated routes.",
        severity="medium",
        cwe="CWE-284",
        source="OWASP",
        example_vulnerable=(
            "def handle_request(request):\n"
            "    method = request.headers.get(\n"
            "        'X-HTTP-Method-Override', request.method\n"
            "    )\n"
            "    if method == 'DELETE':\n"
            "        return delete_resource(request)"
        ),
        example_safe=(
            "def handle_request(request):\n"
            "    # Only trust actual HTTP method; ignore override headers\n"
            "    if request.method == 'DELETE':\n"
            "        if not request.user.has_perm('delete_resource'):\n"
            "            abort(403)\n"
            "        return delete_resource(request)"
        ),
    ),
    AntiPattern(
        pattern_id="AUTH-005",
        description="Mass assignment allows overwriting protected fields (e.g. is_admin) via JSON body.",
        anti_pattern="user.update(**request.json)  — no field allowlist",
        safe_alternative="Use an explicit allowlist of updatable fields.",
        severity="high",
        cwe="CWE-915",
        source="OWASP",
        example_vulnerable=(
            "def update_user(user_id, request):\n"
            "    user = User.query.get(user_id)\n"
            "    for key, value in request.json.items():\n"
            "        setattr(user, key, value)  # Attacker can set is_admin=True\n"
            "    db.session.commit()"
        ),
        example_safe=(
            "ALLOWED_UPDATE_FIELDS = {'name', 'email', 'bio', 'avatar_url'}\n"
            "def update_user(user_id, request):\n"
            "    user = User.query.get(user_id)\n"
            "    for key, value in request.json.items():\n"
            "        if key not in ALLOWED_UPDATE_FIELDS:\n"
            "            abort(400, f'Field {key!r} cannot be updated')\n"
            "        setattr(user, key, value)\n"
            "    db.session.commit()"
        ),
    ),
]

PATH_TRAVERSAL_PATTERNS: list[AntiPattern] = [
    AntiPattern(
        pattern_id="PATH-001",
        description="File read using user-supplied filename without path sanitisation (CWE-22).",
        anti_pattern="open(BASE_DIR + '/' + user_filename)",
        safe_alternative="Use os.path.realpath and verify the result starts with BASE_DIR.",
        severity="high",
        cwe="CWE-22",
        source="OWASP",
        example_vulnerable=(
            "BASE = '/var/uploads'\n"
            "def read_file(filename):\n"
            "    with open(BASE + '/' + filename) as f:\n"
            "        return f.read()"
        ),
        example_safe=(
            "import os\n"
            "BASE = '/var/uploads'\n"
            "def read_file(filename):\n"
            "    safe_path = os.path.realpath(os.path.join(BASE, filename))\n"
            "    if not safe_path.startswith(BASE + os.sep):\n"
            "        raise ValueError('Path traversal detected')\n"
            "    with open(safe_path) as f:\n"
            "        return f.read()"
        ),
    ),
    AntiPattern(
        pattern_id="PATH-002",
        description="Archive extraction (zipfile/tarfile) without validating member paths — zip-slip.",
        anti_pattern="zip_ref.extractall(extract_dir)  — without checking member.filename",
        safe_alternative="Validate each archive member's path before extraction.",
        severity="high",
        cwe="CWE-22",
        source="NVD",
        example_vulnerable=(
            "def extract_archive(zip_path, extract_dir):\n"
            "    with zipfile.ZipFile(zip_path) as z:\n"
            "        z.extractall(extract_dir)  # zip-slip!"
        ),
        example_safe=(
            "def extract_archive(zip_path, extract_dir):\n"
            "    with zipfile.ZipFile(zip_path) as z:\n"
            "        for member in z.infolist():\n"
            "            dest = os.path.realpath(\n"
            "                os.path.join(extract_dir, member.filename)\n"
            "            )\n"
            "            if not dest.startswith(extract_dir + os.sep):\n"
            "                raise ValueError(f'Zip-slip: {member.filename}')\n"
            "            z.extract(member, extract_dir)"
        ),
    ),
    AntiPattern(
        pattern_id="PATH-003",
        description="send_file() with user-controlled path bypasses Flask static file sandbox.",
        anti_pattern="flask.send_file(request.args['path'])",
        safe_alternative="Use flask.send_from_directory with a fixed directory parameter.",
        severity="high",
        cwe="CWE-22",
        source="OWASP",
        example_vulnerable=(
            "from flask import send_file, request\n"
            "def serve_file():\n"
            "    return send_file(request.args.get('file'))"
        ),
        example_safe=(
            "from flask import send_from_directory\n"
            "DOWNLOAD_DIR = '/var/www/downloads'\n"
            "def serve_file(filename):\n"
            "    return send_from_directory(DOWNLOAD_DIR, filename)"
        ),
    ),
    AntiPattern(
        pattern_id="PATH-004",
        description="Template rendering with user-supplied template name allows arbitrary file inclusion.",
        anti_pattern="render_template(request.args['template'])",
        safe_alternative="Validate template name against a strict allowlist.",
        severity="high",
        cwe="CWE-22",
        source="HackerOne",
        example_vulnerable=(
            "from flask import render_template, request\n"
            "def render_page():\n"
            "    tpl = request.args.get('page', 'index.html')\n"
            "    return render_template(tpl)"
        ),
        example_safe=(
            "ALLOWED_TEMPLATES = {'index.html', 'about.html', 'contact.html'}\n"
            "def render_page():\n"
            "    tpl = request.args.get('page', 'index.html')\n"
            "    if tpl not in ALLOWED_TEMPLATES:\n"
            "        abort(404)\n"
            "    return render_template(tpl)"
        ),
    ),
    AntiPattern(
        pattern_id="PATH-005",
        description="Log file rotation using user-controlled filename — path traversal to overwrite files.",
        anti_pattern="open(LOG_DIR + '/' + request_param + '.log', 'a')",
        safe_alternative="Sanitise log filename: strip directory separators and enforce extension.",
        severity="medium",
        cwe="CWE-22",
        source="CWE",
        example_vulnerable=(
            "def write_log(app_name, message):\n"
            "    log_path = f'/var/log/apps/{app_name}.log'\n"
            "    with open(log_path, 'a') as f:\n"
            "        f.write(message + '\\n')"
        ),
        example_safe=(
            "import re\n"
            "LOG_NAME_RE = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')\n"
            "def write_log(app_name, message):\n"
            "    if not LOG_NAME_RE.match(app_name):\n"
            "        raise ValueError('Invalid app_name')\n"
            "    log_path = f'/var/log/apps/{app_name}.log'\n"
            "    with open(log_path, 'a') as f:\n"
            "        f.write(message + '\\n')"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Library class
# ---------------------------------------------------------------------------


class AntiPatternLibrary:
    """In-memory store of vulnerability anti-patterns with search capability.

    Pre-seeded with 25 authoritative patterns across 5 vulnerability classes.
    Supports runtime addition of patterns derived from live threat indicators.

    The library is the static backbone of Layer 2: it provides a stable,
    curated baseline that complements the dynamic threat intelligence from
    ACP connectors.
    """

    def __init__(self) -> None:
        self._patterns: dict[str, AntiPattern] = {}
        self._seed_default_patterns()

    def _seed_default_patterns(self) -> None:
        """Pre-seed the library with the 25 built-in patterns."""
        all_patterns = (
            IDOR_PATTERNS
            + SQLI_PATTERNS
            + SSRF_PATTERNS
            + AUTH_BYPASS_PATTERNS
            + PATH_TRAVERSAL_PATTERNS
        )
        for pattern in all_patterns:
            self._patterns[pattern.pattern_id] = pattern
        logger.info(
            "AntiPatternLibrary seeded with %d patterns.", len(self._patterns)
        )

    def add_pattern(self, pattern: AntiPattern) -> None:
        """Add or replace a pattern in the library.

        Args:
            pattern: AntiPattern to add.

        Raises:
            TypeError: If ``pattern`` is not an AntiPattern instance.
        """
        if not isinstance(pattern, AntiPattern):
            raise TypeError("Expected AntiPattern instance.")
        self._patterns[pattern.pattern_id] = pattern
        logger.debug("Added pattern %s", pattern.pattern_id)

    def get(self, pattern_id: str) -> AntiPattern | None:
        """Retrieve a pattern by ID.

        Args:
            pattern_id: Pattern identifier.

        Returns:
            AntiPattern | None: Pattern or None if not found.
        """
        return self._patterns.get(pattern_id)

    def search(self, query: str, top_k: int = 5) -> list[AntiPattern]:
        """Keyword search over pattern descriptions and anti-pattern strings.

        A simple bag-of-words scoring: returns patterns whose description or
        anti_pattern fields contain the most query terms.

        Args:
            query: Search string (space-separated terms).
            top_k: Maximum number of results to return.

        Returns:
            list[AntiPattern]: Up to ``top_k`` matching patterns, sorted by
                relevance (term overlap score).
        """
        if not query:
            return list(self._patterns.values())[:top_k]
        terms = query.lower().split()
        scored: list[tuple[int, AntiPattern]] = []
        for pattern in self._patterns.values():
            searchable = (
                pattern.description.lower()
                + " "
                + pattern.anti_pattern.lower()
                + " "
                + pattern.cwe.lower()
                + " "
                + pattern.example_vulnerable.lower()
            )
            score = sum(1 for term in terms if term in searchable)
            if score > 0:
                scored.append((score, pattern))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def by_cwe(self, cwe: str) -> list[AntiPattern]:
        """Return all patterns for a given CWE identifier.

        Args:
            cwe: CWE identifier (e.g. "CWE-89").

        Returns:
            list[AntiPattern]: Matching patterns.
        """
        return [p for p in self._patterns.values() if p.cwe == cwe]

    def by_severity(self, severity: str) -> list[AntiPattern]:
        """Return all patterns for a given severity level.

        Args:
            severity: "critical", "high", "medium", or "low".

        Returns:
            list[AntiPattern]: Matching patterns.
        """
        return [p for p in self._patterns.values() if p.severity == severity]

    def add_from_threat_indicator(
        self, indicator: object  # ThreatIndicator — avoid circular import
    ) -> AntiPattern:
        """Derive and store an AntiPattern from a live ThreatIndicator.

        Maps the indicator's semantic rule, CWE, and affected patterns to
        a new AntiPattern with source set to the indicator's origin.

        Args:
            indicator: :class:`~core.ThreatIndicator` instance.

        Returns:
            AntiPattern: The newly created pattern.
        """
        from core.threat_indicator import ThreatIndicator
        if not isinstance(indicator, ThreatIndicator):
            raise TypeError("Expected ThreatIndicator instance.")
        pattern_id = f"TI-{indicator.id}"
        anti_pattern_text = (
            "; ".join(indicator.affected_patterns[:3])
            if indicator.affected_patterns
            else indicator.semantic_rule[:200]
        )
        pattern = AntiPattern(
            pattern_id=pattern_id,
            description=f"[{indicator.source}] {indicator.description[:200]}",
            anti_pattern=anti_pattern_text,
            safe_alternative="Apply security control for: "
            + (indicator.cwe[0] if indicator.cwe else "unknown vulnerability class"),
            severity=indicator.severity,
            cwe=indicator.cwe[0] if indicator.cwe else "unknown",
            source=indicator.source,
            stix_id=indicator.stix_pattern,
            example_vulnerable=anti_pattern_text,
            example_safe="",
        )
        self.add_pattern(pattern)
        return pattern

    def export_jsonl(self, path: str) -> None:
        """Export all patterns to a JSONL file for use as training data.

        Each line is a JSON object with all AntiPattern fields.

        Args:
            path: Output file path.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for pattern in self._patterns.values():
                f.write(json.dumps(asdict(pattern)) + "\n")
        logger.info(
            "Exported %d patterns to %s", len(self._patterns), path
        )

    def __len__(self) -> int:
        return len(self._patterns)

    def __repr__(self) -> str:
        return f"AntiPatternLibrary(patterns={len(self._patterns)})"
