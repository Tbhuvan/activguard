"""Example file with known vulnerabilities — for testing ActivGuard detection."""
import sqlite3
import requests
import os


def get_user_profile(request, user_id):
    # No ownership check — classic IDOR (CWE-639)
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = " + str(user_id))
    return cursor.fetchone()


def search_products(request):
    name = request.GET["name"]
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()
    # SQL injection via string formatting (CWE-89)
    query = "SELECT * FROM products WHERE name = '%s'" % name
    cursor.execute(query)
    return cursor.fetchall()


def fetch_remote_data(request):
    target_url = request.args.get("url")
    # SSRF — user controls the URL (CWE-918)
    response = requests.get(target_url)
    return response.text


def serve_file(request):
    filename = request.args.get("filename")
    base_dir = "/var/www/uploads"
    # Path traversal — no sanitisation (CWE-22)
    filepath = os.path.join(base_dir, filename)
    with open(filepath, "rb") as f:
        return f.read()


def admin_panel(request):
    # Auth bypass — checks query param instead of session (CWE-285)
    if request.args.get("is_admin") == "true":
        return "admin access granted"
    return "access denied"
