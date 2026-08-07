# backend/scripts/test_path_traversal.py
# ─────────────────────────────────────────────────────────
# Standalone probe for the summary-download path-traversal fix.
#
# Logs in as a regular (non-admin) user, then asks the download
# endpoint for "..\..\.env" (URL-encoded as ..%5C..%5C.env) — the
# backslash form is the one that reaches the handler on Windows,
# where pathlib treats \ as a separator.
#
# Expected against the FIXED backend: 400 "Invalid filename."
# A 200 whose body contains JWT_SECRET means the hole is open.
#
# Usage (backend must be running):
#   python scripts/test_path_traversal.py
#   python scripts/test_path_traversal.py http://localhost:8000
#
# Requires: pip install requests
# ─────────────────────────────────────────────────────────

import sys
import requests

# Base URL of the running API — override via argv[1] if not on the default.
BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"

# Regular user, so this also proves the endpoint needs no admin role.
USERNAME = "user1"
PASSWORD = "user123"

# The traversal payload, already percent-encoded. Kept as a raw string in
# the URL so requests forwards %5C verbatim rather than re-encoding it.
PAYLOAD = "..%5C..%5C.env"


def main():
    # ── Step 1: log in, get the access token ──────────────
    login_url = f"{BASE_URL}/api/auth/login"
    try:
        login_resp = requests.post(
            login_url,
            json={"username": USERNAME, "password": PASSWORD},
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"Could not reach {login_url}: {e}")
        print("Is the backend running?")
        sys.exit(1)

    if login_resp.status_code != 200:
        print(f"Login failed: HTTP {login_resp.status_code}")
        print(login_resp.text[:200])
        sys.exit(1)

    token = login_resp.json().get("access_token")
    if not token:
        print("Login succeeded but no access_token in response.")
        print(login_resp.text[:200])
        sys.exit(1)

    print(f"Logged in as '{USERNAME}' — token acquired.")

    # ── Step 2: attempt the traversal download ────────────
    attack_url = f"{BASE_URL}/api/summary/download/{PAYLOAD}"
    headers = {"Authorization": f"Bearer {token}"}

    resp = requests.get(attack_url, headers=headers, timeout=10)

    # ── Step 3: report ────────────────────────────────────
    print(f"\nGET {attack_url}")
    print(f"Status code: {resp.status_code}")
    print(f"Body (first 200 chars): {resp.text[:200]}")


if __name__ == "__main__":
    main()
