"""
test_admin_sync_now.py — POST /admin/sync/now (force GitHub push) test.

'Sync now' button (app) is endpoint ko call karta hai taaki data GitHub pe
turant push ho (180s loop ka wait nahi). Sirf super admin (env ADMIN_USERS)
access kar sakta hai, rate-limited bhi hai.

Run:  python test_admin_sync_now.py
"""

from __future__ import annotations

import os
import shutil
import sys

# IMPORTANT: store import se pehle test data dir set karo
TEST_DATA_DIR = "./test_data_admin_sync_now"
os.environ["SMARTROTATOR_DATA_DIR"] = TEST_DATA_DIR
os.environ["GITHUB_SYNC_ENABLED"] = "false"
os.environ["ADMIN_USERS"] = "adminsync"
os.environ["SYNC_ENC_KEY"] = os.environ.get("SYNC_ENC_KEY", "00" * 32)
if os.path.exists(TEST_DATA_DIR):
    shutil.rmtree(TEST_DATA_DIR)

from starlette.testclient import TestClient  # noqa: E402

from rotator import github_sync  # noqa: E402
from rotator.app import app  # noqa: E402


def main() -> int:
    results = []
    with TestClient(app) as client:
        # ---------- register: admin + normal user ----------
        r = client.post("/auth/register", json={"username": "adminsync", "password": "secret123"})
        admin = r.json()
        results.append(("register adminsync", r.status_code == 200 and admin.get("is_super_admin") is True, r.status_code))

        r = client.post("/auth/register", json={"username": "plainuser", "password": "secret123"})
        plain = r.json()
        results.append(("register plainuser", r.status_code == 200 and plain.get("is_super_admin") is False, r.status_code))

        admin_hdr = {"Authorization": f"Bearer {admin['token']}"}
        plain_hdr = {"Authorization": f"Bearer {plain['token']}"}

        # ---------- no auth → 401 ----------
        r = client.post("/admin/sync/now")
        results.append(("no auth → 401", r.status_code == 401, r.status_code))

        # ---------- non-admin → 403 ----------
        r = client.post("/admin/sync/now", headers=plain_hdr)
        results.append(("non-admin → 403", r.status_code == 403, r.status_code))

        # ---------- admin → 200 + push_data called (monkeypatched) ----------
        calls: list[str] = []
        original_push = github_sync.push_data

        def fake_push(force_mode: str = "normal") -> bool:
            calls.append(force_mode)
            return True

        github_sync.push_data = fake_push
        try:
            r = client.post("/admin/sync/now", headers=admin_hdr)
        finally:
            github_sync.push_data = original_push
        body = r.json() if r.status_code == 200 else {}
        ok = (
            r.status_code == 200
            and body.get("pushed") is True
            and calls == ["normal"]
            and body.get("enabled") is False  # GITHUB_SYNC_ENABLED=false in tests
        )
        results.append(("admin → 200 + forced push", ok, r.status_code))

    # ---------- report ----------
    failed = 0
    for name, ok, code in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{status}] {name} (status={code})")
    print(f"\n{len(results) - failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
