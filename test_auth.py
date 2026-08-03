"""
test_auth.py — auth + per-user quota + admin flow test.

Run:  python test_auth.py
"""

from __future__ import annotations

import os
import shutil
import sys

# IMPORTANT: store import se pehle test data dir set karo
TEST_DATA_DIR = "./test_data_auth"
os.environ["SMARTROTATOR_DATA_DIR"] = TEST_DATA_DIR
os.environ["GITHUB_SYNC_ENABLED"] = "false"
if os.path.exists(TEST_DATA_DIR):
    shutil.rmtree(TEST_DATA_DIR)

from starlette.testclient import TestClient  # noqa: E402

from rotator.app import app  # noqa: E402


def main() -> int:
    results = []
    with TestClient(app) as client:
        # ---------- TEST 1: register (first user = admin) ----------
        r = client.post("/auth/register", json={"username": "alice", "password": "secret123"})
        ok = r.status_code == 200 and "token" in r.json() and r.json()["user"]["role"] == "admin"
        results.append(("register (first=admin)", ok, r.status_code))
        alice = r.json()

        # ---------- TEST 2: duplicate register rejected ----------
        r = client.post("/auth/register", json={"username": "alice", "password": "xxxxxx"})
        results.append(("duplicate register 409", r.status_code == 409, r.status_code))

        # ---------- TEST 3: login ----------
        r = client.post("/auth/login", json={"username": "alice", "password": "secret123"})
        ok = r.status_code == 200 and r.json()["api_key"].startswith("sk-")
        results.append(("login", ok, r.status_code))

        # ---------- TEST 4: wrong password ----------
        r = client.post("/auth/login", json={"username": "alice", "password": "wrong"})
        results.append(("bad login 401", r.status_code == 401, r.status_code))

        hdr = {"Authorization": f"Bearer {alice['token']}"}

        # ---------- TEST 5: /auth/me ----------
        r = client.get("/auth/me", headers=hdr)
        me = r.json()
        ok = r.status_code == 200 and me["user"]["username"] == "alice" and me["today"]["requests"] == 0
        results.append(("auth/me", ok, r.status_code))

        # ---------- TEST 6: gateway call with sk-key (providers empty → 502, quota refunded) ----------
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {alice['api_key']}"},
            json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}]},
        )
        body = r.json().get("detail", "")
        ok = r.status_code == 502 and not any(
            leak in body.lower()
            for leak in ("last error", "gemini", "groq", "openrouter", "zen", "configured")
        )
        results.append(("gateway sk-key 502 + no provider leak", ok, r.status_code))

        # quota refunded?
        r = client.get("/auth/me", headers=hdr)
        results.append(("quota refunded (0 used)", r.json()["today"]["requests"] == 0, r.json()["today"]["requests"]))

        # ---------- TEST 7: gateway without auth → 401 ----------
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}]},
        )
        results.append(("gateway no auth 401", r.status_code == 401, r.status_code))

        # ---------- TEST 8: quota exhaustion → 429 ----------
        # manually bump usage to limit
        import asyncio

        from rotator import store as database

        async def bump():
            await database.set_usage(alice["user"]["id"], database.today_utc(), alice["user"]["daily_limit"])

        asyncio.run(bump())
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {alice['api_key']}"},
            json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}]},
        )
        ok = r.status_code == 429 and "quota" in r.json()["detail"].lower()
        results.append(("quota exhausted 429", ok, r.status_code))

        # ---------- TEST 9: admin users list ----------
        r = client.get("/admin/users", headers=hdr)
        ok = r.status_code == 200 and len(r.json()["users"]) == 1
        results.append(("admin users list", ok, r.status_code))

        # ---------- TEST 10: register normal user + admin set limit ----------
        r = client.post("/auth/register", json={"username": "bob", "password": "bobpass123"})
        bob = r.json()
        bob_id = bob["user"]["id"]
        results.append(("register bob (user role)", bob["user"]["role"] == "user", bob["user"]["role"]))

        r = client.post(f"/admin/users/{bob_id}/limit", json={"daily_limit": 10}, headers=hdr)
        ok = r.status_code == 200 and r.json()["daily_limit"] == 10
        results.append(("admin set limit", ok, r.status_code))

        # bob admin access denied
        r = client.get("/admin/users", headers={"Authorization": f"Bearer {bob['token']}"})
        results.append(("bob admin 403", r.status_code == 403, r.status_code))

        # ---------- TEST 11: rotate key ----------
        old_key = bob["api_key"]
        r = client.post("/auth/rotate-key", headers={"Authorization": f"Bearer {bob['token']}"})
        ok = r.status_code == 200 and r.json()["api_key"] != old_key
        results.append(("rotate key", ok, r.status_code))

        # ---------- TEST 12: custom providers admin (mobile-app jaisa) ----------
        # non-admin access denied
        r = client.post(
            "/admin/providers",
            headers={"Authorization": f"Bearer {bob['token']}"},
            json={"name": "my-gemini", "type": "gemini", "api_keys": ["fake-key"]},
        )
        results.append(("custom provider non-admin 403", r.status_code == 403, r.status_code))

        # invalid type rejected
        r = client.post(
            "/admin/providers",
            headers=hdr,
            json={"name": "bad", "type": "nonsense", "api_keys": ["x"]},
        )
        results.append(("custom provider bad type 400", r.status_code == 400, r.status_code))

        # openai type bina base_url rejected
        r = client.post(
            "/admin/providers",
            headers=hdr,
            json={"name": "bad2", "type": "openai", "api_keys": ["x"]},
        )
        results.append(("openai provider no base_url 400", r.status_code == 400, r.status_code))

        # valid gemini provider add — live fetch fail ho jayega (fake key) par add ho jayega
        r = client.post(
            "/admin/providers",
            headers=hdr,
            json={"name": "my-gemini", "type": "gemini", "api_keys": ["fake-key-000"], "models": ["gemini-2.5-flash"]},
        )
        ok = r.status_code == 200 and r.json()["ok"]
        results.append(("custom provider add", ok, r.status_code))

        # list me dikhna chahiye
        r = client.get("/admin/providers", headers=hdr)
        names = [p["name"] for p in r.json().get("providers", [])]
        results.append(("custom provider list", "my-gemini" in names, names))

        # duplicate add = update (same name)
        r = client.post(
            "/admin/providers",
            headers=hdr,
            json={"name": "my-gemini", "type": "gemini", "api_keys": ["fake-key-000", "fake-key-111"], "models": ["gemini-2.5-flash", "gemini-2.5-pro"]},
        )
        r2 = client.get("/admin/providers", headers=hdr)
        my = next((p for p in r2.json()["providers"] if p["name"] == "my-gemini"), None)
        results.append(("custom provider update (2 keys)", my is not None and my["key_count"] == 2, my))

        # rotator me bhi live hai — /v1/models me dikhna chahiye
        r3 = client.get("/v1/models")
        ids = [m["id"] for m in r3.json()["data"]]
        results.append(("custom provider in /v1/models", "gemini-2.5-flash" in ids, ids[:5]))

        # delete
        r = client.delete("/admin/providers/my-gemini", headers=hdr)
        ok = r.status_code == 200 and r.json()["ok"]
        results.append(("custom provider delete", ok, r.status_code))
        r = client.get("/admin/providers", headers=hdr)
        names = [p["name"] for p in r.json().get("providers", [])]
        results.append(("custom provider gone after delete", "my-gemini" not in names, names))

    # ---------- print ----------
    passed = 0
    print(f"\n{'TEST':<40} {'RESULT':<12} CODE")
    print("-" * 68)
    for name, ok, extra in results:
        mark = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        print(f"{name:<40} {mark:<12} {extra}")
    print("-" * 68)
    print(f"{passed}/{len(results)} passed")
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
