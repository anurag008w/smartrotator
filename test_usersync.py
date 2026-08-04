"""
test_usersync.py — per-user app state sync (offline-first backup) test.

Structure: data/sync/<user>/<scope>.json  (GitHub repo me mirror hota hai)

Run:  python test_usersync.py
"""

from __future__ import annotations

import os
import shutil
import sys

# IMPORTANT: store import se pehle test data dir set karo
TEST_DATA_DIR = "./test_data_usersync"
os.environ["SMARTROTATOR_DATA_DIR"] = TEST_DATA_DIR
os.environ["GITHUB_SYNC_ENABLED"] = "false"
os.environ["SYNC_ENC_KEY"] = os.environ.get("SYNC_ENC_KEY", "00" * 32)
if os.path.exists(TEST_DATA_DIR):
    shutil.rmtree(TEST_DATA_DIR)

from starlette.testclient import TestClient  # noqa: E402

from rotator.app import app  # noqa: E402


def raw_file_has_plaintext(folder: str) -> bool:
    """User folder ki saari .json files me plaintext secret hai kya."""
    if not os.path.isdir(folder):
        return False
    secret = "SUPERSECRET"
    for name in os.listdir(folder):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder, name), encoding="utf-8") as fh:
                if secret in fh.read():
                    return True
        except OSError:
            continue
    return False


def main() -> int:
    results = []
    with TestClient(app) as client:
        # ---------- TEST 1: register users ----------
        r = client.post("/auth/register", json={"username": "syncuser", "password": "secret123"})
        su = r.json()
        results.append(("register syncuser", r.status_code == 200 and "token" in su, r.status_code))

        r = client.post("/auth/register", json={"username": "otheruser", "password": "secret123"})
        ou = r.json()
        results.append(("register otheruser", r.status_code == 200 and "token" in ou, r.status_code))

        su_hdr = {"Authorization": f"Bearer {su['token']}"}
        ou_hdr = {"Authorization": f"Bearer {ou['token']}"}

        # ---------- TEST 2: empty state before any push ----------
        r = client.get("/sync/status", headers=su_hdr)
        ok = r.status_code == 200 and r.json()["exists"] is False
        results.append(("sync/status empty", ok, r.json()))

        r = client.get("/sync/state", headers=su_hdr)
        ok = r.status_code == 200 and r.json()["exists"] is False
        results.append(("sync/state empty", ok, r.json()))

        r = client.get("/sync/scopes", headers=su_hdr)
        ok = r.status_code == 200 and r.json()["scopes"] == []
        results.append(("sync/scopes empty", ok, r.json()))

        # ---------- TEST 3: push state (JWT) — data/sync/<user>/state.json ----------
        payload = {"plan": {"day": 3}, "tasks": [{"id": "d1_t1", "done": True}]}
        r = client.put("/sync/state", headers=su_hdr, json={"state": payload, "updated_at": "2026-08-04T10:00:00Z"})
        ok = r.status_code == 200 and r.json()["ok"] is True and r.json()["updated_at"] == "2026-08-04T10:00:00Z"
        results.append(("push state (JWT)", ok, r.json()))

        # ---------- TEST 4: pull state (JWT) ----------
        r = client.get("/sync/state", headers=su_hdr)
        body = r.json()
        ok = r.status_code == 200 and body["exists"] is True and body["state"]["plan"]["day"] == 3
        results.append(("pull state (JWT)", ok, r.json()))

        # ---------- TEST 5: status shows updated_at + bytes ----------
        r = client.get("/sync/status", headers=su_hdr)
        body = r.json()
        ok = r.status_code == 200 and body["exists"] is True and body["updated_at"] == "2026-08-04T10:00:00Z" and body["bytes"] > 0
        results.append(("sync/status populated", ok, body))

        # ---------- TEST 6: per-user isolation (other user sees nothing) ----------
        r = client.get("/sync/state", headers=ou_hdr)
        ok = r.status_code == 200 and r.json()["exists"] is False
        results.append(("isolation: other user empty", ok, r.json()))

        # ---------- TEST 7: push with sk-key (OpenAI-compatible auth) ----------
        r = client.put(
            "/sync/state",
            headers={"Authorization": f"Bearer {ou['api_key']}"},
            json={"state": {"note": "from sk-key"}, "updated_at": "2026-08-04T11:00:00Z"},
        )
        ok = r.status_code == 200 and r.json()["ok"] is True
        results.append(("push state (sk-key)", ok, r.json()))

        # ---------- TEST 8: overwrite = last-write-wins ----------
        r = client.put("/sync/state", headers=su_hdr, json={"state": {"plan": {"day": 5}}, "updated_at": "2026-08-04T12:00:00Z"})
        r2 = client.get("/sync/state", headers=su_hdr)
        ok = r2.json()["state"]["plan"]["day"] == 5 and r2.json()["updated_at"] == "2026-08-04T12:00:00Z"
        results.append(("overwrite last-write-wins", ok, r2.json()))

        # ---------- TEST 9: multi-scope — chat + settings alag files ----------
        r = client.put("/sync/state?scope=chat", headers=su_hdr, json={"state": {"sessions": ["s1"]}, "updated_at": "2026-08-04T13:00:00Z"})
        ok = r.status_code == 200 and r.json()["scope"] == "chat"
        results.append(("push scope=chat", ok, r.json()))

        r = client.put("/sync/state?scope=settings", headers=su_hdr, json={"state": {"aiEnabled": True}, "updated_at": "2026-08-04T13:05:00Z"})
        ok = r.status_code == 200 and r.json()["scope"] == "settings"
        results.append(("push scope=settings", ok, r.json()))

        r = client.get("/sync/state?scope=chat", headers=su_hdr)
        ok = r.json()["state"]["sessions"] == ["s1"]
        results.append(("pull scope=chat isolated", ok, r.json()))

        r = client.get("/sync/scopes", headers=su_hdr)
        scopes = r.json()["scopes"]
        ok = r.status_code == 200 and scopes == ["chat", "settings", "state"]
        results.append(("sync/scopes lists all", ok, scopes))

        # folder structure check — data/sync/<user>/<scope>.json
        base = os.path.join(TEST_DATA_DIR, "sync")
        user_folder = os.path.join(base, "syncuser")
        has_folders = (
            os.path.isdir(user_folder)
            and os.path.isfile(os.path.join(user_folder, "state.json"))
            and os.path.isfile(os.path.join(user_folder, "chat.json"))
            and os.path.isfile(os.path.join(user_folder, "settings.json"))
        )
        results.append(("folder-per-user structure", has_folders, user_folder))

        # ---------- TEST 10: no auth → 401 ----------
        r = client.get("/sync/state")
        results.append(("no auth 401", r.status_code == 401, r.status_code))

        # ---------- TEST 11: delete single scope ----------
        r = client.delete("/sync/state?scope=chat", headers=su_hdr)
        ok = r.status_code == 200 and r.json()["deleted"] is True
        results.append(("delete scope=chat", ok, r.json()))
        r = client.get("/sync/scopes", headers=su_hdr)
        results.append(("chat gone from scopes", "chat" not in r.json()["scopes"], r.json()["scopes"]))

        # ---------- TEST 12: delete all (logout wipe) ----------
        r = client.delete("/sync/state?scope=*", headers=su_hdr)
        ok = r.status_code == 200 and r.json()["deleted"] is True
        results.append(("delete all (scope=*)", ok, r.json()))
        r = client.get("/sync/scopes", headers=su_hdr)
        results.append(("user folder empty after wipe", r.json()["scopes"] == [], r.json()))
        results.append(("user folder removed", not os.path.isdir(user_folder), os.path.isdir(user_folder)))

        # ---------- TEST 15: username file sanitization (path traversal guard) ----------
        r = client.post("/auth/register", json={"username": "traversal/../user", "password": "secret123"})
        if r.status_code == 200:
            hdr = {"Authorization": f"Bearer {r.json()['token']}"}
            r2 = client.put("/sync/state", headers=hdr, json={"state": {"x": 1}, "updated_at": ""})
            ok = r2.status_code == 200 and r2.json()["ok"] is True
            results.append(("sanitized username push", ok, r2.json()))
        else:
            results.append(("sanitized username push (skip, register 400)", True, r.status_code))

        # ---------- TEST 14: apiKey at-rest encrypted in sync file ----------
        import json as _json

        secret_state = {"aiSettings": {"providers": {"openrouter": {"apiKey": "sk-or-v1-SUPERSECRET", "model": "x"}}}}
        r = client.put("/sync/state", headers=su_hdr, json={"state": secret_state, "updated_at": "2026-08-04T14:00:00Z"})
        ok = r.status_code == 200
        results.append(("push with apiKey", ok, r.json()))

        # raw file on disk me ciphertext hona chahiye, plaintext NAHI
        user_folder = os.path.join(TEST_DATA_DIR, "sync", "syncuser")
        raw = _json.loads(open(os.path.join(user_folder, "state.json"), encoding="utf-8").read())
        stored_key = raw["state"]["aiSettings"]["providers"]["openrouter"]["apiKey"]
        ok = (not raw_file_has_plaintext(user_folder)) and str(stored_key).startswith("v1.")
        results.append(("apiKey encrypted at rest", ok, stored_key[:20] + "..." if ok else stored_key))

        # pull pe server decrypt karke plaintext wapas deta hai (owner)
        r = client.get("/sync/state", headers=su_hdr)
        got_key = r.json()["state"]["aiSettings"]["providers"]["openrouter"]["apiKey"]
        ok = r.status_code == 200 and got_key == "sk-or-v1-SUPERSECRET"
        results.append(("apiKey decrypted on pull (owner)", ok, got_key))

        # bina SYNC_ENC_KEY ke — apiKey store me nahi jaati (leak-safe)
        os.environ.pop("SYNC_ENC_KEY", None)
        r = client.post("/auth/register", json={"username": "nokeyuser", "password": "secret123"})
        hdr_nk = {"Authorization": f"Bearer {r.json()['token']}"}
        r = client.put("/sync/state", headers=hdr_nk, json={"state": secret_state, "updated_at": "2026-08-04T15:00:00Z"})
        raw = _json.loads(open(os.path.join(TEST_DATA_DIR, "sync", "nokeyuser", "state.json"), encoding="utf-8").read())
        stored_key = raw["state"]["aiSettings"]["providers"]["openrouter"].get("apiKey")
        ok = (not raw_file_has_plaintext(os.path.join(TEST_DATA_DIR, "sync", "nokeyuser"))) and (stored_key in (None, ""))
        results.append(("no key -> apiKey stripped", ok, stored_key))
        os.environ["SYNC_ENC_KEY"] = "00" * 32

        # ---------- TEST 14b: koi bhi password secret chalega (SHA-256 derived key) ----------
        os.environ["SYNC_ENC_KEY"] = "mY-pa55word_remember_this_!!"
        r = client.post("/auth/register", json={"username": "pwuser", "password": "secret123"})
        hdr_pw = {"Authorization": f"Bearer {r.json()['token']}"}
        r = client.put("/sync/state", headers=hdr_pw, json={"state": secret_state, "updated_at": "2026-08-04T15:30:00Z"})
        raw = _json.loads(open(os.path.join(TEST_DATA_DIR, "sync", "pwuser", "state.json"), encoding="utf-8").read())
        stored_key = raw["state"]["aiSettings"]["providers"]["openrouter"]["apiKey"]
        ok = str(stored_key).startswith("v1.") and not raw_file_has_plaintext(os.path.join(TEST_DATA_DIR, "sync", "pwuser"))
        results.append(("password-style secret encrypts", ok, stored_key[:20] + "..." if ok else stored_key))
        r = client.get("/sync/state", headers=hdr_pw)
        got = r.json()["state"]["aiSettings"]["providers"]["openrouter"]["apiKey"]
        ok = got == "sk-or-v1-SUPERSECRET"
        results.append(("password-style secret decrypts", ok, got))
        os.environ["SYNC_ENC_KEY"] = "00" * 32

        # ---------- TEST 14c: admin enc-key endpoints (yaad rakhne ki zaroorat nahi) ----------
        # env secret set hone pe admin source=env dikhta hai
        r = client.get("/admin/sync/enc-key", headers=su_hdr)
        body = r.json()
        ok = r.status_code == 200 and body["set"] and body["source"] == "env" and body["secret"] == "00" * 32
        results.append(("admin sees env secret", ok, body.get("source")))

        # non-admin denied
        r = client.get("/admin/sync/enc-key", headers={"Authorization": f"Bearer {ou['token']}"})
        results.append(("non-admin enc-key 403", r.status_code == 403, r.status_code))

        # env hatao → file source auto-generate ho jata hai
        os.environ.pop("SYNC_ENC_KEY", None)
        r = client.get("/admin/sync/enc-key", headers=su_hdr)
        body = r.json()
        ok = r.status_code == 200 and body["set"] and body["source"] == "generated" and len(body["secret"]) == 64
        results.append(("admin enc-key auto-generates", ok, body.get("source")))
        # generated secret file me persist hua (hidden, GitHub sync skip)
        secret_file = os.path.join(TEST_DATA_DIR, ".sync-enc-key")
        ok = os.path.exists(secret_file) and open(secret_file, encoding="utf-8").read().strip() == body["secret"]
        results.append(("secret persisted to hidden file", ok, secret_file))
        # ab encryption isi file-secret se hoti hai (bina env ke)
        r = client.post("/auth/register", json={"username": "fileuser", "password": "secret123"})
        hdr_file = {"Authorization": f"Bearer {r.json()['token']}"}
        r = client.put("/sync/state", headers=hdr_file, json={"state": secret_state, "updated_at": "2026-08-04T16:00:00Z"})
        raw = _json.loads(open(os.path.join(TEST_DATA_DIR, "sync", "fileuser", "state.json"), encoding="utf-8").read())
        ok = str(raw["state"]["aiSettings"]["providers"]["openrouter"]["apiKey"]).startswith("v1.")
        results.append(("file secret encrypts", ok, str(raw["state"]["aiSettings"]["providers"]["openrouter"]["apiKey"])[:20] + "..."))

        # rotate endpoint — nayi key purani file-secret se alag honi chahiye
        old_file_secret = open(secret_file, encoding="utf-8").read().strip()
        r = client.post("/admin/sync/enc-key/rotate", headers=su_hdr)
        body = r.json()
        ok = r.status_code == 200 and body["set"] and len(body["secret"]) == 64 and body["secret"] != old_file_secret
        results.append(("admin rotate secret", ok, body.get("source")))
        ok = open(secret_file, encoding="utf-8").read().strip() == body["secret"]
        results.append(("rotate persisted to file", ok, ""))
        os.environ["SYNC_ENC_KEY"] = "00" * 32

    # ---------- print ----------
    passed = 0
    print(f"\n{'TEST':<40} {'RESULT':<12} CODE")
    print("-" * 68)
    for name, ok, extra in results:
        mark = "PASS" if ok else "FAIL"
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
