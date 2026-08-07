"""
test_add_provider_flow.py — verifies the rewritten "🔌 Providers" tab ADD flow:

- GET /admin/providers/catalog: known presets + env-key detection + already_added
- GET /admin/providers/detect-keys: live env-key detection for a custom name
- POST /admin/providers WITHOUT api_keys: keys pulled straight from <NAME>_KEYS
  env var (the UI never sends api_keys anymore — no paste box in the modal)
- Multiple comma-separated keys in one env var all land on the SAME base_url
  and are all present in rotation (key_count matches)
- POST /admin/providers/{name}/resync-keys: re-pulls keys after the env var
  gains more keys, without deleting/re-adding the provider

Run: python test_add_provider_flow.py
"""
from __future__ import annotations

import os
import shutil

os.environ["SMARTROTATOR_DATA_DIR"] = "./test_data_addprov"
os.environ["GITHUB_SYNC_ENABLED"] = "false"
if os.path.exists("./test_data_addprov"):
    shutil.rmtree("./test_data_addprov")

from starlette.testclient import TestClient  # noqa: E402

from rotator import app as app_module  # noqa: E402
from rotator.router import Rotator  # noqa: E402


def _register_superadmin(client, username):
    os.environ["ADMIN_USERS"] = username
    r = client.post("/auth/register", json={"username": username, "password": "pass1234"})
    return r.json()["token"]


def test_catalog_and_detect_keys():
    app = app_module.app
    # koi env key nahi — is provider ke fresh state se shuru
    os.environ.pop("GROQ_KEYS", None)
    os.environ.pop("CEREBRAS_KEYS", None)
    with TestClient(app) as client:
        app.state.rotator = Rotator(config_path="config.yaml")
        tok = _register_superadmin(client, "root_catalog")
        h = {"Authorization": "Bearer " + tok}

        r = client.get("/admin/providers/catalog", headers=h)
        assert r.status_code == 200, r.text
        presets = r.json()["presets"]
        names = {p["name"] for p in presets}
        assert {"gemini", "groq", "openrouter", "nvidia", "zen"} <= names, names
        groq = next(p for p in presets if p["name"] == "groq")
        assert groq["env_var"] == "GROQ_KEYS", groq
        assert groq["env_key_count"] == 0, groq
        assert groq["already_added"] is False, groq
        assert groq["base_url"] == "https://api.groq.com/openai/v1", groq
        assert "llama-3.3-70b-versatile" in groq["models"], groq

        # custom/manual name detect-keys — no env var set yet
        r = client.get("/admin/providers/detect-keys", params={"name": "cerebras"}, headers=h)
        assert r.status_code == 200, r.text
        assert r.json()["key_count"] == 0, r.json()
        return True


def test_add_provider_pulls_keys_from_env_no_manual_input():
    """Core of the fix: select a provider, base_url + keys auto-fill from env,
    POST /admin/providers is called WITHOUT api_keys (like the new UI does)."""
    app = app_module.app
    os.environ["GROQ_KEYS"] = "groq-key-aaa, groq-key-bbb, groq-key-ccc"
    with TestClient(app) as client:
        app.state.rotator = Rotator(config_path="config.yaml")
        tok = _register_superadmin(client, "root_addflow")
        h = {"Authorization": "Bearer " + tok}

        # catalog now shows 3 detected keys before add
        r = client.get("/admin/providers/catalog", headers=h)
        groq = next(p for p in r.json()["presets"] if p["name"] == "groq")
        assert groq["env_key_count"] == 3, groq

        # UI sends NO api_keys field at all
        r = client.post("/admin/providers", headers=h, json={
            "name": "groq",
            "type": "openai",
            "base_url": "https://api.groq.com/openai/v1",
            "models": ["llama-3.3-70b-versatile"],
            "enabled": True,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["key_count"] == 3, body  # all 3 env keys picked up, one base_url

        r = client.get("/admin/providers", headers=h)
        prov = next(p for p in r.json()["providers"] if p["name"] == "groq")
        assert prov["key_count"] == 3, prov
        assert len(prov["keys"]) == 3, prov
        assert prov["base_url"] == "https://api.groq.com/openai/v1", prov
        return True


def test_resync_keys_after_env_grows():
    app = app_module.app
    os.environ["GROQ_KEYS"] = "groq-key-aaa, groq-key-bbb, groq-key-ccc, groq-key-ddd"
    with TestClient(app) as client:
        app.state.rotator = Rotator(config_path="config.yaml")
        tok = _register_superadmin(client, "root_resync")
        h = {"Authorization": "Bearer " + tok}

        # provider already added with 3 keys from a previous run's store —
        # simulate by adding it fresh here first with the old env value.
        os.environ["GROQ_KEYS"] = "groq-key-aaa, groq-key-bbb, groq-key-ccc"
        client.post("/admin/providers", headers=h, json={
            "name": "groq", "type": "openai", "base_url": "https://api.groq.com/openai/v1",
            "models": ["llama-3.3-70b-versatile"], "enabled": True,
        })

        # admin adds a 4th key to the env var (host secret) — resync picks it up
        os.environ["GROQ_KEYS"] = "groq-key-aaa, groq-key-bbb, groq-key-ccc, groq-key-ddd"
        r = client.post("/admin/providers/groq/resync-keys", headers=h)
        assert r.status_code == 200, r.text
        assert r.json()["key_count"] == 4, r.json()

        r = client.get("/admin/providers", headers=h)
        prov = next(p for p in r.json()["providers"] if p["name"] == "groq")
        assert prov["key_count"] == 4, prov
        return True


if __name__ == "__main__":
    tests = [
        ("catalog + detect-keys", test_catalog_and_detect_keys),
        ("add provider — env keys, no manual input", test_add_provider_pulls_keys_from_env_no_manual_input),
        ("resync keys after env grows", test_resync_keys_after_env_grows),
    ]
    failed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"PASS  {label}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {label}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {label}: {e!r}")
    for k in ("GROQ_KEYS", "CEREBRAS_KEYS", "ADMIN_USERS"):
        os.environ.pop(k, None)
    if os.path.exists("./test_data_addprov"):
        shutil.rmtree("./test_data_addprov")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
