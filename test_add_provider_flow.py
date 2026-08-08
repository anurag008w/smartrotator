"""
test_add_provider_flow.py — verifies the rewritten "🔌 Providers" tab ADD flow:

- GET /admin/providers/catalog: known presets + env-key detection + already_added
- GET /admin/providers/detect-keys: structured per-key list (index + preview) —
  what the modal renders as individually-selectable "API Key 1 / API Key 2 / ..."
- POST /admin/providers WITHOUT api_keys AND WITHOUT models: keys are pulled
  straight from <NAME>_KEYS env var, and models are no longer required at all
  (model selection now lives entirely in the 🎚 Exposed Models tab)
- POST /admin/providers with `selected_keys`: only the chosen subset of the
  env-detected keys ends up active for that base_url — NOT all of them
- Multiple selected keys on one base_url all rotate together (key_count matches)
- POST /admin/providers/test-url: validates a base_url before saving (checked
  here via its input-validation error paths, since the success path needs a
  real network call)
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

        # custom/manual name detect-keys — no env var set yet
        r = client.get("/admin/providers/detect-keys", params={"name": "cerebras"}, headers=h)
        assert r.status_code == 200, r.text
        assert r.json()["key_count"] == 0, r.json()
        assert r.json()["keys"] == [], r.json()
        return True


def test_detect_keys_returns_individually_selectable_list():
    """The modal needs "API Key 1 / API Key 2 / ..." checkboxes — verify the
    structured {index, preview} list detect-keys returns matches env order."""
    app = app_module.app
    os.environ["GROQ_KEYS"] = "groq-key-aaa,groq-key-bbb,groq-key-ccc"
    with TestClient(app) as client:
        app.state.rotator = Rotator(config_path="config.yaml")
        tok = _register_superadmin(client, "root_detect")
        h = {"Authorization": "Bearer " + tok}

        r = client.get("/admin/providers/detect-keys", params={"name": "groq"}, headers=h)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["key_count"] == 3, body
        assert [k["index"] for k in body["keys"]] == [0, 1, 2], body
        assert all(k["preview"] for k in body["keys"]), body
        return True


def test_add_provider_no_models_no_manual_keys():
    """Core of the fix: no `models` field sent at all (Exposed tab's job now),
    no `api_keys` sent (env auto-detected) — provider still saves fine.
    Uses a name NOT in config.yaml so there's no static models fallback."""
    app = app_module.app
    os.environ["FIREWORKS_KEYS"] = "fw-key-aaa, fw-key-bbb, fw-key-ccc"
    with TestClient(app) as client:
        app.state.rotator = Rotator(config_path="config.yaml")
        tok = _register_superadmin(client, "root_addflow")
        h = {"Authorization": "Bearer " + tok}

        # UI sends NO api_keys AND NO models — just name/type/base_url/enabled
        r = client.post("/admin/providers", headers=h, json={
            "name": "fireworks",
            "type": "openai",
            "base_url": "https://api.fireworks.ai/inference/v1",
            "enabled": True,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["key_count"] == 3, body  # all 3 env keys picked up, one base_url

        r = client.get("/admin/providers", headers=h)
        prov = next(p for p in r.json()["providers"] if p["name"] == "fireworks")
        assert prov["key_count"] == 3, prov
        assert len(prov["keys"]) == 3, prov
        assert prov["base_url"] == "https://api.fireworks.ai/inference/v1", prov
        assert prov["models"] == [], prov  # models genuinely empty — Exposed tab's job
        os.environ.pop("FIREWORKS_KEYS", None)
        return True


def test_add_provider_with_selected_keys_subset():
    """Admin unchecks some of the individually-listed keys — only the checked
    subset should actually be active for this base_url."""
    app = app_module.app
    os.environ["ZEN_KEYS"] = "zen-key-1,zen-key-2,zen-key-3,zen-key-4"
    with TestClient(app) as client:
        app.state.rotator = Rotator(config_path="config.yaml")
        tok = _register_superadmin(client, "root_selectkeys")
        h = {"Authorization": "Bearer " + tok}

        r = client.get("/admin/providers/detect-keys", params={"name": "zen"}, headers=h)
        assert r.json()["key_count"] == 4, r.json()

        # admin only checks API Key 1 and API Key 3 (indices 0 and 2)
        r = client.post("/admin/providers", headers=h, json={
            "name": "zen",
            "type": "openai",
            "base_url": "https://opencode.ai/zen/v1",
            "enabled": True,
            "selected_keys": [0, 2],
        })
        assert r.status_code == 200, r.text

        r = client.get("/admin/providers", headers=h)
        prov = next(p for p in r.json()["providers"] if p["name"] == "zen")
        # sab 4 keys stored hoti hain (future re-selection ke liye), lekin
        # 'selected' flag batata hai ACTUALLY kaunsi is base_url pe rotate
        # ho rahi hain — yehi jo dashboard "in use" count me dikhata hai.
        assert prov["key_count"] == 4, prov
        assert len(prov["keys"]) == 4, prov
        selected_indices = [k["index"] for k in prov["keys"] if k["selected"] is not False]
        assert selected_indices == [0, 2], prov["keys"]
        return True


def test_test_url_endpoint_validation():
    """test-url ke validation paths — bina real network call ke."""
    app = app_module.app
    os.environ.pop("FIREWORKS_KEYS", None)
    with TestClient(app) as client:
        app.state.rotator = Rotator(config_path="config.yaml")
        tok = _register_superadmin(client, "root_testurl")
        h = {"Authorization": "Bearer " + tok}

        # koi key hi nahi mili — test fail hona chahiye, 500 nahi
        r = client.post("/admin/providers/test-url", headers=h, json={
            "name": "fireworks", "type": "openai", "base_url": "https://api.fireworks.ai/inference/v1",
        })
        assert r.status_code == 200, r.text
        assert r.json()["ok"] is False, r.json()

        os.environ["FIREWORKS_KEYS"] = "fw-test-key"
        # openai type ke liye base_url zaroori
        r = client.post("/admin/providers/test-url", headers=h, json={
            "name": "fireworks", "type": "openai", "base_url": "",
        })
        assert r.status_code == 200, r.text
        assert r.json()["ok"] is False, r.json()
        os.environ.pop("FIREWORKS_KEYS", None)
        return True


def test_resync_keys_after_env_grows():
    app = app_module.app
    os.environ["GROQ_KEYS"] = "groq-key-aaa, groq-key-bbb, groq-key-ccc"
    with TestClient(app) as client:
        app.state.rotator = Rotator(config_path="config.yaml")
        tok = _register_superadmin(client, "root_resync")
        h = {"Authorization": "Bearer " + tok}

        client.post("/admin/providers", headers=h, json={
            "name": "groq", "type": "openai", "base_url": "https://api.groq.com/openai/v1",
            "enabled": True,
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
        ("detect-keys individually-selectable list", test_detect_keys_returns_individually_selectable_list),
        ("add provider — no models, no manual keys", test_add_provider_no_models_no_manual_keys),
        ("add provider — selected_keys subset only", test_add_provider_with_selected_keys_subset),
        ("test-url validation paths", test_test_url_endpoint_validation),
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
    for k in ("GROQ_KEYS", "CEREBRAS_KEYS", "ZEN_KEYS", "FIREWORKS_KEYS", "ADMIN_USERS"):
        os.environ.pop(k, None)
    if os.path.exists("./test_data_addprov"):
        shutil.rmtree("./test_data_addprov")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
