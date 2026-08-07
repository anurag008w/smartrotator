"""
test_providers_tab.py — verifies "🔌 Providers" tab backend:
- GET /admin/providers: config.yaml wale HIDDEN, sirf custom providers dikhte hain
- POST /admin/providers: empty keys preserve, merge, replace_keys, base_url preserve
- custom provider same-name wale config provider ko override karta hai (live)

Run: python test_providers_tab.py
"""
from __future__ import annotations

import os
import shutil

os.environ["SMARTROTATOR_DATA_DIR"] = "./test_data_provtab"
os.environ["GITHUB_SYNC_ENABLED"] = "false"
if os.path.exists("./test_data_provtab"):
    shutil.rmtree("./test_data_provtab")

from starlette.testclient import TestClient  # noqa: E402

from rotator import app as app_module  # noqa: E402
from rotator.core import KeyRing  # noqa: E402
from rotator.providers import ChatResult, Provider  # noqa: E402
from rotator.router import ProviderConfig, ProviderState, Rotator  # noqa: E402


class EchoProvider(Provider):
    def __init__(self, name, models):
        super().__init__(models)
        self.name = name

    async def chat(self, messages, model, *, max_tokens=4096, temperature=0.7,
                    proxy=None, api_key=None, tools=None, tool_choice=None, **kwargs):
        return ChatResult(text=f"hi from {self.name}", provider=self.name, model=model, key_label=self.name)


def _setup_rotator():
    real = Rotator(config_path="config.yaml")
    # config.yaml wale providers ki tarah runtime state (sirf gemini wala key ke saath)
    provider = EchoProvider("gemini", ["gemini-2.5-flash"])
    cfg = ProviderConfig(
        name="gemini", ptype="gemini", models=["gemini-2.5-flash"],
        keys=["env-key-1", "env-key-2"], base_url=None,
    )
    ring = KeyRing(keys=cfg.keys, models=cfg.models, label="gemini")
    st = ProviderState(cfg=cfg, ring=ring, provider=provider)
    real.providers = [st]
    # canonical config snapshot bhi sync karo (delete/restore ke liye)
    real._config_providers = [st]
    return real


def _register_admin(client):
    r = client.post("/auth/register", json={"username": "boss1", "password": "pass1234"})
    return r.json()["token"]


def test_get_providers_merged():
    import os as _os

    _os.environ["ADMIN_USERS"] = "root1"
    app = app_module.app
    with TestClient(app) as client:
        app.state.rotator = _setup_rotator()
        # non-admin user — 403 aana chahiye
        r = client.post("/auth/register", json={"username": "plain1", "password": "pass1234"})
        h = {"Authorization": "Bearer " + r.json()["token"]}
        r = client.get("/admin/providers", headers=h)
        assert r.status_code == 403, r.text
        return True


def _register_and_promote_superadmin(client):
    """Register karke user ko super admin banao (encrypted admin env ke bina)."""
    r = client.post("/auth/register", json={"username": "root1", "password": "pass1234"})
    tok = r.json()["token"]
    return tok


def test_full_flow():
    import os as _os

    _os.environ["ADMIN_USERS"] = "root1"
    app = app_module.app
    with TestClient(app) as client:
        app.state.rotator = _setup_rotator()
        tok = _register_and_promote_superadmin(client)
        h = {"Authorization": "Bearer " + tok}

        # 1) GET — config.yaml wale providers UI me NAHI dikhte (hidden)
        r = client.get("/admin/providers", headers=h)
        assert r.status_code == 200, r.text
        names = {p["name"] for p in r.json()["providers"]}
        assert "gemini" not in names, names

        # 2) POST — naya custom provider add (base_url + keys + models)
        r = client.post("/admin/providers", headers=h, json={
            "name": "gemini",
            "type": "gemini",
            "base_url": "https://my-worker.workers.dev/v1beta",
            "api_keys": ["new-key-3"],
            "models": ["gemini-2.5-flash"],
            "enabled": True,
        })
        assert r.status_code == 200, r.text
        assert r.json()["key_count"] == 3, r.json()  # 2 runtime config keys + 1 new

        # custom store me check
        r = client.get("/admin/providers", headers=h)
        gem = next(p for p in r.json()["providers"] if p["name"] == "gemini")
        assert gem["base_url"] == "https://my-worker.workers.dev/v1beta", gem
        assert gem["key_count"] == 3, gem
        assert gem["source"] == "custom", gem
        assert any(k["preview"] for k in gem["keys"]), gem

        # 3) replace_keys = true — purani hatake nayi
        r = client.post("/admin/providers", headers=h, json={
            "name": "gemini", "type": "gemini", "api_keys": ["only-key"],
            "models": ["gemini-2.5-flash"], "replace_keys": True, "enabled": True,
        })
        assert r.status_code == 200, r.text
        assert r.json()["key_count"] == 1, r.json()

        # 4) base_url empty + keys empty = preserve sab
        r = client.post("/admin/providers", headers=h, json={
            "name": "gemini", "type": "gemini", "api_keys": [], "models": [],
            "enabled": True,
        })
        assert r.status_code == 200, r.text
        assert r.json()["key_count"] == 1, r.json()  # keys preserve
        gem = next(p for p in client.get("/admin/providers", headers=h).json()["providers"] if p["name"] == "gemini")
        assert gem["base_url"] == "https://my-worker.workers.dev/v1beta", gem
        assert gem["models"] == ["gemini-2.5-flash"], gem

        # 5) delete custom — config default (hidden) wapas, list se gayab
        r = client.delete("/admin/providers/gemini", headers=h)
        assert r.status_code == 200, r.text
        provs = client.get("/admin/providers", headers=h).json()["providers"]
        names = {p["name"] for p in provs}
        assert "gemini" not in names, names
        return True


if __name__ == "__main__":
    tests = [
        ("non-admin forbidden", test_get_providers_merged),
        ("GET merged + POST preserve/merge/replace/delete", test_full_flow),
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
    if os.path.exists("./test_data_provtab"):
        shutil.rmtree("./test_data_provtab")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
