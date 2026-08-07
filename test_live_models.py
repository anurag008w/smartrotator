"""
test_live_models.py — live models fetch tests.

Covers:
  1. NVIDIA-style pagination: server `after` param IGNORE karta hai (poore
     catalog ek page me deta hai) → fetch me duplicates nahi hone chahiye
     aur repeat page pe stop hona chahiye.
  2. OpenAI-style `last_id` pagination ab bhi multi-page fetch karta hai.
  3. config.yaml me nvidia provider sahi build hota hai (real base_url +
     models) jab NVIDIA_KEYS env ho.

Run: python test_live_models.py
"""
from __future__ import annotations

import asyncio
import os
import shutil

os.environ["SMARTROTATOR_DATA_DIR"] = "./test_data_live_models"
os.environ["GITHUB_SYNC_ENABLED"] = "false"
if os.path.exists("./test_data_live_models"):
    shutil.rmtree("./test_data_live_models")

import httpx  # noqa: E402

from rotator import providers as p  # noqa: E402


class FakeAsyncClient:
    """httpx.AsyncClient ka minimal fake — context manager + get()."""

    def __init__(self, handler):
        self.handler = handler
        self.urls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aclose(self):
        pass

    async def get(self, url, headers=None):
        self.urls.append(url)
        return self.handler(url)


def _resp(status: int, payload: dict, url: str) -> httpx.Response:
    return httpx.Response(status, json=payload, request=httpx.Request("GET", url))


def _patch_client(handler) -> list[str]:
    """providers.httpx.AsyncClient ko fake se replace karo. Returns url log."""
    log: list[str] = []

    def _fake_factory(*args, **kwargs):
        client = FakeAsyncClient(handler)
        client.urls = log
        return client

    p.httpx.AsyncClient = _fake_factory
    return log


def test_nvidia_style_pagination_dedup():
    """NVIDIA NIM: /v1/models poora catalog ek page me deta hai aur `after`
    param ignore karta hai. Isliye pehle wala code 5 requests pe 5x duplicate
    models banata tha. Ab: dedup + repeat page pe stop, 99 unique milne chahiye.
    """
    def handler(url):
        items = [{"id": f"model-{i}", "object": "model", "owned_by": "nvidia"} for i in range(99)]
        return _resp(200, {"object": "list", "data": items}, url)

    log = _patch_client(handler)
    models = asyncio.run(
        p.fetch_live_models(
            name="nvidia", ptype="openai",
            base_url="https://integrate.api.nvidia.com/v1",
            api_keys=["nvapi-test"], max_pages=5,
        )
    )
    ids = [m.id for m in models]
    assert len(models) == 99, f"99 unique expected, got {len(models)}"
    assert len(set(ids)) == 99, f"duplicates aaye: {len(ids) - len(set(ids))}"
    # page 1 + ek repeat page (naya model na aane par ruk jana chahiye)
    assert len(log) <= 2, f"zyda requests: {len(log)} -> {log}"
    return True


def test_openai_last_id_pagination_still_works():
    """Proper OpenAI-style (last_id in response) multi-page fetch ab bhi chalta hai."""
    def handler(url):
        if "after=m50" in url:
            # last page — 10 items, last_id nahi (len<20 → break)
            items = [{"id": f"m{i}", "object": "model"} for i in range(50, 60)]
            return _resp(200, {"object": "list", "data": items}, url)
        if "after=m25" in url:
            items = [{"id": f"m{i}", "object": "model"} for i in range(25, 50)]
            return _resp(200, {"object": "list", "data": items, "last_id": "m50"}, url)
        items = [{"id": f"m{i}", "object": "model"} for i in range(25)]
        return _resp(200, {"object": "list", "data": items, "last_id": "m25"}, url)

    log = _patch_client(handler)
    models = asyncio.run(
        p.fetch_live_models(
            name="openai-style", ptype="openai",
            base_url="https://api.example.com/v1",
            api_keys=["sk-test"], max_pages=5,
        )
    )
    ids = [m.id for m in models]
    assert len(models) == 60, f"60 expected, got {len(models)}"
    assert len(set(ids)) == 60, f"duplicates: {len(ids) - len(set(ids))}"
    assert len(log) == 3, f"3 requests expected, got {len(log)} -> {log}"
    return True


def test_nvidia_provider_from_config():
    """config.yaml me nvidia provider (openai type, real base_url + models)
    NVIDIA_KEYS env ke saath sahi build hota hai."""
    os.environ["NVIDIA_KEYS"] = "nvapi-test-key"
    from rotator.router import Rotator  # noqa: E402

    r = Rotator(config_path="config.yaml")
    nvidia = [st for st in r.providers if st.cfg.name == "nvidia"]
    assert len(nvidia) == 1, f"nvidia provider build nahi hua: {len(nvidia)}"
    st = nvidia[0]
    assert st.cfg.ptype == "openai", st.cfg.ptype
    assert st.cfg.base_url == "https://integrate.api.nvidia.com/v1", st.cfg.base_url
    assert "meta/llama-3.3-70b-instruct" in st.cfg.models, st.cfg.models
    assert st.cfg.keys == ["nvapi-test-key"], st.cfg.keys
    assert type(st.provider).__name__ == "OpenAICompatibleProvider", type(st.provider)
    return True


def test_only_key_configured_providers_visible():
    """Rule: jis provider ke paas env me kam se kam ek API key hai wahi dikhta
    hai — /admin/models providers, catalog, /v1/models — sab jagah. Bina key
    wale (jaise PASTE_ placeholder) providers/models kisi bhi tab me nahi."""
    from starlette.testclient import TestClient  # noqa: E402
    from rotator import app as app_module  # noqa: E402

    os.environ["GEMINI_KEYS"] = "gk1"
    os.environ["NVIDIA_KEYS"] = "nvapi-1"
    os.environ["ZEN_KEYS"] = "zk1"
    # GROQ / OPENROUTER keys NAHI — wo kahin nahi dikhne chahiye

    # fresh data dir — pehla registered user hi admin (bootstrap_admin)
    import shutil  # noqa: E402

    if os.path.exists("./test_data_live_models"):
        shutil.rmtree("./test_data_live_models")

    with TestClient(app_module.app) as client:
        r = client.post("/auth/register", json={"username": "keyed", "password": "pass1234"})
        tok = r.json()["token"]
        h = {"Authorization": "Bearer " + tok}

        data = client.get("/admin/models", headers=h).json()
        names = [p["name"] for p in data["providers"]]
        assert set(names) == {"gemini", "nvidia", "zen"}, f"expected sirf keyed providers, got {names}"
        assert set(data["catalog"].keys()) == {"gemini", "nvidia", "zen"}, data["catalog"]
        assert "groq" not in names and "openrouter" not in names, names
        assert "groq" not in data["catalog"] and "openrouter" not in data["catalog"]

        r = client.get("/v1/models", headers=h)
        owned = {m["owned_by"] for m in r.json()["data"]}
        assert owned == {"gemini", "nvidia", "zen"}, owned
    return True


def test_no_keys_no_providers_visible():
    """Bina kisi key ke — koi provider kahin bhi nahi dikhta (bina raw-config
    fallback ke — pehle PASTE_ placeholder wale 'key_count: 0' dikhte the)."""
    from starlette.testclient import TestClient  # noqa: E402
    from rotator import app as app_module  # noqa: E402

    # saare provider keys env se hata do (config me PASTE_ placeholders hain)
    for var in ("GEMINI_KEYS", "GROQ_KEYS", "OPENROUTER_KEYS", "NVIDIA_KEYS", "ZEN_KEYS"):
        os.environ.pop(var, None)

    # fresh data dir — pehla registered user hi admin (bootstrap_admin)
    import shutil  # noqa: E402

    if os.path.exists("./test_data_live_models"):
        shutil.rmtree("./test_data_live_models")

    with TestClient(app_module.app) as client:
        r = client.post("/auth/register", json={"username": "nokey", "password": "pass1234"})
        tok = r.json()["token"]
        h = {"Authorization": "Bearer " + tok}

        data = client.get("/admin/models", headers=h).json()
        assert data["providers"] == [], data["providers"]
        assert data["catalog"] == {}, data["catalog"]

        r = client.get("/v1/models", headers=h)
        assert r.json()["data"] == [], r.json()["data"]
    return True


if __name__ == "__main__":
    tests = [
        ("nvidia-style pagination dedup (no 5x duplicates)", test_nvidia_style_pagination_dedup),
        ("openai last_id pagination still multi-page", test_openai_last_id_pagination_still_works),
        ("nvidia provider builds from config.yaml", test_nvidia_provider_from_config),
        ("only key-configured providers visible (all tabs)", test_only_key_configured_providers_visible),
        ("no keys -> no providers visible anywhere", test_no_keys_no_providers_visible),
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
    if os.path.exists("./test_data_live_models"):
        shutil.rmtree("./test_data_live_models")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
