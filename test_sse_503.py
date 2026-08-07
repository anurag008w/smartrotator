"""
test_sse_503.py — verifies `stream: true` pe provider error ab generic
"SSE HTTP 503" ki jagah OpenAI-style SSE error frame deta hai, aur
non-stream path pe purana 503 JSON behaviour preserve rehta hai.

Run: python test_sse_503.py
"""
from __future__ import annotations

import json
import os
import shutil

os.environ["SMARTROTATOR_DATA_DIR"] = "./test_data_sse503"
os.environ["GITHUB_SYNC_ENABLED"] = "false"
if os.path.exists("./test_data_sse503"):
    shutil.rmtree("./test_data_sse503")

from starlette.testclient import TestClient  # noqa: E402

from rotator import app as app_module  # noqa: E402
from rotator.core import KeyRing  # noqa: E402
from rotator.providers import AllProvidersExhausted, ChatResult, Provider  # noqa: E402
from rotator.router import ProviderConfig, ProviderState, Rotator  # noqa: E402


class FailingProvider(Provider):
    """Fake provider — har call pe rate-limit error throw karta hai."""

    def __init__(self, name, models):
        super().__init__(models)
        self.name = name

    async def chat(self, messages, model, *, max_tokens=4096, temperature=0.7,
                    proxy=None, api_key=None, tools=None, tool_choice=None, **kwargs):
        raise AllProvidersExhausted(f"{self.name} key exhausted")


def _setup_rotator_with_failing():
    real = Rotator(config_path="config.yaml")
    provider = FailingProvider("gemini", ["gemini-2.5-flash"])
    cfg = ProviderConfig(
        name="gemini", ptype="gemini", models=["gemini-2.5-flash"],
        keys=["test-key-1"], base_url="http://x",
    )
    ring = KeyRing(keys=cfg.keys, models=cfg.models, label="gemini")
    st = ProviderState(cfg=cfg, ring=ring, provider=provider)
    real.providers = [st]
    return real


def _register(client, username):
    r = client.post("/auth/register", json={"username": username, "password": "pass1234"})
    return r.json()["token"]


def _parse_sse(raw_text: str) -> list[dict]:
    frames = [f for f in raw_text.split("\n\n") if f.strip()]
    out = []
    for f in frames:
        for line in f.split("\n"):
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    out.append({"done": True})
                else:
                    out.append(json.loads(payload))
    return out


def test_stream_error_is_sse_not_503():
    """stream:true + provider fail -> 200 SSE error frame (LevelUp ab generic
    'SSE HTTP 503' nahi dikhayega, actual message milega)."""
    app = app_module.app
    with TestClient(app) as client:
        app.state.rotator = _setup_rotator_with_failing()
        tok = _register(client, "sse503a")
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer " + tok},
            json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/event-stream"), r.headers
        frames = _parse_sse(r.text)
        assert frames, "koi SSE frame nahi mila!"
        assert frames[-1] == {"done": True}, frames
        err = frames[0].get("error", {})
        assert err.get("message"), frames
        assert err.get("code") == 503, err
        assert "providers" in err["message"].lower(), err["message"]
        return True


def test_non_stream_still_503_json():
    """non-stream + provider fail -> purana 503 JSON (regression guard)."""
    app = app_module.app
    with TestClient(app) as client:
        app.state.rotator = _setup_rotator_with_failing()
        tok = _register(client, "sse503b")
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer " + tok},
            json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 503, r.text
        assert r.headers["content-type"].startswith("application/json"), r.headers
        assert "providers" in r.json()["detail"].lower(), r.json()
        return True


def test_stream_rate_limit_is_sse_too():
    """RateLimitError pe bhi stream:true -> SSE error frame, 503 JSON nahi."""
    from rotator.providers import RateLimitError

    class RateLimitedProvider(Provider):
        def __init__(self, name, models):
            super().__init__(models)
            self.name = name

        async def chat(self, messages, model, *, max_tokens=4096, temperature=0.7,
                        proxy=None, api_key=None, tools=None, tool_choice=None, **kwargs):
            raise RateLimitError(f"{self.name} 429")

    app = app_module.app
    with TestClient(app) as client:
        real = Rotator(config_path="config.yaml")
        provider = RateLimitedProvider("gemini", ["gemini-2.5-flash"])
        cfg = ProviderConfig(
            name="gemini", ptype="gemini", models=["gemini-2.5-flash"],
            keys=["test-key-1"], base_url="http://x",
        )
        ring = KeyRing(keys=cfg.keys, models=cfg.models, label="gemini")
        real.providers = [ProviderState(cfg=cfg, ring=ring, provider=provider)]
        app.state.rotator = real
        tok = _register(client, "sse503c")
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer " + tok},
            json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/event-stream"), r.headers
        frames = _parse_sse(r.text)
        err = frames[0].get("error", {})
        assert err.get("code") == 503, err
        assert frames[-1] == {"done": True}, frames
        return True


if __name__ == "__main__":
    tests = [
        ("stream + all exhausted -> SSE error frame", test_stream_error_is_sse_not_503),
        ("non-stream + all exhausted -> 503 JSON (unchanged)", test_non_stream_still_503_json),
        ("stream + rate limited -> SSE error frame", test_stream_rate_limit_is_sse_too),
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
    if os.path.exists("./test_data_sse503"):
        shutil.rmtree("./test_data_sse503")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
