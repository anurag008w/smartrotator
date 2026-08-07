"""
test_streaming_fix.py — verifies `stream: true` ab real SSE deta hai,
normal model ke liye AND virtual model group ke liye dono.

Run: python test_streaming_fix.py
"""
from __future__ import annotations

import os
import shutil

os.environ["SMARTROTATOR_DATA_DIR"] = "./test_data_streaming"
os.environ["GITHUB_SYNC_ENABLED"] = "false"
if os.path.exists("./test_data_streaming"):
    shutil.rmtree("./test_data_streaming")

from starlette.testclient import TestClient  # noqa: E402

from rotator import app as app_module  # noqa: E402
from rotator.core import KeyRing  # noqa: E402
from rotator.providers import ChatResult, Provider  # noqa: E402
from rotator.router import ProviderConfig, ProviderState, Rotator  # noqa: E402


class EchoProvider(Provider):
    """Fake provider — real Gemini/Groq ko hit kiye bina response deta hai."""

    def __init__(self, name, models):
        super().__init__(models)
        self.name = name

    async def chat(self, messages, model, *, max_tokens=4096, temperature=0.7,
                    proxy=None, api_key=None, tools=None, tool_choice=None, **kwargs):
        return ChatResult(
            text=f"hello from {self.name}/{model} this is a longer reply to test chunking",
            provider=self.name, model=model, key_label=self.name,
        )


def _setup_rotator_with_group():
    real = Rotator(config_path="config.yaml")
    provider = EchoProvider("gemini", ["gemini-2.5-flash"])
    cfg = ProviderConfig(
        name="gemini", ptype="gemini", models=["gemini-2.5-flash"],
        keys=["test-key-1", "test-key-2"], base_url="http://x",
    )
    ring = KeyRing(keys=cfg.keys, models=cfg.models, label="gemini")
    st = ProviderState(cfg=cfg, ring=ring, provider=provider)
    real.providers = [st]
    # ek virtual group "levelup" banao, jaise dashboard banata hai
    real.apply_managed({
        "groups": [
            {
                "id": "levelup",
                "label": "LevelUp",
                "enabled": True,
                "members": [
                    {"provider": "gemini", "models": ["gemini-2.5-flash"], "keys": [0, 1]},
                ],
            }
        ]
    })
    return real


def _parse_sse(raw_text: str) -> list[dict]:
    import json
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


def test_stream_normal_model():
    app = app_module.app
    with TestClient(app) as client:
        app.state.rotator = _setup_rotator_with_group()
        r = client.post("/auth/register", json={"username": "streamer1", "password": "pass1234"})
        tok = r.json()["token"]

        assert client.headers.get("content-type") is None  # sanity noop
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
        # role chunk
        assert frames[0]["choices"][0]["delta"].get("role") == "assistant", frames[0]
        # content reconstruct karke poora text milna chahiye
        full = "".join(
            fr["choices"][0]["delta"].get("content", "")
            for fr in frames if "choices" in fr
        )
        assert "hello from gemini/gemini-2.5-flash" in full, full
        # finish_reason wala chunk aana chahiye
        finish_reasons = [fr["choices"][0].get("finish_reason") for fr in frames if "choices" in fr]
        assert "stop" in finish_reasons, finish_reasons
        return True


def test_stream_virtual_group_model():
    app = app_module.app
    with TestClient(app) as client:
        app.state.rotator = _setup_rotator_with_group()
        r = client.post("/auth/register", json={"username": "streamer2", "password": "pass1234"})
        tok = r.json()["token"]

        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer " + tok},
            json={"model": "levelup", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/event-stream"), r.headers
        frames = _parse_sse(r.text)
        assert frames[-1] == {"done": True}, frames
        full = "".join(
            fr["choices"][0]["delta"].get("content", "")
            for fr in frames if "choices" in fr
        )
        assert "hello from gemini/gemini-2.5-flash" in full, full
        return True


def test_non_stream_unaffected():
    """Purana non-stream behaviour bilkul same rehna chahiye."""
    app = app_module.app
    with TestClient(app) as client:
        app.state.rotator = _setup_rotator_with_group()
        r = client.post("/auth/register", json={"username": "streamer3", "password": "pass1234"})
        tok = r.json()["token"]
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer " + tok},
            json={"model": "levelup", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("application/json"), r.headers
        body = r.json()
        assert "hello from gemini/gemini-2.5-flash" in body["choices"][0]["message"]["content"], body
        return True


if __name__ == "__main__":
    tests = [
        ("stream: normal model -> real SSE", test_stream_normal_model),
        ("stream: virtual group model -> real SSE", test_stream_virtual_group_model),
        ("non-stream unaffected", test_non_stream_unaffected),
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
    if os.path.exists("./test_data_streaming"):
        shutil.rmtree("./test_data_streaming")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
