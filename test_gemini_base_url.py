"""
test_gemini_base_url.py — verifies custom Gemini base_url (Cloudflare Worker
gateway / proxy) ab provider tak pahunchta hai, aur default bhi kaam karta hai.

Run: python test_gemini_base_url.py
"""
from __future__ import annotations

import os
import shutil

os.environ["SMARTROTATOR_DATA_DIR"] = "./test_data_baseurl"
os.environ["GITHUB_SYNC_ENABLED"] = "false"
if os.path.exists("./test_data_baseurl"):
    shutil.rmtree("./test_data_baseurl")

from rotator.providers import (  # noqa: E402
    GEMINI_V1,
    ChatMessage,
    GeminiProvider,
    build_provider,
)


def test_build_provider_custom_base_url():
    p = build_provider("gemini", "gemini", "https://my-worker.workers.dev/v1beta", ["gemini-2.5-flash"])
    assert isinstance(p, GeminiProvider), type(p)
    assert p.base_url == "https://my-worker.workers.dev/v1beta", p.base_url
    return True


def test_build_provider_default_base_url():
    p = build_provider("gemini", "gemini", None, ["gemini-2.5-flash"])
    assert isinstance(p, GeminiProvider), type(p)
    assert p.base_url == GEMINI_V1, p.base_url
    return True


def test_gemini_chat_url_uses_custom_base():
    """chat call custom base_url se hit kare (endpoint build check)."""
    import json

    import httpx

    captured = {}

    class FakeResp:
        status_code = 200
        text = "{}"

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "candidates": [{"content": {"parts": [{"text": "hi from worker"}]}}],
                "usageMetadata": {"totalTokenCount": 3},
            }

    class FakeClient:
        async def post(self, url, headers=None, params=None, json=None):
            captured["url"] = url
            captured["params"] = params
            return FakeResp()

    p = build_provider("gemini", "gemini", "https://my-worker.workers.dev/v1beta", ["gemini-2.5-flash"])
    p._client = FakeClient()

    import asyncio

    async def run():
        return await p.chat(
            [ChatMessage(role="user", content="hi")], "gemini-2.5-flash", api_key="KEY"
        )

    res = asyncio.run(run())
    assert captured["url"] == "https://my-worker.workers.dev/v1beta/models/gemini-2.5-flash:generateContent", captured
    assert captured["params"] == {"key": "KEY"}, captured
    assert "hi from worker" in res.text
    return True


def test_openai_type_custom_base_url():
    """OpenAI-type providers (groq/openrouter/nvidia/zen) base_url already
    config se aata hai — custom gateway URL bhi directly chalta hai."""
    from rotator.providers import OpenAICompatibleProvider

    p = build_provider("nvidia", "openai", "https://my-worker.workers.dev/v1", ["meta/llama-3.3-70b-instruct"])
    assert isinstance(p, OpenAICompatibleProvider), type(p)
    assert p.base_url == "https://my-worker.workers.dev/v1", p.base_url
    assert p.endpoint == "https://my-worker.workers.dev/v1/chat/completions", p.endpoint
    return True


if __name__ == "__main__":
    tests = [
        ("custom base_url -> provider", test_build_provider_custom_base_url),
        ("no base_url -> default GEMINI_V1", test_build_provider_default_base_url),
        ("chat call custom base_url se", test_gemini_chat_url_uses_custom_base),
        ("openai-type custom base_url", test_openai_type_custom_base_url),
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
    if os.path.exists("./test_data_baseurl"):
        shutil.rmtree("./test_data_baseurl")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
