"""
test_tools.py — tool calling (function calling) support tests.

Covers: OpenAI message conversion, Gemini tools conversion, Gemini
functionCall extraction, and full passthrough through router+app.

Run:  python test_tools.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys

os.environ["SMARTROTATOR_DATA_DIR"] = "./test_data_tools"
os.environ["GITHUB_SYNC_ENABLED"] = "false"
if os.path.exists("./test_data_tools"):
    shutil.rmtree("./test_data_tools")

from starlette.testclient import TestClient  # noqa: E402

from rotator import app as app_module  # noqa: E402
from rotator.providers import (  # noqa: E402
    ChatMessage,
    ChatResult,
    GeminiProvider,
    OpenAICompatibleProvider,
)


def test_openai_message_conversion():
    # tool result message
    m = OpenAICompatibleProvider._to_openai_message(
        ChatMessage(role="tool", content='{"temp": 30}', tool_call_id="call_1", name="get_weather")
    )
    assert m == {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 30}'}, m

    # assistant with tool_calls
    calls = [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]
    m = OpenAICompatibleProvider._to_openai_message(
        ChatMessage(role="assistant", content="", tool_calls=calls)
    )
    assert m["role"] == "assistant" and m["tool_calls"] == calls, m

    # normal user message with images still works
    m = OpenAICompatibleProvider._to_openai_message(ChatMessage(role="user", content="hi"))
    assert m == {"role": "user", "content": "hi"}, m
    return True


def test_gemini_tools_conversion():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Mausam batata hai",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    out = GeminiProvider._to_gemini_tools(tools)
    decl = out[0]["functionDeclarations"][0]
    assert decl["name"] == "get_weather", decl
    assert decl["description"] == "Mausam batata hai", decl
    assert decl["parameters"]["properties"]["city"]["type"] == "string", decl
    return True


def test_gemini_contents_with_tool_roundtrip():
    messages = [
        ChatMessage(role="user", content="mausam batao"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[{"id": "c1", "type": "function",
                         "function": {"name": "get_weather", "arguments": '{"city": "delhi"}'}}],
        ),
        ChatMessage(role="tool", content='{"temp": 31}', tool_call_id="c1", name="get_weather"),
    ]
    contents, system = GeminiProvider._to_gemini_contents(messages)
    assert system == [], system
    assert contents[0]["role"] == "user", contents
    assert contents[1]["role"] == "model", contents
    fc = contents[1]["parts"][0]["functionCall"]
    assert fc["name"] == "get_weather" and fc["args"] == {"city": "delhi"}, fc
    fr = contents[2]["parts"][0]["functionResponse"]
    assert fr["name"] == "get_weather" and fr["response"]["result"] == {"temp": 31}, fr
    return True


def test_gemini_tool_calls_extraction():
    data = {
        "candidates": [{"content": {"parts": [
            {"functionCall": {"name": "get_weather", "args": {"city": "mumbai"}}}
        ]}}]
    }
    calls = GeminiProvider._extract_tool_calls(data)
    assert len(calls) == 1, calls
    assert calls[0]["type"] == "function", calls
    assert calls[0]["function"]["name"] == "get_weather", calls
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "mumbai"}, calls
    return True


def test_full_passthrough_through_app():
    """Fake provider jo tools echo karta hai — pura flow: HTTP → app → router → provider."""
    from rotator.core import KeyRing
    from rotator.providers import Provider
    from rotator.router import ProviderConfig, ProviderState, Rotator

    class EchoToolsProvider(Provider):
        name = "echo"

        def __init__(self, models):
            super().__init__(models)
            self.last_tools = None
            self.last_messages = None

        async def chat(self, messages, model, *, max_tokens=4096, temperature=0.7,
                       proxy=None, api_key=None, tools=None, tool_choice=None, **kwargs):
            self.last_tools = tools
            self.last_messages = messages
            return ChatResult(
                text="", provider=self.name, model=model, key_label="echo",
                tool_calls=[{"id": "call_echo", "type": "function",
                             "function": {"name": "get_weather", "arguments": '{"city": "delhi"}'}}],
            )

    app = app_module.app
    with TestClient(app) as client:
        # fake rotator install karo (real providers ke jagah sirf echo provider)
        real = Rotator(config_path="config.yaml")
        provider = EchoToolsProvider(["gemini-2.5-flash"])
        cfg = ProviderConfig(
            name="echo", ptype="openai", models=["gemini-2.5-flash"],
            keys=["test-key-1"], base_url="http://x",
        )
        ring = KeyRing(keys=cfg.keys, models=cfg.models, label="echo")
        real.providers = [ProviderState(cfg=cfg, ring=ring, provider=provider)]
        app.state.rotator = real

        r = client.post("/auth/register", json={"username": "tooler", "password": "tool1234"})
        tok = r.json()["token"]
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "x",
                                                   "parameters": {"type": "object", "properties": {}}}}]
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer " + tok},
            json={
                "messages": [{"role": "user", "content": "mausam batao"}],
                "tools": tools,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        msg = body["choices"][0]["message"]
        assert msg.get("tool_calls"), body
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather", body
        assert body["choices"][0]["finish_reason"] == "tool_calls", body
        # provider ko tools mili?
        assert provider.last_tools == tools, provider.last_tools
        # tool role roundtrip bhi check
        r2 = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer " + tok},
            json={
                "messages": [
                    {"role": "user", "content": "mausam batao"},
                    {"role": "assistant", "content": None,
                     "tool_calls": [{"id": "c1", "type": "function",
                                     "function": {"name": "get_weather", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": '{"temp": 31}'},
                ],
                "tools": tools,
            },
        )
        assert r2.status_code == 200, r2.text
        msgs = provider.last_messages
        assert msgs[1].role == "assistant" and msgs[1].tool_calls, msgs
        assert msgs[2].role == "tool" and msgs[2].tool_call_id == "c1" and msgs[2].name == "get_weather", msgs
        return True


class FakeResponse:
    def __init__(self, data):
        self._data = data
        self.status_code = 200
        self.text = json.dumps(data)

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class FakeClient:
    def __init__(self):
        self.last_payload = None

    async def post(self, url, headers=None, params=None, json=None, **kwargs):
        self.last_payload = json
        return FakeResponse(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}}
        )


def test_openai_web_search_strip_vs_passthrough():
    """OpenAI-compatible provider: default web_search strip, passthrough flag se keep."""
    web_tool = {"type": "web_search"}
    fn_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "x",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    # default (Groq/OpenRouter/zen): web_search strip hota hai, function tool jaata hai
    client = FakeClient()
    provider = OpenAICompatibleProvider("groq", "https://api.groq.com/openai/v1", ["llama-3.3-70b-versatile"])
    provider._client = client  # type: ignore[assignment]
    asyncio.run(
        provider.chat(
            [ChatMessage(role="user", content="hi")],
            "llama-3.3-70b-versatile",
            api_key="k",
            tools=[web_tool, fn_tool],
        )
    )
    sent_tools = client.last_payload["tools"]
    assert sent_tools == [fn_tool], sent_tools

    # passthrough: web_search tool upstream tak jaata hai
    client2 = FakeClient()
    provider2 = OpenAICompatibleProvider("openai", "https://api.openai.com/v1", ["gpt-5"], web_search_passthrough=True)
    provider2._client = client2  # type: ignore[assignment]
    asyncio.run(
        provider2.chat(
            [ChatMessage(role="user", content="hi")],
            "gpt-5",
            api_key="k",
            tools=[web_tool, fn_tool],
        )
    )
    sent_tools2 = client2.last_payload["tools"]
    assert sent_tools2 == [web_tool, fn_tool], sent_tools2
    return True


def test_gemini_web_search_grounding_tool():
    """Gemini provider: web_search tool -> google_search grounding (current shape)."""
    from rotator.providers import build_provider

    web_tool = {"type": "web_search"}
    fn_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "x",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    client = FakeClient()
    provider = build_provider("gemini", "gemini", None, ["gemini-2.5-flash"])
    provider._client = client  # type: ignore[assignment]
    asyncio.run(
        provider.chat(
            [ChatMessage(role="user", content="latest news?")],
            "gemini-2.5-flash",
            api_key="k",
            tools=[web_tool, fn_tool],
        )
    )
    body = client.last_payload
    # search tool -> google_search grounding; function tool -> functionDeclarations
    tools = body["tools"]
    assert {"google_search": {}} in tools, tools
    assert any("functionDeclarations" in t for t in tools), tools
    return True


def main() -> int:
    tests = [
        ("openai message conversion", test_openai_message_conversion),
        ("gemini tools conversion", test_gemini_tools_conversion),
        ("gemini contents tool roundtrip", test_gemini_contents_with_tool_roundtrip),
        ("gemini tool_calls extraction", test_gemini_tool_calls_extraction),
        ("openai web_search strip vs passthrough", test_openai_web_search_strip_vs_passthrough),
        ("gemini web_search grounding tool", test_gemini_web_search_grounding_tool),
        ("full passthrough via app", test_full_passthrough_through_app),
    ]
    passed = 0
    print(f"\n{'TEST':<35} {'RESULT':<10}")
    print("-" * 50)
    for name, fn in tests:
        try:
            ok = fn()
            mark = "✅ PASS" if ok else "❌ FAIL"
            if ok:
                passed += 1
        except Exception as exc:  # noqa: BLE001
            mark = f"❌ FAIL ({exc.__class__.__name__}: {exc})"
        print(f"{name:<35} {mark}")
    print("-" * 50)
    print(f"{passed}/{len(tests)} passed")
    if os.path.exists("test_tools.db"):
        os.remove("test_tools.db")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
