"""
test_live_endpoint.py — Gemini Live API proxy tests.

Covers:
  1. is_live_model() live/tts model detection
  2. _extract_setup() setup message parsing (string / bytes / dict)
  3. relay_live_session() transparent bidirectional relay (mock upstream)
  4. Relay: invalid setup → error, valid setup → downstream connect

Run: python test_live_endpoint.py
"""
from __future__ import annotations

import asyncio
import json

from rotator import live as live_proxy


def test_live_model_detection():
    cases = {
        # (model, is_live)
        "gemini-3.1-flash-live-preview": True,
        "gemini-3.1-flash-tts-preview": True,
        "gemini-2.5-flash-native-audio": True,
        "gemini-2.0-flash-live-001": True,
        "gemini-3.5-flash": False,
        "gemini-2.5-pro": False,
        "models/gemini-3.1-flash-live-preview": True,  # 'models/' prefix bhi
        "meta/llama-4-maverick:free": False,
        "big-pickle": False,
        "": False,
    }
    for model, expected in cases.items():
        got = live_proxy.is_live_model(model)
        assert got is expected, f"{model!r}: expected {expected}, got {got}"
    return True


def test_extract_setup_valid():
    raw = json.dumps(
        {
            "setup": {
                "model": "models/gemini-3.1-flash-live-preview",
                "generation_config": {"response_modalities": ["AUDIO"]},
                "system_instruction": {"parts": [{"text": "hi"}]},
            }
        }
    )
    ok, model, decoded = live_proxy._extract_setup(raw)
    assert ok is True, f"valid setup rejected: {ok}"
    assert model == "gemini-3.1-flash-live-preview", model
    # decoded valid JSON hai aur setup retain karta hai
    assert json.loads(decoded)["setup"]["model"] == "models/gemini-3.1-flash-live-preview"
    return True


def test_extract_setup_bytes():
    raw = b'{"setup":{"model":"models/gemini-3.1-flash-live-preview"}}'
    ok, model, _decoded = live_proxy._extract_setup(raw)
    assert ok is True and model == "gemini-3.1-flash-live-preview"
    return True


def test_extract_setup_invalid():
    # non-json
    ok, _m, _d = live_proxy._extract_setup("not json at all")
    assert ok is False
    # json but no setup
    ok, _m, _d = live_proxy._extract_setup('{"foo":1}')
    assert ok is False
    # setup but no model
    ok, _m, _d = live_proxy._extract_setup('{"setup":{"generation_config":{"response_modalities":["AUDIO"]}}}')
    assert ok is False
    return True


# --------------------------------------------------------------------------
# Relay test — mock upstream websocket
# --------------------------------------------------------------------------
class MockUpstream:
    """websockets.connect() ka fake — messages collect karta hai aur scripted
    replies bhejta hai."""

    def __init__(self, responses):
        self.sent = []
        self._responses = list(responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, data):
        self.sent.append(data)

    async def __aiter__(self):
        for r in self._responses:
            if isinstance(r, str):
                yield r
            else:
                yield bytes(r)
        return


async def _fake_connect(*args, **kwargs):
    global _mock
    return _mock


class Collector:
    """client_send ke liye — saare messages collect karta hai."""

    def __init__(self):
        self.messages = []

    async def __call__(self, data):
        self.messages.append(data)

    def join(self):
        return "\n".join(
            m.decode() if isinstance(m, bytes) else str(m) for m in self.messages
        )


async def _make_client_stream(items):
    """client_recv iterator — scripted client messages yield karta hai."""
    for it in items:
        yield it


def test_relay_basic_flow():
    """Valid setup → client frames Google ko jayein, Google ke responses
    client ko."""
    global _mock
    _mock = MockUpstream(
        responses=[
            json.dumps({"setupComplete": {}}),
            json.dumps({"serverContent": {"modelTurn": {"parts": [{"inlineData": {"data": "AQID", "mimeType": "audio/pcm"}}]}}}),
        ]
    )
    live_proxy.websockets.connect = _fake_connect

    async def run():
        collector = Collector()
        client_items = [
            json.dumps({"setup": {"model": "models/gemini-3.1-flash-live-preview"}}),
            json.dumps({"realtimeInput": {"audio": {"data": "AAEC"}}}),
            json.dumps({"clientContent": {"turns": [{"role": "user", "parts": [{"text": "hi"}]}]}}),
        ]
        stream = _make_client_stream(client_items)
        stats = await live_proxy.relay_live_session(
            collector,
            stream.__aiter__(),
            gemini_key="test-key",
        )
        return collector, stats

    collector, stats = asyncio.run(run())
    assert stats.get("ok") is True, stats
    assert stats.get("setup_model") == "gemini-3.1-flash-live-preview", stats

    out = collector.join()
    assert '"setupComplete"' in out, f"setupComplete client ko nahi aayi: {out}"
    assert '"audio/pcm"' in out, f"audio response client ko nahi aayi: {out}"

    # upstream ko client ke setup + realtimeInput + clientContent milna chahiye
    up_messages = _mock.sent
    assert len(up_messages) >= 3, f"upstream messages kam: {len(up_messages)}"
    assert "realtimeInput" in up_messages[1]
    assert "clientContent" in up_messages[2]
    return True


def test_relay_invalid_setup_no_upstream():
    """Invalid pahla message → error client ko, upstream connect nahi hona
    chahiye."""

    global _mock
    _mock = MockUpstream(responses=[])
    live_proxy.websockets.connect = _fake_connect

    async def run():
        collector = Collector()
        stream = _make_client_stream(["gibberish not json"])
        stats = await live_proxy.relay_live_session(
            collector,
            stream.__aiter__(),
            gemini_key="test-key",
        )
        return collector, stats

    collector, stats = asyncio.run(run())
    assert stats.get("ok") is False, stats
    assert stats.get("reason") == "invalid_setup", stats
    assert "INVALID_SETUP" in collector.join()
    return True


if __name__ == "__main__":
    tests = [
        ("live model detection", test_live_model_detection),
        ("setup parse valid", test_extract_setup_valid),
        ("setup parse bytes", test_extract_setup_bytes),
        ("setup parse invalid", test_extract_setup_invalid),
        ("relay basic flow", test_relay_basic_flow),
        ("relay invalid setup no upstream", test_relay_invalid_setup_no_upstream),
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
            import traceback
            traceback.print_exc()
            print(f"ERROR {label}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
