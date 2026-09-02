"""
live.py — Gemini Live API (BidiGenerateContent) WebSocket proxy.

SmartRotator ke text models (`generateContent`) aur Gemini ke LIVE models
(`gemini-*-live-preview`, `gemini-*-tts-*`) alag hoti hain. Live models sirf
Google ke stateful WebSocket API (`BidiGenerateContent`) se chalti hain —
unary REST `generateContent` unhe support nahi karta. Isliye aapke dusre
apps me "model aa raha hai par kaam nahi kar raha" wala problem ata tha.

Yeh module ek generic bidirectional WebSocket proxy provide karta hai:
client → SmartRotator → Google Gemini Live API.

Flow:
  client (wss://host/v1/live?key=USER_KEY&model=gemini-3.1-flash-live-preview)
        ──setup──▶ SmartRotator (auth + quota + apni gemini key pin)
        ──frames──▶ Google Live API (wss://generativelanguage.googleapis.com/...)
        ◀──frames── SmartRotator (relay) ◀── Google

SmartRotator apni (rotated) gemini key user ke liye inject karta hai, aur
saare client ↔ Google frames transparently relay karta hai. Quota per-session
reserve hota hai (auth/chat-completions jaisa hi).

Note: Google ke Live API se "server-to-server" approach docs recommend karte
hain — client apna audio/text web-socket me bhejta hai, SmartRotator usse
Google tak relay karta hai. Isliye raw audio passthrough kisi conversion ke
bina chalta hai (Google raw PCM 16kHz accept karta hai, same base64 JSON).

Gemini Live API raw WebSocket JSON protobuf schema (snake_case) use karta hai:

  Client setup (pahla message):
    {"setup": {
        "model": "models/gemini-3.1-flash-live-preview",
        "generation_config": {"response_modalities": ["AUDIO"]},
        "system_instruction": {"parts": [{"text": "..."}]},
        "input_audio_transcription": {},
        "output_audio_transcription": {},
        "tools": [...]
    }}

  Client frames:
    {"realtime_input": {"audio": {"data": "<base64 PCM16 16kHz>", "mime_type": "..."}, "text": "...", "video": {...}}}
    {"client_content": {"turns": [{"role": "user", "parts": [...]}], "turn_complete": true}}
    {"tool_response": {"function_responses": [{"id": "...", "name": "...", "response": {...}}]}}

  Server frames:
    {"setup_complete": {}}
    {"server_content": {"turn_complete": true, "interrupted": false, "model_turn": {"parts": [...]}}}
    {"tool_call": {"function_calls": [{"id": "...", "name": "...", "args": {...}}]}}
    {"go_away": {"reconnect_time": 100, "timeout": 100}}

Yeh module in sabko transparent passthrough karta hai.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional, Tuple

import websockets

logger = logging.getLogger("smartrotator.live")

# Google Gemini Live API — BidiGenerateContent raw WebSocket endpoint (v1beta).
# Authentication `?key=` query param se hoti hai (raw WS docs ke mutabik).
LIVE_WS_BASE = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

# Jin model names ko "live" maane —- normal text models wo nahi chalate.
# Regex covers: gemini-3.1-flash-live-preview, gemini-*-tts-preview,
# gemini-live-*, *-live-preview, native-audio, etc.
_LIVE_MODEL_RE = re.compile(
    r"(live|tts|native[-_]audio|realtime|voice|speech|maritalk)", re.IGNORECASE
)


def is_live_model(model_id: str) -> bool:
    """Kya yeh model Gemini Live API (WebSocket) require karta hai?"""
    if not model_id:
        return False
    if "/" in model_id:
        model_id = model_id.rsplit("/", 1)[-1]
    return bool(_LIVE_MODEL_RE.search(model_id))


def _normalize_model(model_id: str) -> str:
    """model hisse se 'models/' prefix hatao aur host forms normalise karo."""
    m = (model_id or "").strip()
    if m.startswith("models/"):
        m = m[len("models/"):]
    return m


def pick_gemini_live_key(rotator) -> Optional[Tuple[str, str]]:
    """SmartRotator ke configured providers me se pehla ENABLED gemini provider
    dhoondo, aur uski KeyRing se ek available key pick karo.

    Returns (api_key, key_label) ya (None, None) agar koi gemini key available
    nahi. Key pick hone se rotation progress hogi (KeyRing.pick() state update
    karta hai) — isliye ise sirf successful connect pe hi call karo.
    """
    for st in rotator.providers:
        if st.cfg.ptype != "gemini" or not st.cfg.keys:
            continue
        picked = st.ring.pick(st.cfg.models)
        if picked is None:
            continue
        state, _model = picked
        st.ring.report_success(state, None)  # connection try karna — fail aware
        return state.key, state.label
    return None, None


async def relay_live_session(
    client_send,    # async callable: (str | bytes) -> None  (client ko bhejo)
    client_recv,    # async iterator: client se JSON string/bytes aayengi
    *,
    gemini_key: str,
    heartbeat_interval: float = 20.0,
) -> dict:
    """Gemini Live API session proxy — transparent bidirectional relay.

    Pure proxy: client ke frames (setup/realtimeInput/clientContent/toolResponse)
    seedha Google ko bhejte hain, aur Google ke server frames (setupComplete/
    serverContent/audio) seedha client ko. SmartRotator sirf ek apni gemini key
    inject karta hai (Google connection ke URL me `?key=`).

    Params:
      client_send: client WebSocket pe message bhejne ka async callable.
      client_recv: client se messages de raha async iterator
                   (har item str ya bytes — text/base64 JSON).
      gemini_key:  SmartRotator ki gemini API key.

    Returns session stats dict.
    """
    # 1) client ka pehla message = BidiGenerateContentSetup (required).
    #    Hame ise Google ko RE-route karne se pehle model validate karne ke
    #    liye parse karna hota hai. Har client message raw relay hoga.
    first = await client_recv.__anext__()
    setup_ok, setup_model, decoded_setup = _extract_setup(first)
    if not setup_ok or not setup_model:
        if not setup_ok:
            await client_send(
                json.dumps(
                    {
                        "error": {
                            "code": "INVALID_SETUP",
                            "message": (
                                "Pahla message valid BidiGenerateContent setup hona "
                                "chahiye: {\"setup\": {\"model\": \"models/...\"}}"
                            ),
                        }
                    }
                )
            )
        # invalid setup ho toh bina upstream connection ke close karo
        return {"ok": False, "reason": "invalid_setup"}

    # 2) Google Live API se connect karo (apni key ke saath)
    ws_url = f"{LIVE_WS_BASE}?key={gemini_key}"
    try:
        upstream = await websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=20,
            max_size=64 * 1024 * 1024,  # bade audio frames ke liye
            open_timeout=20,
            close_timeout=10,
        )
    except Exception as exc:  # websockets.InvalidURI / InvalidStatus / timeout ...
        logger.warning("live: upstream connect failed: %r", exc)
        try:
            await client_send(
                json.dumps(
                    {
                        "error": {
                            "code": "UPSTREAM_CONNECT_FAILED",
                            "message": f"Gemini Live API se connect nahi hua: {exc}",
                        }
                    }
                )
            )
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "reason": "upstream_connect_failed", "error": str(exc)}

    try:
        # 3) client ka setup Google ko bhejo (normalized model + bina badlaav)
        await upstream.send(decoded_setup)

        # 4) bidirectional relay — do tasks aage-piche frames copy karte hain.
        client_ok = asyncio.Event()
        upstream_ok = asyncio.Event()

        async def client_to_upstream():
            try:
                async for msg in client_recv:
                    if isinstance(msg, (bytes, bytearray)):
                        await upstream.send(bytes(msg))
                    else:
                        await upstream.send(str(msg))
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            finally:
                client_ok.set()

        async def upstream_to_client():
            try:
                async for msg in upstream:
                    if isinstance(msg, (bytes, bytearray)):
                        await client_send(bytes(msg))
                    else:
                        await client_send(str(msg))
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            finally:
                upstream_ok.set()

        relay_client = asyncio.create_task(client_to_upstream())
        relay_upstream = asyncio.create_task(upstream_to_client())

        # 5) session tab tak live rahe jab tak dono taraf se aana band na ho
        #    (ya ek taraf error/close ho jaye). Jab client band ho, upstream
        #    bhi band karo.
        done, _pending = await asyncio.wait(
            {relay_client, relay_upstream},
            return_when=asyncio.FIRST_COMPLETED,
        )
        # jise task finish hua usme exception ho sakta hai — catch karo
        for task in done:
            exc = task.exception()
            if exc:
                logger.debug("live: relay task error: %r", exc)

        # cleanup: doosre relay task ko cancel karo
        for task in (relay_client, relay_upstream):
            if not task.done():
                task.cancel()
        await asyncio.gather(relay_client, relay_upstream, return_exceptions=True)

        return {"ok": True, "setup_model": setup_model}

    finally:
        try:
            await upstream.close()
        except Exception:  # noqa: BLE001
            pass


def _extract_setup(raw) -> Tuple[bool, str, Optional[str]]:
    """Client ke pehle message se setup model nikaalo (aur normalized raw text
    bhi — Google ko wahi bhejna). Raw string ho ya dict.

    Returns: (is_valid_setup, model_id, decoded_setup_json_string)
    is_valid_setup False → model_id "" aur decoded None.
    """
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        if isinstance(raw, str):
            obj = json.loads(raw)
        elif isinstance(raw, dict):
            obj = raw
        else:
            return False, "", None
    except (json.JSONDecodeError, ValueError):
        return False, "", None

    setup = obj.get("setup")
    if not isinstance(setup, dict):
        return False, "", None
    model = _normalize_model(setup.get("model", ""))
    if not model:
        return False, "", None

    # normalized + valid JSON string — Google ko wahi bhejo
    try:
        decoded = json.dumps(obj, ensure_ascii=False)
    except TypeError:
        return False, "", None
    return True, model, decoded
