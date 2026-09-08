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
import os
import re
from typing import Optional, Tuple

import websockets

try:  # FastAPI stream ka WebSocketDisconnect — relay me expected disconnect
    from fastapi import WebSocketDisconnect
except ImportError:  # pragma: no cover — standalone/test bina FastAPI ke
    class WebSocketDisconnect(Exception):  # type: ignore[no-redef]
        def __init__(self, code: int = 1000, reason: str = "", **kwargs):  # noqa: N802
            super().__init__(reason)
            self.code = code
            self.reason = reason

logger = logging.getLogger("smartrotator.live")

# Google Gemini Live API — BidiGenerateContent raw WebSocket endpoint (v1beta).
# Authentication `?key=` query param se hoti hai (raw WS docs ke mutabik).
#
# UPSTREAM OVERRIDE: SmartRotator ki Google keys hamesha iske paas rehti hain,
# par live ka network path ek Cloudflare Worker (GEMINI_LIVE_UPSTREAM) ho sakta
# hai. SmartRotator apne GEMINI_KEYS pool se key pick karke is URL me `?key=`
# ke saath bhejta hai; CF worker transparently Google se connect karta hai
# (key client ko nahi dikhti). Default: seedha Google ka endpoint.
LIVE_WS_BASE = os.environ.get(
    "GEMINI_LIVE_UPSTREAM",
    (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    ),
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


# Native-audio live models (gemini-3.1-flash-live-preview, gemini-2.5-flash-
# native-audio-*) sirf response_modalities=["AUDIO"] accept karte hain (docs +
# python-genai issue #2238: TEXT pe 1011 Internal Error, TEXT+AUDIO pe
# invalid-argument). NOTE: `gemini-live-2.5-flash-preview` TEXT support karta
# hai — isliye sirf suffix `-live-preview` / `native-audio` wale normalize
# karte hain, generic `live` substring nahi.
_NATIVE_AUDIO_RE = re.compile(
    r"(?:native[-_]audio|[-_]live[-_]preview$)", re.IGNORECASE
)

# Gemini TTS models Live API (BidiGenerateContent) me support NAHI hote —
# wo unary/streaming REST generateContent (speech_config) use karte hain.
_TTS_LIVE_UNSUPPORTED = re.compile(r"tts", re.IGNORECASE)


def normalize_live_setup(setup_json: str, model_id: str) -> tuple[str, Optional[str]]:
    """Google ke live models ke liye setup message normalize karo.

    Problem (research-confirmed):
      - `gemini-3.1-flash-live-preview` (native audio) sirf
        `response_modalities: ["AUDIO"]` accept karta hai.
      - `["TEXT"]` bhejo → 1011 Internal Error / connection close.
      - `["TEXT", "AUDIO"]` dono → invalid argument / close.
      - Text response ke liye official workaround: `["AUDIO"]` +
        `output_audio_transcription: {}` — phir text `server_content.
        output_transcription.text` me aata hai.

    Isliye jab client TEXT ya TEXT+AUDIO maange, hum:
      1. modalities ko [`AUDIO`] pe force karte hain
      2. `output_audio_transcription` add karte hain (agar missing ho)
         → client ko audio + text dono milte hain, connection nahi tootta.

    TTS models Live API me support nahi karte — clear error dete hain.

    Returns: (normalized_setup_json, error_message_or_None)
    """
    try:
        obj = json.loads(setup_json)
        setup = obj.get("setup")
        if not isinstance(setup, dict):
            return setup_json, None  # pehle _extract_setup verify kar chuka hai

        model = _normalize_model(setup.get("model", "") or model_id)

        # TTS — Live API unsupported (REST speech_config chahiye)
        if _TTS_LIVE_UNSUPPORTED.search(model):
            return setup_json, (
                f"'{model}' Gemini TTS model hai — Live API (BidiGenerateContent) "
                "ise support nahi karta. TTS ke liye unary/streaming "
                "generateContent + speech_config use karo (SmartRotator ke normal "
                "REST routes)."
            )

        # Sirf native-audio live models pe normalize chahiye
        if not _NATIVE_AUDIO_RE.search(model):
            return setup_json, None  # text-capable live models — as-is pass

        gen_cfg = setup.get("generation_config")
        if not isinstance(gen_cfg, dict):
            gen_cfg = {}
            setup["generation_config"] = gen_cfg

        modalities = gen_cfg.get("response_modalities")
        # JSON string bhi ho sakta hai ("response_modalities": "TEXT") —
        # `list("TEXT")` = ['T','E','X','T'] galat hota, pehle wrap karo.
        if isinstance(modalities, str):
            modalities = [modalities]
        elif not isinstance(modalities, list):
            modalities = list(modalities) if modalities else []

        norm_modalities = [str(m).upper() for m in modalities]

        # Client ne TEXT chaha (TEXT ya TEXT+AUDIO) → AUDIO + transcription
        if "TEXT" in norm_modalities:
            gen_cfg["response_modalities"] = ["AUDIO"]
            if "output_audio_transcription" not in setup:
                setup["output_audio_transcription"] = {}
        elif not norm_modalities:
            # koi modality specify nahi — native audio default AUDIO (docs)
            gen_cfg["response_modalities"] = ["AUDIO"]

        # client ko transparent rakho — normalized JSON string
        return json.dumps(obj, ensure_ascii=False), None
    except (ValueError, TypeError, json.JSONDecodeError):
        return setup_json, None


def _normalize_model(model_id: str) -> str:
    """model hisse se 'models/' prefix hatao aur host forms normalise karo."""
    m = (model_id or "").strip()
    if m.startswith("models/"):
        m = m[len("models/"):]
    return m


def pick_gemini_live_key(rotator) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[object], Optional[object]]:
    """SmartRotator ke configured providers me se pehla ENABLED gemini provider
    dhoondo, aur uski KeyRing se ek available key pick karo.

    Returns (api_key, key_label, live_upstream, ring, state) sab None ho toh
    koi gemini key available nahi. Key pick hone se rotation progress hogi
    (KeyRing.pick() state update karta hai) — isliye ise sirf successful
    connect pe hi call karo.

    `live_upstream`: us key ka per-key base_url (key_base_urls) hai toh CF
    worker /v1/live URL banakar deta hai (wss://<worker>/v1/live) — isse live
    bhi usi key ke network/egress se jata hai (1 key = 1 URL design). Agar
    per-key URL nahi hai toh None — relay server Google direct use karega.

    `ring`/`state` caller ko return hote hain taaki success/failure report
    ACTUAL outcome ke baad ho (connect fail hone pe bhi success report karna
    galat hai — pehle yahan report_success premature tha).
    """
    for st in rotator.providers:
        if st.cfg.ptype != "gemini" or not st.cfg.keys:
            continue
        picked = st.ring.pick(st.cfg.models)
        if picked is None:
            continue
        state, _model = picked
        # NOTE: report_success/report_failure YAHAN NAHI — caller relay ke
        # baad actual outcome ke aadhar pe report karta hai (audit fix).
        # per-key base_url → CF worker live WS URL (agar workers.dev hai)
        per_key_url = st.cfg.key_base_urls.get(state.key, "") or ""
        live_upstream = _live_ws_from_base_url(per_key_url)
        return state.key, state.label, live_upstream, st.ring, state
    return None, None, None, None, None


def _live_ws_from_base_url(base_url: str) -> Optional[str]:
    """Per-key REST base_url se Gemini Live WebSocket URL banata hai.

    SmartRotator me har Google key ka apna CF worker base_url hota hai, e.g.:
        https://smartrotator.acc3.workers.dev/gemini/v1beta
    Live ke liye wahi worker `/v1/live` pe WebSocket serve karta hai:
        wss://smartrotator.acc3.workers.dev/v1/live

    Agar URL workers.dev CF worker nahi hai (e.g. direct Google ya koi aur
    gateway), toh None — relay default Google endpoint use karega.
    """
    b = (base_url or "").strip()
    if not b or ".workers.dev" not in b:
        return None
    # scheme aur path ka koi bharosa nahi — host nikal ke wss://<host>/v1/live
    try:
        from urllib.parse import urlparse

        host = urlparse(b).hostname
    except Exception:  # noqa: BLE001
        host = None
    if not host:
        return None
    return f"wss://{host}/v1/live"


async def relay_live_session(
    client_send,    # async callable: (str | bytes) -> None  (client ko bhejo)
    client_recv,    # async iterator: client se JSON string/bytes aayengi
    *,
    gemini_key: str,
    upstream_base: Optional[str] = None,
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
      upstream_base:
                   Pick hui key ka per-key CF worker /v1/live URL (wss://...).
                   None ho toh default LIVE_WS_BASE use hota hai. Isse key ka
                   apna network/egress path live ke liye bhi use hota hai.

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

    # 2) SETUP NORMALIZATION — native-audio live models sirf AUDIO modality
    #    accept karte hain. Client TEXT/TEXT+AUDIO maange toh 1011/invalid-
    #    argument milta hai (research-confirmed). AUDIO + output_audio_
    #    transcription force karo taaki connection na tootte aur text bhi
    #    aaye (`server_content.output_transcription.text`).
    normalized_setup, normalize_error = normalize_live_setup(
        decoded_setup, setup_model
    )
    if normalize_error:
        await client_send(
            json.dumps(
                {
                    "error": {
                        "code": "MODEL_NOT_LIVE_COMPATIBLE",
                        "message": normalize_error,
                    }
                }
            )
        )
        return {"ok": False, "reason": "model_not_live_compatible"}        
    decoded_setup = normalized_setup

    # 2) Google Live API se connect karo (apni key ke saath).
    #    upstream_base (pick hui key ka per-key CF worker /v1/live) ho toh
    #    wahan jata hai — key hidden, egress usi network se. Warna default
    #    Google endpoint. `?`/`&` dono handle karo.
    base = upstream_base or LIVE_WS_BASE
    sep = "&" if "?" in base else "?"
    ws_url = f"{base}{sep}key={gemini_key}"
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
        #    AUDIT FIX #7: agar send khud fail ho (upstream connect ke turant
        #    baad close ho gaya) client ko SPECIFIC error mile, generic
        #    SESSION_ERROR nahi.
        try:
            await upstream.send(decoded_setup)
        except Exception as exc:  # noqa: BLE001
            logger.warning("live: setup send to upstream failed: %r", exc)
            try:
                await client_send(
                    json.dumps(
                        {
                            "error": {
                                "code": "SETUP_FORWARD_FAILED",
                                "message": (
                                    "Setup message Google ko forward nahi ho saka: "
                                    f"{exc}"
                                ),
                            }
                        }
                    )
                )
            except Exception:  # noqa: BLE001
                pass
            return {
                "ok": False,
                "reason": "setup_send_failed",
                "error": str(exc),
            }

        # 4) bidirectional relay — do tasks aage-piche frames copy karte hain.

        async def client_to_upstream():
            try:
                async for msg in client_recv:
                    if isinstance(msg, (bytes, bytearray)):
                        await upstream.send(bytes(msg))
                    else:
                        await upstream.send(str(msg))
            except asyncio.CancelledError:  # noqa: BLE001
                pass
            except WebSocketDisconnect as wsd:
                # Client ne normal close kiya (disconnect = expected, error nahi)
                logger.info(
                    "live: client disconnected (%s, code=%s)",
                    getattr(wsd, "reason", "") or "no reason",
                    getattr(wsd, "code", 1000),
                )
            except Exception as exc:  # noqa: BLE001
                # Audio/setup aage-piche relay ke duran upstream fail
                # (Google ne conn close kiya / send fail). Hot diagnostic —
                # Render pe WARNING visible, silent swallow na ho.
                logger.warning(
                    "live: client->upstream relay error: %r  (len frame ho sakta "
                    "hai audio; closing session)",
                    exc,
                )

        async def upstream_to_client():
            try:
                async for msg in upstream:
                    if isinstance(msg, (bytes, bytearray)):
                        await client_send(bytes(msg))
                    else:
                        await client_send(str(msg))
            except asyncio.CancelledError:  # noqa: BLE001
                pass
            except WebSocketDisconnect as wsd:
                logger.info(
                    "live: upstream/disconnect (code=%s)", getattr(wsd, "code", 1000)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "live: upstream->client relay error: %r", exc
                )

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
        session_error = None
        for task in done:
            exc = task.exception()
            if exc:
                logger.warning("live: relay task error: %r", exc)
                session_error = repr(exc)

        # cleanup: doosre relay task ko cancel karo
        for task in (relay_client, relay_upstream):
            if not task.done():
                task.cancel()
        await asyncio.gather(relay_client, relay_upstream, return_exceptions=True)

        result = {"ok": True, "setup_model": setup_model}
        if session_error:
            result["error"] = session_error
        return result

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
