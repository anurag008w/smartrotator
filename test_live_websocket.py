"""
test_live_websocket.py — E2E integration test for /v1/live WebSocket route.

Ye script local server ko TestClient se chalati hai:
  1. rotator me gemini key set hoti hai (fake — upstream connect mock)
  2. TestClient se /v1/live WebSocket connect hota hai (real user token)
  3. valid setup message bheja jata hai
  4. upstream (Google) connect mock fail → server `UPSTREAM_CONNECT_FAILED`
     error message client ko bhejta hai

Isse pura live path verify hota hai: auth → key pick → setup parse → upstream
attempt → error relay. Real Google key hone pe audio/video frames relay hoti hain.

Run: python test_live_websocket.py
"""
from __future__ import annotations

import json
import os
import shutil

os.environ["SMARTROTATOR_DATA_DIR"] = "./test_data_live_ws"
os.environ["GEMINI_KEYS"] = "fake-gemini-key-123"
os.environ["GROQ_KEYS"] = ""
os.environ["OPENROUTER_KEYS"] = ""
os.environ["NVIDIA_KEYS"] = ""
os.environ["ZEN_KEYS"] = ""
if os.path.exists("./test_data_live_ws"):
    shutil.rmtree("./test_data_live_ws")


def test_live_websocket_flow():
    from starlette.testclient import TestClient
    from rotator import app as app_module
    from rotator import live as live_proxy

    # upstream connect ko patch karo — fast error path verify (bina real network)
    async def fake_connect(*a, **k):
        raise ConnectionError("simulated upstream failure")

    live_proxy.websockets.connect = fake_connect

    # dono WS paths test karo — /v1/live (custom) aur Google-exact /ws path
    paths = [
        "/v1/live",
        "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
    ]

    with TestClient(app_module.app) as client:
        # 0) pehla user admin = jab tak auth pass ho sake uska token lo
        r = client.post("/auth/register", json={"username": "liveuser", "password": "pass1234"})
        tok = r.json()["token"]

        for path in paths:
            with client.websocket_connect(f"{path}?key={tok}") as ws:
                # 1) valid setup bhejo (raw Gemini snake_case schema)
                ws.send_text(
                    json.dumps(
                        {
                            "setup": {
                                "model": "models/gemini-3.1-flash-live-preview",
                                "generation_config": {"response_modalities": ["AUDIO"]},
                                "system_instruction": {"parts": [{"text": "hi"}]},
                            }
                        }
                    )
                )
                # 2) patched upstream → connect fail → server error bhejega
                got_error = False
                try:
                    while True:
                        msg = ws.receive_json()
                        if "error" in msg:
                            got_error = True
                            print(
                                f"  [{path}] server error: {msg['error'].get('code')} "
                                f": {msg['error'].get('message')}"
                            )
                            break
                except Exception:
                    pass
                assert got_error, f"[{path}] server ne error msg nahi bheja (upstream fail expected)"
    return True


if __name__ == "__main__":
    try:
        test_live_websocket_flow()
        print("PASS  live websocket flow (error path)")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"ERROR {e!r}")
        raise SystemExit(1)
