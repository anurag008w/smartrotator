"""
test_rotation.py — smoke test: rotation order verify karta hai.

  Key 1 → model 1, model 2   (models pehle rotate)
  Key 2 → model 1, model 2
  ... jab poora provider quota exhaust →
  next provider

Run:  python test_rotation.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

from rotator.providers import ChatMessage, ChatResult
import rotator.router as router_mod

# ---------------------------------------------------------------
# Fake provider jo HTTP nahi karta — bas call record karta hai
# ---------------------------------------------------------------
CALLS: list[tuple[str, str, str]] = []  # (provider, model, key_prefix)


class FakeProvider:
    def __init__(self, name, models):
        self.name = name
        self.models = models

    async def chat(self, messages, model, *, max_tokens=4096, temperature=0.7,
                   proxy=None, api_key=None, tools=None, tool_choice=None):
        CALLS.append((self.name, model, api_key[:12]))
        return ChatResult(
            text="ok",
            provider=self.name,
            model=model,
            key_label=f"{self.name}[0]",
        )

    async def aclose(self):
        pass


def fake_build_provider(name, ptype, base_url, models):
    return FakeProvider(name, models)


CONFIG_YAML = """
rotation:
  strategy: round_robin
  provider_strategy: sequential
  cooldown_seconds: 5
  max_fallback_attempts: 16
  fail_after_attempts: 3

providers:
  - name: gemini
    type: gemini
    models: [gemini-2.5-flash, gemini-2.5-pro]
    api_keys: ["gk-111111111111", "gk-222222222222"]
    rpd_limit: 1000
  - name: groq
    type: openai
    base_url: https://api.groq.com/openai/v1
    models: [llama-3.3-70b-versatile, llama-4-scout]
    api_keys: ["gq-333333333333"]
    rpd_limit: 1000
"""


async def main() -> int:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as fh:
        fh.write(CONFIG_YAML)

    router_mod.build_provider = fake_build_provider
    rot = router_mod.Rotator(config_path=path)
    assert len(rot.providers) == 2, "2 providers load hone chahiye"

    probe = [ChatMessage(role="user", content="hi")]

    # ---------- TEST 1: key-major, model-minor rotation ----------
    CALLS.clear()
    for _ in range(4):
        await rot.chat(probe)

    expected_cycle = [
        ("gemini", "gemini-2.5-flash", "gk-111111111"),
        ("gemini", "gemini-2.5-pro", "gk-111111111"),
        ("gemini", "gemini-2.5-flash", "gk-222222222"),
        ("gemini", "gemini-2.5-pro", "gk-222222222"),
    ]

    print("TEST 1 — rotation order (key-major, model-minor):")
    for i, (p, m, k) in enumerate(CALLS, 1):
        print(f"  {i}. {p:<8} {m:<24} {k}")

    ok1 = CALLS == expected_cycle
    print(f"  -> {'✅ PASS' if ok1 else '❌ FAIL'}")

    # ---------- TEST 2: provider drain fallback ----------
    CALLS.clear()
    # gemini ki saari keys ka daily quota exhaust kar do
    for state in rot.providers[0].ring.items:
        state.daily_count = state._rpd_limit if hasattr(state, "_rpd_limit") else rot.providers[0].ring._rpd_limit
    # groq available hai
    await rot.chat(probe)

    print("TEST 2 — provider drain fallback (gemini exhausted → groq):")
    print(f"  picked: {CALLS[0][0]} / {CALLS[0][1]} / {CALLS[0][2]}")
    ok2 = CALLS[0][0] == "groq"
    print(f"  -> {'✅ PASS' if ok2 else '❌ FAIL'}")

    # ---------- TEST 3: pinned single model ----------
    CALLS.clear()
    # gemini ka quota reset karo (TEST 2 me exhaust kiya tha)
    for state in rot.providers[0].ring.items:
        state.daily_count = 0
    await rot.chat(probe, model="gemini-2.5-pro")
    print("TEST 3 — pinned model (sirf gemini-2.5-pro, keys rotate):")
    print(f"  picked: {CALLS[0][1]} / {CALLS[0][2]}")
    ok3 = CALLS[0][1] == "gemini-2.5-pro" and CALLS[0][2] == "gk-111111111"
    print(f"  -> {'✅ PASS' if ok3 else '❌ FAIL'}")

    await rot.aclose()
    os.unlink(path)
    ok = ok1 and ok2 and ok3
    print(f"\nOverall: {'✅ ALL PASS' if ok else '❌ SOME FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
