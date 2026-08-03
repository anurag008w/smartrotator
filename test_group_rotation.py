"""
test_group_rotation.py — group member ke andar models queue + keys subset rotation.

Run:  python test_group_rotation.py
"""

from __future__ import annotations

import asyncio
import os
import shutil

TEST_DATA_DIR = "./test_data_group_rot"
os.environ["SMARTROTATOR_DATA_DIR"] = TEST_DATA_DIR
os.environ["GITHUB_SYNC_ENABLED"] = "false"
if os.path.exists(TEST_DATA_DIR):
    shutil.rmtree(TEST_DATA_DIR)
os.makedirs(TEST_DATA_DIR, exist_ok=True)
with open(os.path.join(TEST_DATA_DIR, "config.yaml"), "w") as fh:
    fh.write(
        "providers:\n"
        "  - name: gemini\n    type: gemini\n    models: [g1, g2]\n    api_keys: [gk0]\n"
        "  - name: openrouter\n    type: openai\n    models: [o1]\n    api_keys: [ok0]\n    base_url: https://openrouter.ai/api/v1\n"
    )

from rotator.core import KeyRing, KeyState  # noqa: E402
from rotator.router import ProviderConfig, Rotator  # noqa: E402


def test_keyring_model_then_key_order():
    """Rotation order: Key1→m1, Key1→m2, Key2→m1, Key2→m2 (models rotate first)."""
    ring = KeyRing(
        keys=["k1", "k2"],
        models=["m1", "m2"],
        label="test",
        strategy="round_robin",
    )
    seq = []
    for _ in range(4):
        picked = ring.pick()
        assert picked is not None
        state, model = picked
        seq.append((state.key, model))
    expected = [("k1", "m1"), ("k1", "m2"), ("k2", "m1"), ("k2", "m2")]
    assert seq == expected, f"order galat: {seq}"
    print("TEST 1 — key-major model-minor order      ✅ PASS", seq)


def test_keyring_skips_cooldown_pair():
    """Ek (key,model) pair fail → chhota cooldown, phir next pair try."""
    ring = KeyRing(keys=["k1"], models=["m1", "m2"], label="test", strategy="round_robin", cooldown_seconds=60)
    state, model = ring.pick()
    assert (state.key, model) == ("k1", "m1")
    ring.report_failure(state, model)   # (k1, m1) cooldown pe
    picked = ring.pick()
    assert picked is not None, "doosra model available hona chahiye"
    assert picked[1] == "m2", f"m1 cooldown pe hai, m2 aana chahiye, aaya: {picked}"
    print("TEST 2 — (key,model) cooldown skip         ✅ PASS")


def test_group_member_keys_subset():
    """Group member me sirf selected keys use hoti hain."""
    rot = Rotator(config_path="test_data_group_rot/config.yaml")
    rot._strategy = "round_robin"
    rot._cooldown = 60.0
    rot._ban_after = 3
    # 2 providers: gemini (3 keys, 2 models), openrouter (2 keys, 1 model)
    rot.providers = [
        rot._build_state(ProviderConfig(name="gemini", ptype="gemini", models=["g1", "g2"], keys=["gk0", "gk1", "gk2"])),
        rot._build_state(ProviderConfig(name="openrouter", ptype="openai", models=["o1"], keys=["ok0", "ok1"], base_url="https://openrouter.ai/api/v1")),
    ]
    managed = {
        "provider_models": {"gemini": ["g1", "g2"], "openrouter": ["o1"]},
        "provider_order": ["gemini", "openrouter"],
        "groups": [
            {
                "id": "levelup",
                "label": "LevelUp",
                "enabled": True,
                "members": [
                    # gemini member: sirf key 0 aur 1, models queue [g2, g1]
                    {"provider": "gemini", "models": ["g2", "g1"], "keys": [0, 1]},
                    {"provider": "openrouter", "models": ["o1"], "keys": [1]},
                ],
            }
        ],
    }
    rot.apply_managed(managed)

    # member 1 ki ring: sirf gk0, gk1 + models [g2, g1]
    rings = rot._member_rings["levelup"]
    assert len(rings) == 2
    m1_keys = [s.key for s in rings[0]._items]
    assert m1_keys == ["gk0", "gk1"], f"keys subset galat: {m1_keys}"
    m2_keys = [s.key for s in rings[1]._items]
    assert m2_keys == ["ok1"], f"member 2 keys galat: {m2_keys}"

    # order: gk0→g2, gk0→g1, gk1→g2, gk1→g1, phir member2 ok1→o1
    seq = []
    for _ in range(5):
        picked = rings[0].pick() if len(seq) < 4 else rings[1].pick()
        assert picked is not None
        state, model = picked
        seq.append((state.key, model))
    expected = [("gk0", "g2"), ("gk0", "g1"), ("gk1", "g2"), ("gk1", "g1"), ("ok1", "o1")]
    assert seq == expected, f"rotation order galat: {seq}"
    print("TEST 3 — member keys subset + models queue ✅ PASS", seq)


async def test_group_rotation_fallback():
    """Member 1 ke saare (key,model) fail → member 2 try."""
    rot = Rotator(config_path="test_data_group_rot/config.yaml")
    rot._strategy = "round_robin"
    rot._cooldown = 3600.0  # long cooldown — fail hote hi sab khatam
    rot._ban_after = 1
    rot.providers = [
        rot._build_state(ProviderConfig(name="gemini", ptype="gemini", models=["g1"], keys=["gk0"])),
        rot._build_state(ProviderConfig(name="openrouter", ptype="openai", models=["o1"], keys=["ok0"], base_url="https://openrouter.ai/api/v1")),
    ]
    managed = {
        "provider_models": {"gemini": ["g1"], "openrouter": ["o1"]},
        "provider_order": ["gemini", "openrouter"],
        "groups": [
            {
                "id": "levelup",
                "label": "LevelUp",
                "enabled": True,
                "members": [
                    {"provider": "gemini", "models": ["g1"], "keys": []},
                    {"provider": "openrouter", "models": ["o1"], "keys": []},
                ],
            }
        ],
    }
    rot.apply_managed(managed)
    rot.provider_strategy = "sequential"

    # gemini provider ko fail karne wala mock
    class MockProvider:
        async def chat(self, *a, **kw):
            raise Exception("boom gemini")

    class MockProviderOK:
        async def chat(self, *a, **kw):
            from rotator.providers import ChatResult
            return ChatResult(text="hi from openrouter", provider="openrouter", model=kw.get("model", "o1"), key_label="openrouter[0]")

    async def aclose(self):
        pass

    rot.providers[0].provider = MockProvider()
    rot.providers[1].provider = MockProviderOK()

    from rotator.providers import ProviderError
    from rotator.router import ChatMessage

    # seed karo: gemini ka (gk0,g1) pehle hi fail mark — taaki drain ho
    rings = rot._member_rings["levelup"]
    st, m = rings[0].pick()
    try:
        await rot.providers[0].provider.chat([], "g1", api_key="gk0")
    except Exception:
        rings[0].report_failure(st, m)
        rot.providers[0].failures += 1
        rot.providers[0].last_error = "boom gemini"

    # ab group chat karo — gemini exhausted → openrouter chale
    result = await rot.chat([ChatMessage(role="user", content="hi")], model="levelup", max_fallback_attempts=5)
    assert "openrouter" in result.model or result.model == "o1", f"fallback nahi hua: {result.model}"
    assert result.text == "hi from openrouter"
    print("TEST 4 — member1 exhausted → member2       ✅ PASS ->", result.model)

    # Dashboard `models: ["levelup"]` (array) se group call — pehle yeh
    # ProviderError deta tha ("no matching model"), ab group resolve hona chahiye
    result2 = await rot.chat([ChatMessage(role="user", content="hi")], models=["levelup"], max_fallback_attempts=5)
    assert result2.text == "hi from openrouter", f"models-array group call fail: {result2.text}"
    print("TEST 4b — models=['levelup'] array group call ✅ PASS ->", result2.model)


def main() -> int:
    results = []
    tests = [test_keyring_model_then_key_order, test_keyring_skips_cooldown_pair, test_group_member_keys_subset]
    for t in tests:
        try:
            t()
            results.append(True)
        except Exception as exc:
            print(f"{t.__name__}: ❌ FAIL {exc}")
            results.append(False)
    try:
        asyncio.run(test_group_rotation_fallback())
        results.append(True)
    except Exception as exc:
        print(f"test_group_rotation_fallback: ❌ FAIL {exc}")
        results.append(False)
    total = len(results)
    passed = sum(results)
    print(f"--------------------------------------------------------------------\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
