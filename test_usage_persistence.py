"""
test_usage_persistence.py — usage.json persistence + restart recovery tests.

Bug fixed: `_usage` dict INT keys use karta hai (user_id), par JSON
round-trip pe saare dict keys STRING ban jaate hain ("1"). Isliye restart
pe saara usage lookup miss hota tha → sab 0 dikhta tha, aur naya write
int-key add kar deta tha → usage.json me DUPLICATE keys ban jaati thin →
agla load purana data kha jata tha (last-wins).

Covers:
  1. Restart simulation: usage.json me string keys ("1") ke saath data →
     init_db() → get_usage_row / get_usage_month / month_days sahi data
     return kare (0 nahi).
  2. Write ke baad usage.json me koi duplicate key na ho (int+str same
     user ke liye dono versions na likhe).
  3. Already-corrupt file (duplicate keys dono versions) load pe DEEP-MERGE
     ho — purana data preserve, naya data bhi add.

Run: python test_usage_persistence.py
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil

os.environ["SMARTROTATOR_DATA_DIR"] = "./test_data_usage_persist"
os.environ["GITHUB_SYNC_ENABLED"] = "false"
if os.path.exists("./test_data_usage_persist"):
    shutil.rmtree("./test_data_usage_persist")

from rotator import store  # noqa: E402


def _write_usage(payload: dict) -> None:
    store.USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    store.USAGE_FILE.write_text(json.dumps(payload), encoding="utf-8")


def _read_usage() -> dict:
    return json.loads(store.USAGE_FILE.read_text(encoding="utf-8"))


async def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


async def test_restart_recovers_string_key_usage() -> None:
    """Restart simulation: JSON me keys strings hain → sahi data milna chahiye."""
    print("\n[1] restart pe string-keyed usage.json recovery")
    # simulate: JSON file jo round-trip ke baad save hui (keys = "1", "2")
    _write_usage({
        "1": {
            "2026-08-06": {"requests": 5, "tokens": 100},
            "2026-08-07": {"requests": 3, "tokens": 50},
        },
        "2": {
            "2026-08-06": {"requests": 1, "tokens": 10},
        },
    })
    await store.init_db()

    row = await store.get_usage_row(1, "2026-08-06")
    await _assert(row.requests == 5, "user 1 day 08-06 requests == 5")
    await _assert(row.tokens == 100, "user 1 day 08-06 tokens == 100")

    row2 = await store.get_usage_row(2, "2026-08-06")
    await _assert(row2.requests == 1, "user 2 day 08-06 requests == 1")

    month = await store.get_usage_month(1, "2026-08")
    await _assert(month.requests == 8, "user 1 month requests == 8 (5+3)")

    days = await store.get_usage_month_days(1, "2026-08")
    await _assert(len(days) == 2, "user 1 month_days me 2 days")

    total = await store.get_monthly_totals(1, 6)
    await _assert(total[0].requests == 8, "monthly_totals[0] requests == 8")

    hist = await store.get_usage_between(1, 7)
    await _assert(len(hist) == 2, "last 7 days history me 2 days")


async def test_write_creates_no_duplicate_keys() -> None:
    """Reserve + tokens ke baad usage.json me ek hi "1" key honi chahiye."""
    print("\n[2] write ke baad duplicate JSON keys nahi")
    _write_usage({
        "1": {
            "2026-08-07": {"requests": 3, "tokens": 50},
        },
    })
    await store.init_db()

    # naya request: reserve + tokens (int-key path)
    await store.reserve_unlimited(1, "2026-08-07")
    await store.record_tokens(1, "2026-08-07", 10)

    raw = store.USAGE_FILE.read_text(encoding="utf-8")
    # duplicate "1" key ho toh raw me "1" do baar milega
    await _assert(raw.count('"1"') == 1, 'usage.json me "1" key sirf ek baar')

    data = _read_usage()
    await _assert(data["1"]["2026-08-07"]["requests"] == 4, "requests merged 3+1 == 4")
    await _assert(data["1"]["2026-08-07"]["tokens"] == 60, "tokens merged 50+10 == 60")


async def test_corrupt_duplicate_keys_deep_merge() -> None:
    """Purani corrupt file (int+str dono "1" keys) load pe merge ho — data lost na ho."""
    print("\n[3] corrupt duplicate keys file recovery (deep-merge)")
    # Simulate purani corrupt file: duplicate "1" keys (purana + naya version).
    # JSON spec me duplicate keys allowed hote hain — json.loads last-wins,
    # isliye hum raw text likhte hain taaki _pairs_hook trigger ho.
    store.USAGE_FILE.write_text(
        '{ "1": {"2026-08-06": {"requests": 5, "tokens": 100}}, '
        ' "1": {"2026-08-07": {"requests": 3, "tokens": 50}} }',
        encoding="utf-8",
    )
    await store.init_db()

    row6 = await store.get_usage_row(1, "2026-08-06")
    await _assert(row6.requests == 5, "purana day 08-06 preserve (5)")
    row7 = await store.get_usage_row(1, "2026-08-07")
    await _assert(row7.requests == 3, "naya day 08-07 add (3)")

    month = await store.get_usage_month(1, "2026-08")
    await _assert(month.requests == 8, "donon days merged month total 8")


async def main() -> None:
    await test_restart_recovers_string_key_usage()
    await test_write_creates_no_duplicate_keys()
    await test_corrupt_duplicate_keys_deep_merge()
    print("\nALL USAGE PERSISTENCE TESTS PASSED ✅")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        raise SystemExit(1)
