#!/usr/bin/env python3
"""
keepalive.py — Render free tier sleep prevention.

Render free web services 15 min inactivity pe so jate hain.
Yeh script (Render cron job ke through) PING_URL ko har baar hit
karti hai taaki instance warm rahe.

Deploy me render.yaml ka 'smartrotator-keepalive' cron ise chala raha hai.
Local test:  python keepalive.py
"""

from __future__ import annotations

import os

import httpx

PING_URL = os.environ.get("PING_URL", "http://localhost:8000/health")


def main() -> None:
    try:
        resp = httpx.get(PING_URL, timeout=15)
        print(f"Pinged {PING_URL} -> {resp.status_code}")
    except Exception as exc:  # noqa: BLE001
        print(f"Ping failed: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
