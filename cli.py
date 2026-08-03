#!/usr/bin/env python3
"""
cli.py — SmartRotator command-line tools.

Commands:
  python cli.py status              → keys/providers/proxy ki current haalat
  python cli.py models              → configured models ki list
  python cli.py test-keys           → har key ko 1 request bhej ke check karo
  python cli.py chat "prompt"       → ek baar ka chat (--model, --image)
  python cli.py fetch-proxies       → Proxifly se free proxies download (proxies.txt)
  python cli.py serve               → FastAPI server chalao
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys

import httpx

from rotator.providers import ChatMessage, ImageInput
from rotator.router import Rotator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="smartrotator", description="Multi-provider LLM rotator")
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show provider/key/proxy status")
    sub.add_parser("models", help="list configured models")

    tk = sub.add_parser("test-keys", help="test every configured key")
    tk.add_argument("--model", default=None, help="test a specific model")

    ch = sub.add_parser("chat", help="send a one-shot message")
    ch.add_argument("prompt", help="your message")
    ch.add_argument("--model", default=None, help="model id (single pin)")
    ch.add_argument("--models", default=None, help="comma-separated model ids (multi-select rotation)")
    ch.add_argument("--image", action="append", default=[], help="image URL or local path (repeatable)")
    ch.add_argument("--system", default=None, help="system prompt")
    ch.add_argument("--max-tokens", type=int, default=4096)
    ch.add_argument("--temperature", type=float, default=0.7)

    fp = sub.add_parser("fetch-proxies", help="download free proxies into proxies.txt")
    fp.add_argument("--url", default=None, help="proxy list URL (default: proxifly http list)")

    sub.add_parser("serve", help="run the FastAPI server")

    return p


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
async def cmd_status(rot: Rotator) -> None:
    print(json.dumps(rot.status(), indent=2, ensure_ascii=False))


async def cmd_models(rot: Rotator) -> None:
    for m in rot.models():
        print(f"{m['provider']:<12} {m['id']}")


async def cmd_test_keys(rot: Rotator, model: str | None) -> None:
    results = []

    for st in rot.providers:
        for state in st.ring.items:
            test_model = model or (st.cfg.models[0] if st.cfg.models else "")
            probe = ChatMessage(role="user", content="Reply with exactly: OK")
            try:
                res = await st.provider.chat(
                    [probe],
                    test_model,
                    max_tokens=16,
                    temperature=0.0,
                    api_key=state.key,
                )
                state.mark_success()
                results.append(
                    {
                        "provider": st.cfg.name,
                        "key": state.label,
                        "model": test_model,
                        "status": "OK",
                        "reply": res.text[:60],
                    }
                )
            except Exception as exc:  # noqa: BLE001 — test command shows all errors
                state.mark_failure(rot.settings.get("cooldown_seconds", 60), 3)
                results.append(
                    {
                        "provider": st.cfg.name,
                        "key": state.label,
                        "model": test_model,
                        "status": "FAIL",
                        "error": str(exc)[:160],
                    }
                )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n{ok}/{len(results)} keys OK")


async def cmd_chat(rot: Rotator, args) -> None:
    images = []
    for ref in args.image:
        img = await load_image(ref)
        if img:
            images.append(img)

    messages: list[ChatMessage] = []
    if args.system:
        messages.append(ChatMessage(role="system", content=args.system))
    messages.append(ChatMessage(role="user", content=args.prompt, images=images))

    try:
        res = await rot.chat(
            messages,
            model=args.model,
            models=[m.strip() for m in args.models.split(",") if m.strip()] if args.models else None,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\n[{res.provider} / {res.model} / {res.key_label}]\n")
        print(res.text)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


async def cmd_fetch_proxies(url: str | None, out_path: str = "proxies.txt") -> None:
    target = url or (
        "https://raw.githubusercontent.com/proxifly/free-proxy-list/"
        "main/proxies/protocols/http/data.txt"
    )
    print(f"Fetching proxies from {target} ...")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(target)
        resp.raise_for_status()

    lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip() and ":" in ln]
    # keep only IP:PORT style entries
    proxies = [ln for ln in lines if ln.count(":") == 1 and ln.split(":")[0].replace(".", "").isdigit()]

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# auto-fetched free proxies\n")
        fh.write("\n".join(proxies))
        fh.write("\n")

    print(f"Saved {len(proxies)} proxies to {out_path}")
    print("NOTE: free proxies mostly get blocked by LLM providers.")
    print("Key rotation is the reliable path — proxy is optional garnish.")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
async def load_image(ref: str) -> ImageInput | None:
    """Load image from URL or local path → base64 ImageInput."""
    if ref.startswith("http://") or ref.startswith("https://"):
        return ImageInput(url=ref)
    try:
        with open(ref, "rb") as fh:
            data = fh.read()
        mime = "image/png" if ref.lower().endswith(".png") else "image/jpeg"
        return ImageInput(data_base64=base64.b64encode(data).decode(), mime_type=mime)
    except OSError as exc:
        print(f"Could not read image {ref}: {exc}", file=sys.stderr)
        return None


def main() -> None:
    args = build_parser().parse_args()
    rot = Rotator(config_path=args.config)

    try:
        if args.command == "status":
            asyncio.run(cmd_status(rot))
        elif args.command == "models":
            asyncio.run(cmd_models(rot))
        elif args.command == "test-keys":
            asyncio.run(cmd_test_keys(rot, args.model))
        elif args.command == "chat":
            asyncio.run(cmd_chat(rot, args))
        elif args.command == "fetch-proxies":
            asyncio.run(cmd_fetch_proxies(args.url))
        elif args.command == "serve":
            import uvicorn

            uvicorn.run("rotator.app:app", host="0.0.0.0", port=8000, log_level="info")
    finally:
        if args.command != "serve":
            asyncio.run(rot.aclose())


if __name__ == "__main__":
    main()
