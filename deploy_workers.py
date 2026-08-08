#!/usr/bin/env python3
"""
deploy_workers.py — Multiple Cloudflare accounts ke liye SmartRotator Worker deploy

Har account ke liye:
  1. Account label, Account ID, API Token maangta hai (aur optional R2 keys)
  2. Credentials ko ~/.config/smartrotator/accounts/<label>.env me save karta hai
     (600 permissions, GitHub se bahar — kabhi repo me nahi jaata)
  3. Wrangler se worker deploy karta hai
  4. Worker URL nikal kar base_urls ka ready-to-use format deta hai

Usage:
  python deploy_workers.py
  python deploy_workers.py --account myacc   (specific account redeploy)
  python deploy_workers.py --list            (saved accounts dekho)
"""

import argparse
import getpass
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
SECRETS_DIR = Path.home() / ".config" / "smartrotator" / "accounts"
WORKER_SCRIPT = REPO_DIR / "cloudflare-worker.js"
WORKER_NAME = "smartrotator"  # har account me yehi naam — per-account namespace hai

# Providers ke liye route paths — output format ke liye
ROUTES = {
    "gemini": "/gemini/v1beta",
    "groq": "/groq/openai/v1",
    "openrouter": "/openrouter/api/v1",
    "nvidia": "/nvidia/v1",
    "zen": "/zen/v1",
}


def ensure_env():
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(SECRETS_DIR, 0o700)


def list_accounts():
    if not SECRETS_DIR.exists():
        print("Koi account saved nahi hai.")
        return
    files = sorted(SECRETS_DIR.glob("*.env"))
    if not files:
        print("Koi account saved nahi hai.")
        return
    print(f"\nSaved accounts ({len(files)}):")
    for f in files:
        acc = ""
        try:
            for line in f.read_text().splitlines():
                if line.startswith("CLOUDFLARE_ACCOUNT_ID="):
                    acc = line.split("=", 1)[1][:12] + "..."
        except Exception:
            pass
        print(f"  - {f.stem:<20} account_id: {acc}")
    print()


def load_account_env(label: str) -> dict:
    path = SECRETS_DIR / f"{label}.env"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def prompt_account(label: str) -> dict:
    """Naya account details maango (ya saved wale ka use karo)."""
    saved = load_account_env(label)

    print(f"\n=== Account: {label} ===")
    if saved:
        print(f"  Saved creds mili (account_id: {saved.get('CLOUDFLARE_ACCOUNT_ID', '?')[:12]}...)")
        use = input("  Inhi ko use karein? [Y/n]: ").strip().lower()
        if use not in ("n", "no"):
            return saved
    else:
        print("  (kuch nahi mila — nayi credentials daalo)")

    acc_id = input("  Cloudflare Account ID: ").strip()
    while not acc_id:
        print("  ❌ Account ID zaroori hai!")
        acc_id = input("  Cloudflare Account ID: ").strip()

    token = getpass.getpass("  API Token (input hidden): ").strip()
    while not token:
        print("  ❌ API Token zaroori hai!")
        token = getpass.getpass("  API Token (input hidden): ").strip()

    # R2 optional — worker deploy ke liye zaroori nahi, bas bacha ke rakhte hain
    print("  R2 credentials (optional — Enter dabao to skip):")
    r2_key = input("    R2 Access Key ID: ").strip()
    r2_secret = getpass.getpass("    R2 Secret Access Key: ").strip()
    r2_endpoint = input("    R2 S3 API Endpoint: ").strip()

    creds = {"CLOUDFLARE_ACCOUNT_ID": acc_id, "CLOUDFLARE_API_TOKEN": token}
    if r2_key and r2_secret and r2_endpoint:
        creds["R2_ACCESS_KEY_ID"] = r2_key
        creds["R2_SECRET_ACCESS_KEY"] = r2_secret
        creds["R2_S3_API_ENDPOINT"] = r2_endpoint

    path = SECRETS_DIR / f"{label}.env"
    lines = [f"# SmartRotator Cloudflare creds — {label} (private, GitHub se bahar)\n"]
    lines += [f"{k}={v}\n" for k, v in creds.items()]
    path.write_text("".join(lines))
    os.chmod(path, 0o600)
    print(f"  💾 Saved -> {path} (600 permissions)")

    return creds


def deploy_account(label: str, creds: dict) -> bool:
    """Ek account pe worker deploy karo, URL nikal kar batao."""
    if not (REPO_DIR / "wrangler.toml").exists():
        print("❌ wrangler.toml nahi mila — script ko repo root me chalao")
        return False
    if not WORKER_SCRIPT.exists():
        print("❌ cloudflare-worker.js nahi mila")
        return False

    env = dict(os.environ)
    env["CLOUDFLARE_ACCOUNT_ID"] = creds["CLOUDFLARE_ACCOUNT_ID"]
    env["CLOUDFLARE_API_TOKEN"] = creds["CLOUDFLARE_API_TOKEN"]
    env["CLOUDFLARE_API_TOKEN"] = creds["CLOUDFLARE_API_TOKEN"]

    print(f"\n🚀 Deploying worker '{WORKER_NAME}' to account '{label}' ...")
    proc = subprocess.run(
        ["npx", "wrangler", "deploy"],
        cwd=str(REPO_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = (proc.stdout or "") + (proc.stderr or "")

    if proc.returncode != 0:
        print("❌ Deploy fail hua:")
        print(output[-1200:])
        return False

    # worker URL output se nikal lo — https://<name>.<subdomain>.workers.dev
    url_match = re.search(r"https://[\w.-]+\.workers\.dev", output)
    if not url_match:
        print("⚠️ Deploy ho gaya par URL output me nahi mila — dashboard me check karo")
        print(output[-800:])
        return False

    worker_url = url_match.group(0).rstrip("/")
    print(f"\n✅ Deployed! Worker URL: {worker_url}\n")

    print("=" * 62)
    print(f"  📍 ACCOUNT: {label}")
    print("=" * 62)
    for prov, route in ROUTES.items():
        print(f"{prov:<12} {worker_url}{route}")
    print("=" * 62)
    print("\nYe base_urls SmartRotator ke 🔌 Providers tab me daal do.")
    print(f"API keys wahi rahengi — worker sirf route + UA-spoof + IP-diversify karta hai.\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="SmartRotator multi-account Cloudflare Worker deployer")
    parser.add_argument("--account", help="Specific account label deploy karo (pehle se saved)")
    parser.add_argument("--list", action="store_true", help="Saved accounts list karo")
    parser.add_argument("--name", default=WORKER_NAME, help=f"Worker name (default: {WORKER_NAME})")
    args = parser.parse_args()

    ensure_env()

    if args.list:
        list_accounts()
        return

    if args.account:
        label = args.account
        creds = load_account_env(label)
        if not creds:
            print(f"❌ Account '{label}' saved nahi hai — pehle bina --account chalao ya --list dekho")
            return
        deploy_account(label, creds)
        return

    print("""
╔══════════════════════════════════════════════════════════╗
║   SmartRotator — Cloudflare Worker Multi-Account Setup   ║
║   Har account ke liye creds maangega + deploy karega     ║
║   Creds ~/.config/smartrotator/accounts/ me safe honge   ║
╚══════════════════════════════════════════════════════════╝
""")

    while True:
        print("\n──────────────────────────────────────────────")
        label = input("Account label (e.g. acc1, acc2, myaccount): ").strip()
        if not label:
            print("Label khali nahi ho sakta.")
            continue
        label = re.sub(r"[^a-zA-Z0-9_-]", "_", label)

        creds = prompt_account(label)
        deploy_account(label, creds)

        more = input("Aur account add karna hai? [y/N]: ").strip().lower()
        if more not in ("y", "yes"):
            break

    print("\n✅ Done! Sab credentials:", SECRETS_DIR)
    list_accounts()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBye! 👋")
        sys.exit(0)
