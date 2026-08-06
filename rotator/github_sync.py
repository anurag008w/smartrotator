"""
github_sync.py — GitHub data sync for SmartRotator (DB ki jagah).

Users + usage + managed model config ab JSON files me `data/` directory me
rehte hain, aur ek private GitHub repo me push/pull hote hain. Isliye:

  - restart / redeploy pe data survive karta hai (DB ki zaroorat nahi)
  - har GITHUB_SYNC_INTERVAL seconds me data change hone pe auto-push
  - startup pe pehle pull (remote se latest data le lo)

Implementation: pure `git` + `GH_TOKEN` (Personal Access Token) — koi gh
CLI dependency nahi, koi extra pip package nahi. Render pe bhi chalta hai.

Env vars:
  GITHUB_SYNC_ENABLED   "true" to enable (default: auto — GH_TOKEN+repo mile toh)
  GITHUB_REPO           "owner/repo" override (default: smartrotator-data)
  GH_TOKEN              GitHub PAT (Render secret — already set)
  GITHUB_SYNC_INTERVAL  seconds between pushes (default 180 = 3 min)
  SMARTROTATOR_DATA_DIR data dir override (default: <repo root>/data)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("smartrotator.github_sync")

# ── Config ──────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent          # rotator/
_PROJECT_ROOT = _APP_DIR.parent                     # repo root
DATA_DIR = Path(os.environ.get("SMARTROTATOR_DATA_DIR", str(_PROJECT_ROOT / "data")))

REPO_NAME = os.environ.get("GITHUB_REPO", "").strip() or "smartrotator-data"
GH_TOKEN = os.environ.get("GH_TOKEN", "").strip()
SYNC_ENABLED_OVERRIDE = os.environ.get("GITHUB_SYNC_ENABLED", "").strip().lower()

try:
    GITHUB_SYNC_INTERVAL = max(30, int(os.environ.get("GITHUB_SYNC_INTERVAL", "180") or "180"))
except (ValueError, TypeError):
    GITHUB_SYNC_INTERVAL = 180  # 3 minutes

# ── Fingerprint cache for smart auto-push ──────────
_last_push_fingerprint: str = ""
# Boot pull sahi raha kya? False = pull fail hua — tab tak PUSH mat karo
# (warna fresh/khali state GitHub ke sahi data ke upar chali jayegi).
_last_pull_ok: bool = False


# ── Helpers ─────────────────────────────────────────
def _redact(text: str) -> str:
    """Logs me GH_TOKEN kabhi na dikhe — git stderr me authenticated URL aata hai."""
    if not text:
        return text
    if GH_TOKEN and GH_TOKEN in text:
        text = text.replace(GH_TOKEN, "***")
    # anonymous-style token bhi replace (ghp_xxx patterns)
    return text


def _run(cmd: list[str], check: bool = False, capture: bool = True, timeout: int = 60, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run a command safely with timeout."""
    try:
        result = subprocess.run(
            cmd, capture_output=capture, text=True, timeout=timeout, cwd=cwd
        )
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{_redact(result.stderr)}")
        return result
    except FileNotFoundError:
        return subprocess.CompletedProcess(cmd, returncode=127, stdout="", stderr="command not found")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, returncode=-1, stdout="", stderr="timeout")


def _safe_url() -> str:
    """Public repo URL (bina token ke)."""
    if "/" not in REPO_NAME:
        return f"https://github.com/{REPO_NAME}"
    return f"https://github.com/{REPO_NAME}"


def _auth_url() -> str:
    """Git remote URL with token embedded (push/pull ke liye)."""
    base = REPO_NAME if "/" in REPO_NAME else f"anonymous/{REPO_NAME}"
    if GH_TOKEN:
        return f"https://{GH_TOKEN}@github.com/{base}.git"
    return f"https://github.com/{base}.git"


def is_enabled() -> bool:
    """GitHub sync enable hai ya nahi."""
    if SYNC_ENABLED_OVERRIDE:
        return SYNC_ENABLED_OVERRIDE != "false"
    return bool(GH_TOKEN)


def compute_fingerprint() -> str:
    """Data dir ka fingerprint — files + mtimes + sizes se."""
    import hashlib

    h = hashlib.md5()
    if not DATA_DIR.exists():
        return ""
    for f in sorted(DATA_DIR.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(DATA_DIR))
            h.update(rel.encode())
            try:
                stat = f.stat()
                h.update(str(stat.st_mtime_ns).encode())
                h.update(str(stat.st_size).encode())
            except OSError:
                pass
    return h.hexdigest()


def has_data_changed() -> bool:
    """Last push ke baad se data badla hai kya."""
    global _last_push_fingerprint
    fp = compute_fingerprint()
    return fp != _last_push_fingerprint


def mark_pushed() -> None:
    """Successful push ke baad fingerprint update karo."""
    global _last_push_fingerprint
    _last_push_fingerprint = compute_fingerprint()


# ── GitHub API (repo ensure) ────────────────────────
def _api_request(method: str, path: str, body: dict | None = None, timeout: int = 30):
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        method=method,
        headers={
            "Authorization": f"token {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "smartrotator-sync",
        },
    )
    if body is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(body).encode()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def ensure_repo() -> bool:
    """Repo exists? Nahin toh create karo (private)."""
    if "/" in REPO_NAME:
        owner, name = REPO_NAME.split("/", 1)
        info = _api_request("GET", f"/repos/{REPO_NAME}")
        if info is not None:
            return True
        # create under user's account
        try:
            _api_request("POST", "/user/repos", {"name": name, "private": True, "description": "SmartRotator data backup"})
            log.info("created private repo: %s", REPO_NAME)
            return True
        except Exception as exc:
            log.warning("repo create failed: %s", exc)
            return False
    return False


# ── Git push (data dir → GitHub) ────────────────────
def _data_ready_to_push() -> bool:
    """Push karne se pehle check: kya data complete/seeded hai?

    Sirf usage/sync-files + 0 users = fresh-boot (pull fail hua) ya partial
    state — aisa data GitHub ke sahi data ke upar push karna data-loss hai.
    users.json missing ya 0 users → skip. Admin bhi hamesha hota hai, isliye
    0 users kabhi valid durable state nahi hai."""
    users_path = DATA_DIR / "users.json"
    if not users_path.exists():
        log.warning("github_sync: users.json nahi hai — push skip (pull fail ya fresh boot)")
        return False
    try:
        raw = json.loads(users_path.read_text(encoding="utf-8"))
        users = raw.get("users", []) if isinstance(raw, dict) else []
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("github_sync: users.json padh nahi paya (%s) — push skip", exc)
        return False
    if not users:
        log.warning("github_sync: users.json me 0 users — push skip (sahi data overwrite na ho)")
        return False
    return True


def push_data(force_mode: str = "normal") -> bool:
    """Local data/ ko GitHub pe push karo. True on success."""
    if not is_enabled():
        return False
    if not _last_pull_ok:
        log.warning("github_sync: boot pull abhi tak successful nahi — push skip")
        return False
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        log.info("github_sync: data dir empty, nothing to push")
        return False
    if not _data_ready_to_push():
        return False

    try:
        ensure_repo()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_dir = tmp_path / "repo"

            # clone (ya empty init) remote repo
            r = _run(["git", "clone", "--depth", "1", _auth_url(), str(repo_dir)], timeout=90)
            if r.returncode != 0:
                repo_dir.mkdir(parents=True, exist_ok=True)
                _run(["git", "init"], cwd=str(repo_dir), check=True)
                _run(["git", "checkout", "-b", "main"], cwd=str(repo_dir), check=False)
            _run(["git", "remote", "remove", "origin"], cwd=str(repo_dir), check=False)
            _run(["git", "remote", "add", "origin", _auth_url()], cwd=str(repo_dir), check=True)
            _run(["git", "config", "user.email", "smartrotator@sync"], cwd=str(repo_dir), check=True)
            _run(["git", "config", "user.name", "SmartRotator Sync"], cwd=str(repo_dir), check=True)

            # fresh local data copy on top
            for item in repo_dir.iterdir():
                if item.name == ".git":
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            for item in DATA_DIR.iterdir():
                if item.name.startswith("."):
                    continue
                dst = repo_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)

            _run(["git", "add", "-A"], cwd=str(repo_dir), check=True)
            commit_msg = f"smartrotator sync {time.strftime('%Y-%m-%d %H:%M:%S')}"
            r = _run(["git", "commit", "-m", commit_msg], cwd=str(repo_dir))
            if r.returncode != 0:
                log.info("github_sync: nothing to commit")
                return False

            _run(["git", "branch", "-M", "main"], cwd=str(repo_dir), check=True)
            push_flag = "--force-with-lease" if force_mode == "force-with-lease" else ""
            r = _run(["git", "push", "-u", "origin", "main"] + ([push_flag] if push_flag else []), cwd=str(repo_dir), timeout=90)
            if r.returncode != 0:
                stderr = _redact(r.stderr or "").lower()
                if any(k in stderr for k in ("rejected", "diverged", "non-fast-forward", "failed to push")):
                    log.warning("normal push rejected, retrying --force: %s", _redact((r.stderr or "").strip()[:300]))
                    r = _run(["git", "push", "-u", "origin", "main", "--force"], cwd=str(repo_dir), timeout=90)
            if r.returncode != 0:
                log.error("github_sync: push failed: %s", _redact((r.stderr or "").strip()[:400]))
                return False
        mark_pushed()
        log.info("github_sync: pushed data to %s", _safe_url())
        return True
    except Exception as exc:
        log.error("github_sync: push error: %s", exc)
        return False


# ── Git pull (GitHub → local) ───────────────────────
def pull_data(force_mode: str = "normal") -> bool:
    """GitHub repo se local data/ me latest lao. True on success."""
    global _last_pull_ok
    if not is_enabled():
        _last_pull_ok = False
        return False
    try:
        if "/" not in REPO_NAME or _api_request("GET", f"/repos/{REPO_NAME}") is None:
            log.info("github_sync: repo %s nahi mila — data shuru se", REPO_NAME)
            _last_pull_ok = False
            return False
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            r = _run(["git", "clone", "--depth", "1", _auth_url(), str(tmp_path / "repo")], timeout=90)
            if r.returncode != 0:
                log.warning("github_sync: clone failed: %s", (r.stderr or "").strip()[:300])
                _last_pull_ok = False
                return False
            repo_dir = tmp_path / "repo"
            items = [f for f in repo_dir.iterdir() if f.name not in (".git", ".gitignore")]
            if not items:
                log.info("github_sync: remote repo empty")
                _last_pull_ok = True
                return True
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            # local non-hidden files replace karo
            for item in DATA_DIR.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            for item in repo_dir.iterdir():
                if item.name.startswith("."):
                    continue
                dst = DATA_DIR / item.name
                if item.is_dir():
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)
        log.info("github_sync: pulled %d items from %s", len(items), _safe_url())
        mark_pushed()
        _last_pull_ok = True
        return True
    except Exception as exc:
        log.error("github_sync: pull error: %s", exc)
        _last_pull_ok = False
        return False


# ── Background sync loop (har 3 min) ────────────────
async def sync_loop(stop_event: asyncio.Event | None = None) -> None:
    """Background task — har GITHUB_SYNC_INTERVAL pe data change check + push.

    Agar boot pull fail hua tha, har loop me PEHLE pull retry karo — jab tak
    pull successful na ho, push nahi hoga (taaki fresh/khali data GitHub ke
    sahi data ke upar na jaye)."""
    if not is_enabled():
        return
    # startup pe ek baar immediate sync (data badla ho toh)
    if has_data_changed():
        await asyncio.to_thread(push_data)
    while True:
        try:
            await asyncio.sleep(GITHUB_SYNC_INTERVAL)
        except asyncio.CancelledError:
            break
        if stop_event is not None and stop_event.is_set():
            break
        try:
            # boot pull fail hua tha? pehle dobara pull karo (GitHub transient
            # ho sakta hai) — push tabhi jab pull ok ho.
            if not _last_pull_ok:
                log.warning("github_sync: boot pull fail tha — dobara pull try")
                await asyncio.to_thread(pull_data)
                continue
            if has_data_changed():
                await asyncio.to_thread(push_data)
        except Exception as exc:
            log.error("github_sync: loop error: %s", exc)


def sync_status() -> dict:
    """Status info (dashboard /status pe dikh sake)."""
    return {
        "enabled": is_enabled(),
        "repo": REPO_NAME,
        "interval_seconds": GITHUB_SYNC_INTERVAL,
        "data_dir": str(DATA_DIR),
        "data_changed": has_data_changed(),
        "pull_ok": _last_pull_ok,
    }
