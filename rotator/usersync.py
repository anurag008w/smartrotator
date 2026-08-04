"""
usersync.py — Per-user app state sync (offline-first app backup).

Har user ka data ek alag folder me rehta hai (`data/sync/<user>/`), aur
uske andar scope-wise files (`state.json`, `chat.json`, `settings.json`...).
Ye directory github_sync.py se private GitHub repo me push/pull hoti hai,
isliye user data restart / redeploy / data loss pe survive karta hai.

Folder structure (GitHub repo me bhi wahi mirror hota hai):

  data/sync/
    alice/
      state.json      # app state (plan, tasks, progress, memory)
      chat.json       # chat sessions
      settings.json   # AI settings, profile, prefs
    bob/
      state.json
      ...

Design (app push-authoritative):
  - PUT /sync/state  → server bas save karta hai (last-write-wins).
  - GET /sync/state  → fresh install / naye device pe pull.
  - Har mutation app pe hota hai, phir online hote hi push — server kabhi
    app ka data overwrite nahi karta, sirf store karta hai.

Har file format:
  {
    "updated_at": "2026-08-04T12:00:00Z",
    "state": { ... us scope ka data ... }
  }

Security (API keys at rest):
  - Provider apiKey values GitHub repo me PLAINTEXT kabhi nahi jati.
  - Agar SYNC_ENC_KEY env set hai (koi bhi password/secret) toh apiKey fields
    ciphertext ("gAAAAAB...") me store hoti hain. Secret se AES-256 key derive
    hoti hai (SHA-256). Server owner ke liye decrypt karke deta hai (pull pe),
    taaki WhatsApp/AI control server-side bhi chal sake.
  - Agar SYNC_ENC_KEY NAHI set hai, apiKey fields store hone se pehle hata di
    jaati hain (repo leak-safe default).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from . import github_sync

SYNC_DIR = github_sync.DATA_DIR / "sync"

_lock = asyncio.Lock()

# Username → folder-safe (alphanumeric + _ - . ; nothing else)
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")

# Scope name → filename-safe (alphanumeric + _ - only)
_SCOPE_RE = re.compile(r"[^A-Za-z0-9_-]")

# Master key for apiKey at-rest encryption.
# Priority: env SYNC_ENC_KEY → file data/.sync-enc-key (auto-generated).
# File hidden hai (dotfile) — github_sync dotfiles skip karta hai, isliye
# secret kabhi GitHub repo me NAHI jata. Admin dashboard /admin/sync/enc-key
# se dekh/rotate kar sakte hain — yaad rakhne ki zaroorat nahi.
def _enc_secret_file() -> Path:
    return github_sync.DATA_DIR / ".sync-enc-key"


def _enc_key_hex() -> str:
    env_val = os.environ.get("SYNC_ENC_KEY", "").strip()
    if env_val:
        return env_val
    try:
        return _enc_secret_file().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_enc_secret(secret: str) -> bool:
    """Auto-generated secret ko hidden file me store karo."""
    try:
        path = _enc_secret_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secret + "\n", encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return True
    except OSError:
        return False


def get_or_create_secret() -> tuple[str, str]:
    """(secret, source). source: env | file | generated.
    Agar na env hai na file, random secret banao aur persist karo."""
    env_val = os.environ.get("SYNC_ENC_KEY", "").strip()
    if env_val:
        return env_val, "env"
    try:
        file_val = _enc_secret_file().read_text(encoding="utf-8").strip()
        if file_val:
            return file_val, "file"
    except OSError:
        pass
    import secrets

    generated = secrets.token_hex(32)
    _write_enc_secret(generated)
    return generated, "generated"


def rotate_secret() -> tuple[str, str]:
    """Naya secret banao aur file me save karo. WARNING: purana encrypted
    data is nayi key se unlock nahi hoga. Agar SYNC_ENC_KEY env set hai toh
    env key priority leta hai — file rotate ka asar NAHI hoga (pehle env hatao).
    """
    import secrets

    generated = secrets.token_hex(32)
    _write_enc_secret(generated)
    return generated, "generated"


# --------------------------------------------------------------------------
# apiKey at-rest encryption helpers
# --------------------------------------------------------------------------
def _master_key() -> Optional[bytes]:
    """Secret se AES-256 key derive karo.
    - 64-hex ya 32-byte raw string → direct use
    - koi bhi password → SHA-256 hash (32 bytes)
    Empty secret → None (apiKey strip hoti hai).
    """
    secret = _enc_key_hex()
    if not secret:
        return None
    try:
        key = bytes.fromhex(secret)
        if len(key) == 32:
            return key
    except (ValueError, TypeError):
        pass
    import hashlib

    return hashlib.sha256(secret.encode("utf-8")).digest()


def _encrypt_value(plain: str) -> str:
    """AES-256-GCM encrypt → "v1.<b64nonce>.<b64ct+tag>". Bina key → ''. """
    key = _master_key()
    if not key or not plain:
        return ""
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, plain.encode("utf-8"), b"levelup-sync-v1")
    return "v1." + base64.urlsafe_b64encode(nonce).decode() + "." + base64.urlsafe_b64encode(ct).decode()


def _decrypt_value(token: str) -> str:
    """Reverse of _encrypt_value. Bina key / invalid → ''. """
    if not token.startswith("v1."):
        return ""
    key = _master_key()
    if not key:
        return ""
    try:
        _, nonce_b64, ct_b64 = token.split(".", 2)
        nonce = base64.urlsafe_b64decode(nonce_b64.encode())
        ct = base64.urlsafe_b64decode(ct_b64.encode())
        return AESGCM(key).decrypt(nonce, ct, b"levelup-sync-v1").decode("utf-8")
    except Exception:
        return ""


def _walk_set(obj, leaf: str, fn) -> None:
    """Har dict me `leaf` key ko fn() se map karo (recursive)."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k == leaf:
                obj[k] = fn(v)
            else:
                _walk_set(v, leaf, fn)
    elif isinstance(obj, list):
        for item in obj:
            _walk_set(item, leaf, fn)


def _encrypt_state(state: dict) -> dict:
    """apiKey fields ko ciphertext me banao (at rest). Bina master key → hata do."""
    state = json.loads(json.dumps(state))  # deep copy
    _walk_set(state, "apiKey", lambda v: _encrypt_value(v) if isinstance(v, str) and v else v)
    return state


def _decrypt_state(state: dict) -> dict:
    """apiKey ciphertext → plaintext (owner ke pull pe)."""
    state = json.loads(json.dumps(state))  # deep copy
    _walk_set(state, "apiKey", lambda v: _decrypt_value(v) if isinstance(v, str) and v.startswith("v1.") else v)
    return state


def has_encryption() -> bool:
    """SYNC_ENC_KEY set hai ya nahi — status me dikh sake."""
    return bool(_master_key())


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _safe_username(username: str) -> str:
    """Username ko filesystem-safe banao (path traversal guard)."""
    name = _SAFE_RE.sub("_", username or "").strip(".")
    if not name:
        return "anonymous"
    return name


def _safe_scope(scope: str) -> str:
    """Scope name ko filename-safe banao (default: state)."""
    name = _SCOPE_RE.sub("_", scope or "").strip("_")
    return name or "state"


def _user_dir(username: str) -> Path:
    return SYNC_DIR / _safe_username(username)


def _scope_file(username: str, scope: str) -> Path:
    return _user_dir(username) / f"{_safe_scope(scope)}.json"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _parse_record(record) -> dict:
    if not isinstance(record, dict):
        return {"updated_at": "", "state": {}}
    return {
        "updated_at": record.get("updated_at", ""),
        "state": record.get("state", {}),
    }


# --------------------------------------------------------------------------
# Store operations (scope = ek data blob, e.g. state / chat / settings)
# --------------------------------------------------------------------------
async def get_user_scope(username: str, scope: str = "state") -> Optional[dict]:
    """User ke ek scope ka saved data (None agar kabhi sync nahi hua). apiKey decrypted."""
    async with _lock:
        record = _read_json(_scope_file(username, scope), None)
        if record is None:
            return None
        parsed = _parse_record(record)
        parsed["state"] = _decrypt_state(parsed["state"])
        return parsed


async def list_user_scopes(username: str) -> list[str]:
    """User folder me kaunse scopes sync hain."""
    async with _lock:
        udir = _user_dir(username)
        if not udir.is_dir():
            return []
        return sorted(p.stem for p in udir.glob("*.json") if p.is_file())


async def save_user_scope(username: str, scope: str, state: dict, client_updated_at: str = "") -> dict:
    """Ek scope save karo (push-authoritative — bas overwrite). apiKey encrypted."""
    async with _lock:
        updated_at = client_updated_at or _now_utc()
        stored_state = _encrypt_state(state)
        record = {"updated_at": updated_at, "state": stored_state}
        _write_json(_scope_file(username, scope), record)
        return {"scope": _safe_scope(scope), "updated_at": updated_at, "ok": True}


async def delete_user_scope(username: str, scope: str) -> bool:
    """Ek scope delete karo."""
    async with _lock:
        path = _scope_file(username, scope)
        if not path.exists():
            return False
        try:
            path.unlink()
        except OSError:
            return False
        return True


async def delete_user_all(username: str) -> bool:
    """User ka poora sync folder hatao (logout wipe / reset)."""
    async with _lock:
        udir = _user_dir(username)
        if not udir.is_dir():
            return False
        try:
            shutil.rmtree(udir)
        except OSError:
            return False
        return True


async def sync_status(username: str, scope: str = "state") -> dict:
    """Sync status — app decide kare local aage hai ya peeche."""
    async with _lock:
        path = _scope_file(username, scope)
        if not path.exists():
            return {"exists": False, "scope": _safe_scope(scope), "updated_at": "", "bytes": 0}
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        record = _parse_record(_read_json(path, {}))
        return {
            "exists": True,
            "scope": _safe_scope(scope),
            "updated_at": record["updated_at"],
            "bytes": size,
        }
