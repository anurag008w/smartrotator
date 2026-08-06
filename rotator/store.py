"""
store.py — JSON-file data store (DB ki jagah).

Users + usage + managed model config ab SQL/Postgres me nahi, `data/`
directory ke JSON files me rehte hain. Ye directory GitHub sync
(github_sync.py) se private repo me push/pull hoti hai:

  data/users.json     — users (password hash, api keys, role, limit)
  data/usage.json     — per-user per-day {requests, tokens}
  data/managed.json   — Model Manager config (models/groups/order)
  data/providers.json — custom providers (admin dashboard se add/remove)

Har mutation ke baad atomic write hota hai (tmp file + rename), taaki
aadha-likha file kabhi na bane.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from . import github_sync

logger = logging.getLogger("smartrotator")

DATA_DIR = github_sync.DATA_DIR
USERS_FILE = DATA_DIR / "users.json"
USAGE_FILE = DATA_DIR / "usage.json"
MANAGED_FILE = DATA_DIR / "managed.json"
PROVIDERS_FILE = DATA_DIR / "providers.json"

_lock = asyncio.Lock()

# in-memory data (files se load hota hai startup pe)
_users: dict[int, dict] = {}          # id -> user dict (api_key PLAINTEXT in-memory)
_next_id: int = 1
_migrations: list[str] = []           # apply ho chuki migrations (users.json me persist)
_usage: dict[int, dict[str, dict]] = {}   # user_id -> {day: {"requests":n, "tokens":n}}
_managed: dict = {"provider_models": {}, "provider_order": [], "groups": []}
_custom_providers: list[dict] = []    # admin dashboard se add kiye providers (keys PLAINTEXT in-memory)

# corrupt JSON detection — users.json corrupt mila toh empty list persist KABHI
# nahi karna (warna ek register pe poora user DB wipe ho jata).
_corrupt_files: set[Path] = set()
_users_corrupt: bool = False


# --------------------------------------------------------------------------
# Data classes (app.py ko SQLAlchemy jaisa hi interface)
# --------------------------------------------------------------------------
@dataclass
class User:
    id: int
    username: str
    password_hash: str
    salt: str
    api_key: str
    role: str = "user"
    daily_limit: int = 30
    monthly_limit: int = 1000
    created_at: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


@dataclass
class UsageRow:
    user_id: int
    day: str
    requests: int = 0
    tokens: int = 0


# --------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------
def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def month_utc() -> str:
    """Current month key — usage.json me month-wise check ke liye ("YYYY-MM")."""
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n - 1)).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
# Atomic JSON write + corrupt-file safety
# --------------------------------------------------------------------------
def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(data, ensure_ascii=False, indent=2))
        fh.flush()
        os.fsync(fh.fileno())  # crash pe aadha-likha content na bache
    tmp.replace(path)
    # rename bhi durable ho — directory fsync (ext4 metadata journaling ke bina)
    try:
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def _read_json(path: Path, default):
    """Read + corrupt-file backup. Corrupt mila toh default return + .corrupt-*
    backup banao (taaki data recover ho sake). `_corrupt_files` me note karo."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _corrupt_files.add(path)
        try:
            backup = path.with_name(path.name + f".corrupt-{int(time.time())}")
            shutil.copy2(path, backup)
            logger.warning(
                "store: %s corrupt/read nahi hua (%s) — backup: %s",
                path.name, exc, backup,
            )
        except OSError:
            logger.error("store: %s corrupt hai aur backup bhi nahi bana (%s)", path.name, exc)
        return default


# --------------------------------------------------------------------------
# At-rest encryption (data/ public repo me track hota hai isliye zaroori)
#
# users.json + providers.json me API keys PLAINTEXT me nahi likhte —
# AES-256-GCM ciphertext store karte hain.
#
# Key: env STORE_ENC_KEY ya hidden file data/.store-enc-key (dotfile —
# github_sync skip karta hai aur .gitignore me bhi hai). Alag key rakhi hai
# taaki usersync ke .sync-enc-key behavior (no-key = strip) se interfere na ho.
#
# In-memory me keys plaintext rehti hain (auth/runtime ke liye zaroori).
# --------------------------------------------------------------------------
_store_key_cache: Optional[bytes] = None


def _store_enc_key() -> bytes:
    """Store encryption key (cached). Env → file → generate+persist."""
    global _store_key_cache
    if _store_key_cache is not None:
        return _store_key_cache
    path = DATA_DIR / ".store-enc-key"
    try:
        env_key = os.environ.get("STORE_ENC_KEY", "").strip()
        if env_key:
            key = hashlib.sha256(env_key.encode("utf-8")).digest()
            _store_key_cache = key
            return key
        if path.exists():
            raw = path.read_text(encoding="utf-8").strip()
            if raw:
                try:
                    k = bytes.fromhex(raw)
                    if len(k) == 32:
                        _store_key_cache = k
                        return k
                except ValueError:
                    pass
                key = hashlib.sha256(raw.encode("utf-8")).digest()
                _store_key_cache = key
                return key
        path.parent.mkdir(parents=True, exist_ok=True)
        generated = secrets.token_hex(32)
        path.write_text(generated + "\n", encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        key = bytes.fromhex(generated)
        _store_key_cache = key
        return key
    except OSError:
        logger.error(
            "store: data/.store-enc-key write nahi ho paya — ephemeral key use "
            "ho rahi hai (restart pe purane encrypted keys unlock nahi honge)"
        )
        key = hashlib.sha256(os.urandom(32)).digest()
        _store_key_cache = key
        return key


def _encrypt_key(plain: str) -> str:
    """Encrypt API key for at-rest storage → "v1.<b64nonce>.<b64ct+tag>"."""
    if not plain:
        return ""
    key = _store_enc_key()
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, plain.encode("utf-8"), b"smartrotator-store-v1")
    return "v1." + base64.urlsafe_b64encode(nonce).decode() + "." + base64.urlsafe_b64encode(ct).decode()


def _decrypt_key(token: str) -> str:
    """Decrypt at-rest key. `v1.` prefix nahi = legacy plaintext (purana file)."""
    if not token or not token.startswith("v1."):
        return token
    key = _store_enc_key()
    try:
        _, nonce_b64, ct_b64 = token.split(".", 2)
        nonce = base64.urlsafe_b64decode(nonce_b64.encode())
        ct = base64.urlsafe_b64decode(ct_b64.encode())
        return AESGCM(key).decrypt(nonce, ct, b"smartrotator-store-v1").decode("utf-8")
    except Exception:  # noqa: BLE001 — galat key / corrupt ciphertext
        logger.warning("store: api key decrypt nahi hua (key rotate hui?) — empty key")
        return ""


# --------------------------------------------------------------------------
# Init
# --------------------------------------------------------------------------
async def init_db() -> None:
    """Startup pe files load karo (github pull ke BAAD call karo)."""
    global _next_id, _users_corrupt
    async with _lock:
        _corrupt_files.clear()
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        _users.clear()
        raw = _read_json(USERS_FILE, {"next_id": 1, "users": []})
        # at-rest keys decrypt karke in-memory plaintext rakho (auth ke liye)
        for u in raw.get("users", []):
            row = dict(u)
            enc = row.pop("api_key_enc", "")
            row["api_key"] = _decrypt_key(enc) if enc else row.get("api_key", "")
            _users[int(row["id"])] = row
            _next_id = max(_next_id, int(row["id"]) + 1)
        if raw.get("next_id"):
            _next_id = max(_next_id, int(raw["next_id"]))
        _users_corrupt = USERS_FILE in _corrupt_files

        _migrations.clear()
        _migrations.extend(raw.get("_migrations", []) or [])
        _apply_pending_migrations_locked()

        _usage.clear()
        _usage.update(_read_json(USAGE_FILE, {}))

        _managed.clear()
        m = _read_json(MANAGED_FILE, None)
        if m:
            _managed.update(m)

        _custom_providers.clear()
        _custom_providers.extend(_decrypt_providers(_read_json(PROVIDERS_FILE, [])))


def _user_from_dict(u: dict) -> User:
    # in-memory dict me api_key plaintext hai (at-rest encryption disk pe hai)
    return User(
        id=int(u["id"]),
        username=u["username"],
        password_hash=u["password_hash"],
        salt=u["salt"],
        api_key=u.get("api_key", ""),
        role=u.get("role", "user"),
        daily_limit=int(u.get("daily_limit", 30)),
        monthly_limit=int(u.get("monthly_limit", 1000)),
        created_at=u.get("created_at", ""),
    )


# --------------------------------------------------------------------------
# Users
# --------------------------------------------------------------------------
async def get_user_by_username(username: str) -> Optional[User]:
    async with _lock:
        for u in _users.values():
            if u["username"] == username:
                return _user_from_dict(u)
        return None


async def get_user_by_api_key(api_key: str) -> Optional[User]:
    async with _lock:
        for u in _users.values():
            if u["api_key"] == api_key:
                return _user_from_dict(u)
        return None


async def get_user_by_id(user_id: int) -> Optional[User]:
    async with _lock:
        u = _users.get(int(user_id))
        return _user_from_dict(u) if u else None


async def create_user(
    username: str,
    password_hash: str,
    salt: str,
    api_key: str,
    daily_limit: int = 30,
    monthly_limit: int = 1000,
    role: str = "user",
) -> User:
    global _next_id
    async with _lock:
        uid = _next_id
        _next_id += 1
        u = {
            "id": uid,
            "username": username,
            "password_hash": password_hash,
            "salt": salt,
            "api_key": api_key,
            "role": role,
            "daily_limit": daily_limit,
            "monthly_limit": monthly_limit,
            "created_at": _now_utc(),
        }
        _users[uid] = u
        _persist_users_locked()
        return _user_from_dict(u)


async def list_users() -> list[User]:
    async with _lock:
        return [_user_from_dict(u) for u in sorted(_users.values(), key=lambda x: x["id"])]


async def count_users() -> int:
    async with _lock:
        return len(_users)


async def set_daily_limit(user_id: int, daily_limit: int) -> Optional[User]:
    async with _lock:
        u = _users.get(int(user_id))
        if not u:
            return None
        u["daily_limit"] = daily_limit
        _persist_users_locked()
        return _user_from_dict(u)


async def set_limits(
    user_id: int,
    daily_limit: Optional[int] = None,
    monthly_limit: Optional[int] = None,
) -> Optional[User]:
    """Daily + monthly limit set karo (jo diya gaya wahi update hota hai)."""
    async with _lock:
        u = _users.get(int(user_id))
        if not u:
            return None
        if daily_limit is not None:
            u["daily_limit"] = int(daily_limit)
        if monthly_limit is not None:
            u["monthly_limit"] = int(monthly_limit)
        _persist_users_locked()
        return _user_from_dict(u)


async def rotate_api_key(user_id: int, new_key: str) -> Optional[User]:
    async with _lock:
        u = _users.get(int(user_id))
        if not u:
            return None
        u["api_key"] = new_key
        _persist_users_locked()
        return _user_from_dict(u)


async def set_password(user_id: int, password_hash: str, salt: str) -> Optional[User]:
    """User ka password update karo (change-password feature)."""
    async with _lock:
        u = _users.get(int(user_id))
        if not u:
            return None
        u["password_hash"] = password_hash
        u["salt"] = salt
        _persist_users_locked()
        return _user_from_dict(u)


async def set_role(user_id: int, role: str) -> Optional[User]:
    """User ka role update karo (admin promote/demote)."""
    async with _lock:
        u = _users.get(int(user_id))
        if not u:
            return None
        if role not in ("admin", "user"):
            raise ValueError("role must be 'admin' or 'user'")
        u["role"] = role
        _persist_users_locked()
        return _user_from_dict(u)


def _persist_users_locked() -> None:
    # Corrupt-file wipe guard: users.json corrupt tha aur abhi bhi koi user nahi
    # hai → kuch bhi write mat karo (warna ek register pe poora DB wipe ho jata).
    if _users_corrupt and not _users:
        raise RuntimeError(
            "users.json corrupt tha (data/users.json.corrupt-* backup dekho) — "
            "empty users list persist nahi kar rahe. File manually recover karke restart karo."
        )
    out_users = []
    for u in sorted(_users.values(), key=lambda x: x["id"]):
        row = dict(u)
        # at-rest encryption — public repo me track hone ke baad bhi keys safe
        row["api_key_enc"] = _encrypt_key(row.pop("api_key", ""))
        out_users.append(row)
    _write_json(
        USERS_FILE,
        {
            "next_id": _next_id,
            "_migrations": _migrations,
            "users": out_users,
        },
    )


# --------------------------------------------------------------------------
# Data migrations (ek-baar wale changes — flag se track, har startup pe nahi)
# --------------------------------------------------------------------------
def _apply_pending_migrations_locked() -> None:
    """Pending migrations apply karo (lock ke andar call karna)."""
    changed = False
    if "limits_30" not in _migrations:
        # sab non-admin users ki daily limit 30 karo (naya default).
        # Admin-set custom limits override na ho — bas ek baar hota hai.
        for u in _users.values():
            if u.get("role", "user") != "admin":
                u["daily_limit"] = 30
        _migrations.append("limits_30")
        changed = True
    if changed:
        _persist_users_locked()


# --------------------------------------------------------------------------
# Usage
# --------------------------------------------------------------------------
def _usage_cell(user_id: int, day: str) -> dict:
    return _usage.setdefault(int(user_id), {}).setdefault(day, {"requests": 0, "tokens": 0})


async def get_usage_row(user_id: int, day: str) -> UsageRow:
    async with _lock:
        c = _usage_cell(user_id, day)
        return UsageRow(user_id=int(user_id), day=day, requests=int(c["requests"]), tokens=int(c["tokens"]))


async def get_usage_month(user_id: int, month: str) -> UsageRow:
    """Poore month ka usage sum karo — "YYYY-MM" prefix wale saare days."""
    async with _lock:
        total_req = 0
        total_tok = 0
        for day, cell in _usage.get(int(user_id), {}).items():
            if day.startswith(month):
                total_req += int(cell["requests"])
                total_tok += int(cell["tokens"])
        return UsageRow(user_id=int(user_id), day=month, requests=total_req, tokens=total_tok)


async def get_usage_month_days(user_id: int, month: str) -> list[UsageRow]:
    """Month ke har din ka usage — bar chart ke liye (sirf jinke data hai)."""
    async with _lock:
        rows = []
        for day in sorted(_usage.get(int(user_id), {}).keys()):
            if day.startswith(month):
                c = _usage[int(user_id)][day]
                rows.append(UsageRow(user_id=int(user_id), day=day, requests=int(c["requests"]), tokens=int(c["tokens"])))
        return rows


async def get_monthly_totals(user_id: int, months: int = 6) -> list[UsageRow]:
    """Last N months ke totals (month → requests/tokens) — month-over-month graph."""
    async with _lock:
        now = datetime.now(timezone.utc)
        prefixes = []
        y, m = now.year, now.month
        for _ in range(months):
            prefixes.append(f"{y:04d}-{m:02d}")
            m -= 1
            if m == 0:
                m = 12
                y -= 1
        prefixes = set(prefixes)
        agg: dict[str, list[int]] = {}
        for day, cell in _usage.get(int(user_id), {}).items():
            prefix = day[:7]
            if prefix in prefixes:
                agg.setdefault(prefix, [0, 0])
                agg[prefix][0] += int(cell["requests"])
                agg[prefix][1] += int(cell["tokens"])
        rows = []
        y, m = now.year, now.month
        for _ in range(months):
            key = f"{y:04d}-{m:02d}"
            a = agg.get(key, [0, 0])
            rows.append(UsageRow(user_id=int(user_id), day=key, requests=a[0], tokens=a[1]))
            m -= 1
            if m == 0:
                m = 12
                y -= 1
        return rows


async def get_usage_between(user_id: int, days: int = 7) -> list[UsageRow]:
    async with _lock:
        start = _days_ago(days)
        rows = []
        for day in sorted(_usage.get(int(user_id), {}).keys()):
            if day >= start:
                c = _usage[int(user_id)][day]
                rows.append(UsageRow(user_id=int(user_id), day=day, requests=int(c["requests"]), tokens=int(c["tokens"])))
        return rows


async def reserve_quota(user_id: int, day: str, limit: int) -> bool:
    """Request reserve — daily quota bacha hai toh True (aur count +1)."""
    async with _lock:
        c = _usage_cell(user_id, day)
        if int(c["requests"]) >= int(limit):
            return False
        c["requests"] = int(c["requests"]) + 1
        _persist_usage_locked()
        return True


async def reserve_quota_with_monthly(
    user_id: int,
    day: str,
    month: str,
    daily_limit: int,
    monthly_limit: int,
) -> bool:
    """Daily + monthly dono quota check karke reserve karo (atomic).

    Admin unlimited hai — unke liye reserve_unlimited use karo.
    """
    async with _lock:
        c = _usage_cell(user_id, day)
        # monthly total: is month ke saare days (aaj ka cell included)
        month_total = 0
        for d, cell in _usage.get(int(user_id), {}).items():
            if d.startswith(month):
                month_total += int(cell["requests"])
        if int(monthly_limit) > 0 and month_total >= int(monthly_limit):
            return False
        if int(c["requests"]) >= int(daily_limit):
            return False
        c["requests"] = int(c["requests"]) + 1
        _persist_usage_locked()
        return True


async def reserve_unlimited(user_id: int, day: str) -> None:
    """Admin ke liye — koi limit check nahi, par usage TRACK hota hai."""
    async with _lock:
        c = _usage_cell(user_id, day)
        c["requests"] = int(c["requests"]) + 1
        _persist_usage_locked()


async def set_usage(user_id: int, day: str, requests: int, tokens: int = 0) -> None:
    """Usage directly set karo (admin/debug/tests ke liye)."""
    async with _lock:
        _usage.setdefault(int(user_id), {})[day] = {
            "requests": int(requests),
            "tokens": int(tokens),
        }
        _persist_usage_locked()


async def refund_quota(user_id: int, day: str) -> None:
    async with _lock:
        c = _usage_cell(user_id, day)
        c["requests"] = max(0, int(c["requests"]) - 1)
        _persist_usage_locked()


async def record_tokens(user_id: int, day: str, tokens: int) -> None:
    async with _lock:
        c = _usage_cell(user_id, day)
        c["tokens"] = int(c["tokens"]) + int(tokens)
        _persist_usage_locked()


def _persist_usage_locked() -> None:
    _write_json(USAGE_FILE, _usage)


# --------------------------------------------------------------------------
# Model Manager — managed config (dashboard se save hota hai)
# --------------------------------------------------------------------------
async def load_managed_config() -> dict:
    async with _lock:
        return {
            "provider_models": dict(_managed.get("provider_models", {})),
            "provider_order": list(_managed.get("provider_order", [])),
            "groups": list(_managed.get("groups", [])),
        }


async def save_managed_config(
    *,
    provider_models: dict[str, list[str]],
    provider_order: list[str],
    groups: list[dict],
) -> None:
    async with _lock:
        _managed["provider_models"] = provider_models or {}
        _managed["provider_order"] = provider_order or []
        _managed["groups"] = groups or []
        _write_json(MANAGED_FILE, _managed)


# --------------------------------------------------------------------------
# Custom providers — admin dashboard se add/remove (live models feature)
# --------------------------------------------------------------------------
async def list_custom_providers() -> list[dict]:
    """Dashboard-added providers (name/type/base_url/keys/models/enabled)."""
    async with _lock:
        return [dict(p) for p in _custom_providers]


async def get_custom_provider(name: str) -> Optional[dict]:
    async with _lock:
        for p in _custom_providers:
            if p.get("name") == name:
                return dict(p)
        return None


async def save_custom_providers(providers: list[dict]) -> None:
    """Poori custom providers list replace karo (atomic write)."""
    async with _lock:
        _custom_providers.clear()
        _custom_providers.extend(providers)
        _persist_providers_locked()


def _providers_for_disk() -> list[dict]:
    """In-memory providers (plaintext keys) → disk copy (encrypted keys)."""
    out = []
    for p in _custom_providers:
        row = dict(p)
        row["api_keys"] = [_encrypt_key(k) for k in (p.get("api_keys") or [])]
        out.append(row)
    return out


def _decrypt_providers(rows) -> list[dict]:
    """Disk copy (encrypted keys) → in-memory (plaintext keys)."""
    out = []
    for p in rows or []:
        row = dict(p)
        row["api_keys"] = [_decrypt_key(k) for k in (p.get("api_keys") or [])]
        out.append(row)
    return out


def _persist_providers_locked() -> None:
    _write_json(PROVIDERS_FILE, _providers_for_disk())


async def upsert_custom_provider(provider: dict) -> None:
    """Ek provider add/update karo (naam se match)."""
    async with _lock:
        name = provider.get("name", "")
        replaced = False
        for i, p in enumerate(_custom_providers):
            if p.get("name") == name:
                _custom_providers[i] = provider
                replaced = True
                break
        if not replaced:
            _custom_providers.append(provider)
        _persist_providers_locked()


async def remove_custom_provider(name: str) -> bool:
    """Provider hatao. Returns True agar mila tha."""
    async with _lock:
        before = len(_custom_providers)
        _custom_providers[:] = [p for p in _custom_providers if p.get("name") != name]
        if len(_custom_providers) != before:
            _persist_providers_locked()
            return True
        return False
