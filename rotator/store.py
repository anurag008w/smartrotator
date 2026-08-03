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
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from . import github_sync

DATA_DIR = github_sync.DATA_DIR
USERS_FILE = DATA_DIR / "users.json"
USAGE_FILE = DATA_DIR / "usage.json"
MANAGED_FILE = DATA_DIR / "managed.json"
PROVIDERS_FILE = DATA_DIR / "providers.json"

_lock = asyncio.Lock()

# in-memory data (files se load hota hai startup pe)
_users: dict[int, dict] = {}          # id -> user dict
_next_id: int = 1
_usage: dict[int, dict[str, dict]] = {}   # user_id -> {day: {"requests":n, "tokens":n}}
_managed: dict = {"provider_models": {}, "provider_order": [], "groups": []}
_custom_providers: list[dict] = []    # admin dashboard se add kiye providers


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
    daily_limit: int = 50
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


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n - 1)).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
# Atomic JSON write
# --------------------------------------------------------------------------
def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


# --------------------------------------------------------------------------
# Init
# --------------------------------------------------------------------------
async def init_db() -> None:
    """Startup pe files load karo (github pull ke BAAD call karo)."""
    global _next_id
    async with _lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        _users.clear()
        raw = _read_json(USERS_FILE, {"next_id": 1, "users": []})
        for u in raw.get("users", []):
            _users[int(u["id"])] = u
            _next_id = max(_next_id, int(u["id"]) + 1)
        if raw.get("next_id"):
            _next_id = max(_next_id, int(raw["next_id"]))

        _usage.clear()
        _usage.update(_read_json(USAGE_FILE, {}))

        _managed.clear()
        m = _read_json(MANAGED_FILE, None)
        if m:
            _managed.update(m)

        _custom_providers.clear()
        _custom_providers.extend(_read_json(PROVIDERS_FILE, []))


def _user_from_dict(u: dict) -> User:
    return User(
        id=int(u["id"]),
        username=u["username"],
        password_hash=u["password_hash"],
        salt=u["salt"],
        api_key=u["api_key"],
        role=u.get("role", "user"),
        daily_limit=int(u.get("daily_limit", 50)),
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
    daily_limit: int = 50,
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


async def rotate_api_key(user_id: int, new_key: str) -> Optional[User]:
    async with _lock:
        u = _users.get(int(user_id))
        if not u:
            return None
        u["api_key"] = new_key
        _persist_users_locked()
        return _user_from_dict(u)


def _persist_users_locked() -> None:
    _write_json(
        USERS_FILE,
        {"next_id": _next_id, "users": sorted(_users.values(), key=lambda x: x["id"])},
    )


# --------------------------------------------------------------------------
# Usage
# --------------------------------------------------------------------------
def _usage_cell(user_id: int, day: str) -> dict:
    return _usage.setdefault(int(user_id), {}).setdefault(day, {"requests": 0, "tokens": 0})


async def get_usage_row(user_id: int, day: str) -> UsageRow:
    async with _lock:
        c = _usage_cell(user_id, day)
        return UsageRow(user_id=int(user_id), day=day, requests=int(c["requests"]), tokens=int(c["tokens"]))


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
    """Request reserve — quota bacha hai toh True (aur count +1)."""
    async with _lock:
        c = _usage_cell(user_id, day)
        if int(c["requests"]) >= int(limit):
            return False
        c["requests"] = int(c["requests"]) + 1
        _persist_usage_locked()
        return True


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
        _write_json(PROVIDERS_FILE, _custom_providers)


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
        _write_json(PROVIDERS_FILE, _custom_providers)


async def remove_custom_provider(name: str) -> bool:
    """Provider hatao. Returns True agar mila tha."""
    async with _lock:
        before = len(_custom_providers)
        _custom_providers[:] = [p for p in _custom_providers if p.get("name") != name]
        if len(_custom_providers) != before:
            _write_json(PROVIDERS_FILE, _custom_providers)
            return True
        return False
