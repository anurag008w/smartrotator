"""
app.py — SmartRotator service: auth + per-user quota + Web UI + API gateway.

┌─────────────────────────────────────────────────────────────────┐
│  APP  ──login──▶ /auth/login → JWT token (koi API key nahi)     │
│       ──Bearer JWT──▶ /v1/chat/completions ──▶ per-user quota    │
│       ──▶ rotator (multi-key round-robin) ──▶ LLM providers      │
│                                                                  │
│  Browser ──▶ /dashboard  (login, chat, usage, settings)          │
│  (optional) sk- API key OpenAI SDK ke liye bhi chalti hai        │
└─────────────────────────────────────────────────────────────────┘

OpenAI-compatible:
  POST /v1/chat/completions   (Authorization: Bearer JWT-login-token ya sk-USER_KEY)
  GET  /v1/models
  GET  /health                (keep-alive)
  GET  /status                (key/proxy health — public)
  GET  /                      (dashboard SPA)

Auth:
  POST /auth/register  POST /auth/login  GET /auth/me
  POST /auth/rotate-key
Admin:
  GET  /admin/users    POST /admin/users/{id}/limit
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import github_sync
from . import store as database
from . import usersync
from .catalog import MODEL_CATALOG
from .auth import (
    create_jwt,
    decode_jwt,
    generate_api_key,
    hash_password,
    is_scrypt_hash,
    is_user_api_key,
    verify_password,
)
from .providers import (
    AllProvidersExhausted,
    ChatMessage,
    ImageInput,
    ProviderError,
    RateLimitError,
    fetch_live_models,
)
from .router import Rotator

logger = logging.getLogger("smartrotator")

CONFIG_PATH = os.environ.get("ROTATOR_CONFIG", "config.yaml")

# config.yaml har request pe disk se YAML parse karna slow hai (~25ms) —
# 30s TTL cache rakh do. Runtime me config.yaml change nahi hota (managed
# config DB me rehta hai), isliye 30s freshness kaafi hai.
CONFIG_CACHE_TTL = 30.0
_config_cache: dict = {"t": 0.0, "data": {}}


def _load_config() -> dict:
    now = time.time()
    cached = _config_cache
    if now - cached["t"] < CONFIG_CACHE_TTL:
        return cached["data"]
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    _config_cache["t"] = now
    _config_cache["data"] = data
    return data


# --------------------------------------------------------------------------
# App lifecycle
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    rotator = Rotator(config_path=CONFIG_PATH)
    app.state.rotator = rotator

    # 1) GitHub sync: restart pe PEHLE pull (data restore, DB ki zaroorat nahi)
    await asyncio.to_thread(github_sync.pull_data)

    # 2) JSON store load karo (users + usage + managed config + custom providers)
    await database.init_db()

    # 2b) Custom providers (dashboard se add kiye) apply karo
    custom = await database.list_custom_providers()
    rotator.apply_custom_providers(custom)

    # 3) Model Manager: dashboard ka saved config (models/groups/order) apply karo
    managed = await database.load_managed_config()
    rotator.apply_managed(managed)

    # live models cache (provider APIs se fetch, 5 min TTL)
    app.state.live_models_cache: dict[str, dict] = {}  # name -> {fetched_at, models, error}

    # 4) background sync task — har 3 min me data change ho toh push
    stop_event = asyncio.Event()
    sync_task = asyncio.create_task(github_sync.sync_loop(stop_event))
    app.state.github_sync_task = sync_task
    app.state.github_sync_stop = stop_event

    yield

    stop_event.set()
    sync_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass
    # shutdown se pehle ek aakhri push (latest data GitHub pe)
    if github_sync.is_enabled():
        await asyncio.to_thread(github_sync.push_data)
    await rotator.aclose()


app = FastAPI(
    title="SmartRotator",
    description="Multi-provider LLM gateway with per-user quotas + dashboard.",
    version="0.3.0",
    lifespan=lifespan,
)


# CORS: the LevelUp mobile/web app calls this server directly (login + /v1
# gateway). Native apps bypass CORS, but the web preview needs permissive
# headers. Auth is enforced by JWT/sk- keys, so a wildcard origin is fine.
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Global exception handler — koi bhi unhandled error raw 500 na dikhe
#
# Pehle provider parsing me kuch edge-cases (galat schema, non-JSON body,
# 200-pe-error) unhandled exceptions fek dete the → user ko plain
# "Internal Server Error" milta tha. Root causes ab providers.py me fix hain,
# par ye safety net rakhna acha hai: koi bhi unexpected exception log ho
# aur JSON body ke saath aaye (raw 500 ki jagah).
# --------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):  # noqa: BLE001
    logger.exception(
        "Unhandled error on %s %s: %s", request.method, request.url.path, exc
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                "Server me ek unexpected error aayi. Ye bug ho sakta hai — "
                "logs me traceback dekh kar batao, fix kar dungi. "
                "Agar ye provider response ki wajah se hai, thodi der baad try karo."
            )
        },
    )


def _auth_settings() -> dict:
    cfg = _load_config().get("auth", {}) or {}
    secret = os.environ.get(cfg.get("jwt_secret_env", "JWT_SECRET"), cfg.get("jwt_secret", ""))
    if not secret:
        secret = os.environ.get("JWT_SECRET", "dev-secret-change-me-please-set-env-JWT_SECRET-32bytes")
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "default_daily_limit": int(cfg.get("default_daily_limit", 50)),
        "jwt_secret": secret,
        "jwt_hours": int(cfg.get("jwt_hours", 24)),
        "admin_usernames": _admin_usernames(cfg),
    }


def _admin_usernames(cfg: dict) -> set[str]:
    env_name = cfg.get("admin_usernames_env", "ADMIN_USERS")
    from_env = [u.strip() for u in os.environ.get(env_name, "").split(",") if u.strip()]
    from_cfg = [u.strip() for u in cfg.get("admin_usernames", []) if u.strip()]
    return set(from_env + from_cfg)


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------
class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[dict] = Field(..., min_length=1)
    max_tokens: int = 8192
    temperature: float = 0.7
    stream: bool = False  # accepted for compatibility; returns non-streamed
    models: Optional[list[str]] = None  # UI multi-select: rotation inhi models se
    tools: Optional[list[dict]] = None  # function calling (OpenAI format)
    tool_choice: Optional[dict] = None


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class SyncStateRequest(BaseModel):
    """App-side push-authoritative: poora user state server ko bheja jata hai."""
    state: dict = {}
    updated_at: str = ""


class SetLimitRequest(BaseModel):
    daily_limit: int = Field(..., ge=1, le=100000)


class ChangePasswordRequest(BaseModel):
    old_password: str = ""
    new_password: str = Field(..., min_length=6, max_length=128)


class SetRoleRequest(BaseModel):
    role: str = Field(..., pattern="^(admin|user)$")


class GroupMember(BaseModel):
    provider: str
    model: str = ""            # backward compat: single model
    models: list[str] = []     # naya: models queue (member me multiple models)
    keys: list[int] = []       # selected key indices (empty = saari keys)


class ModelGroupInput(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    label: str = ""
    enabled: bool = True
    members: list[GroupMember] = []


class SaveModelsRequest(BaseModel):
    provider_models: dict[str, list[str]] = {}
    provider_order: list[str] = []
    groups: list[ModelGroupInput] = []


class CustomProviderInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    type: str = "openai"          # gemini | openai
    base_url: str = ""
    api_keys: list[str] = []
    models: list[str] = []
    enabled: bool = True


# --------------------------------------------------------------------------
# Public endpoints
# --------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(request: Request):
    rotator: Rotator = request.app.state.rotator
    # live models (cache me jo fresh hai) bhi merge karo — mobile app jaisa
    live = _live_models_for_list(request)
    data = [
        {"id": m["id"], "object": "model", "owned_by": m["provider"], "type": m["type"]}
        for m in rotator.models()
    ]
    # live models jo configured list me nahi — unhe bhi dikhao (tag: live)
    seen = {m["id"] for m in data}
    for m in live:
        if m["id"] not in seen:
            data.append(
                {"id": m["id"], "object": "model", "owned_by": m["provider"], "type": "live"}
            )
    return {"object": "list", "data": data, "default_model": rotator.default_model}


@app.get("/v1/models/raw")
async def list_models_raw(request: Request):
    """Raw (bina live merge) models — dashboard picker ke liye lighter."""
    rotator: Rotator = request.app.state.rotator
    data = [
        {"id": m["id"], "object": "model", "owned_by": m["provider"], "type": m["type"]}
        for m in rotator.models()
    ]
    return {"object": "list", "data": data, "default_model": rotator.default_model}


@app.get("/status")
async def status(request: Request):
    rotator: Rotator = request.app.state.rotator
    return rotator.status()


# --------------------------------------------------------------------------
# Auth endpoints
# --------------------------------------------------------------------------
@app.post("/auth/register")
async def register(req: RegisterRequest):
    settings = _auth_settings()
    if not settings["enabled"]:
        raise HTTPException(status_code=403, detail="Auth disabled")

    existing = await database.get_user_by_username(req.username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")

    # role: env ADMIN_USERS me naam ho → admin. Env set hai toh koi bhi
    # random pehla user auto-admin NAHI banega (sirf owner). Env khali
    # (local dev / self-host) pe pehla registered user admin hota hai.
    if req.username in settings["admin_usernames"]:
        role = "admin"
    elif settings["admin_usernames"]:
        role = "user"
    else:
        count = await database.count_users()
        role = "admin" if count == 0 else "user"

    password_hash, salt = hash_password(req.password)
    user = await database.create_user(
        username=req.username,
        password_hash=password_hash,
        salt=salt,
        api_key=generate_api_key(),
        daily_limit=settings["default_daily_limit"],
        role=role,
    )

    token = create_jwt(
        user.id, user.username, user.role, settings["jwt_secret"], settings["jwt_hours"]
    )
    return {
        "token": token,
        "api_key": user.api_key,
        "user": _user_public(user),
        "is_super_admin": _is_super_admin(user, settings),
    }


@app.post("/auth/login")
async def login(req: LoginRequest):
    settings = _auth_settings()
    if not settings["enabled"]:
        raise HTTPException(status_code=403, detail="Auth disabled")

    user = await database.get_user_by_username(req.username)
    if not user or not verify_password(req.password, user.password_hash, user.salt):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Purane PBKDF2 hash ko scrypt pe migrate karo (login pe ek baar re-hash)
    if not is_scrypt_hash(user.password_hash):
        new_hash, _ = hash_password(req.password, user.salt)
        migrated = await database.set_password(user.id, new_hash, user.salt)
        if migrated is not None:
            user = migrated

    token = create_jwt(
        user.id, user.username, user.role, settings["jwt_secret"], settings["jwt_hours"]
    )
    return {
        "token": token,
        "api_key": user.api_key,
        "user": _user_public(user),
        "is_super_admin": _is_super_admin(user, settings),
    }


@app.get("/auth/me")
async def auth_me(request: Request):
    settings = _auth_settings()
    user = await _require_user(request, settings)
    today = database.today_utc()
    row = await database.get_usage_row(user.id, today)
    history = await database.get_usage_between(user.id, 7)
    return {
        "user": _user_public(user),
        "is_super_admin": _is_super_admin(user, settings),
        "api_key": user.api_key,
        "today": {"day": today, "requests": row.requests, "tokens": row.tokens},
        "history": [
            {"day": h.day, "requests": h.requests, "tokens": h.tokens} for h in history
        ],
    }


@app.post("/auth/rotate-key")
async def rotate_key(request: Request):
    settings = _auth_settings()
    user = await _require_user(request, settings)
    db_user = await database.rotate_api_key(user.id, generate_api_key())
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"api_key": db_user.api_key}


@app.post("/auth/change-password")
async def change_password(req: ChangePasswordRequest, request: Request):
    """Apna password badlo — old password verify karke (ya admin pehli baar set)."""
    settings = _auth_settings()
    user = await _require_user(request, settings)

    db_user = await database.get_user_by_id(user.id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # old password check — optional agar old_password khali hai (first-time set)
    if req.old_password:
        if not verify_password(req.old_password, db_user.password_hash, db_user.salt):
            raise HTTPException(status_code=401, detail="Old password galat hai")

    password_hash, salt = hash_password(req.new_password)
    updated = await database.set_password(user.id, password_hash, salt)
    if updated is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True, "message": "Password update ho gaya"}


# --------------------------------------------------------------------------
# Sync endpoints (offline-first app backup — per-user, push-authoritative)
# --------------------------------------------------------------------------
@app.get("/sync/status")
async def sync_status(request: Request, scope: str = "state"):
    """User ke ek scope ka sync status — app decide kare local aage hai ya peeche."""
    settings = _auth_settings()
    user = await _require_user(request, settings)
    status = await usersync.sync_status(user.username, scope)
    return {"username": user.username, **status}


@app.get("/sync/scopes")
async def sync_scopes(request: Request):
    """User folder me kaunse scopes sync hain (fresh install pe pull list)."""
    settings = _auth_settings()
    user = await _require_user(request, settings)
    scopes = await usersync.list_user_scopes(user.username)
    return {"username": user.username, "scopes": scopes}


@app.get("/sync/state")
async def sync_get_state(request: Request, scope: str = "state"):
    """User ke ek scope ka saved data (fresh install / naye device pe pull)."""
    settings = _auth_settings()
    user = await _require_user(request, settings)
    record = await usersync.get_user_scope(user.username, scope)
    if record is None:
        return {"username": user.username, "scope": scope, "exists": False, "updated_at": "", "state": {}}
    return {"username": user.username, "scope": scope, "exists": True, **record}


@app.put("/sync/state")
async def sync_put_state(req: SyncStateRequest, request: Request, scope: str = "state"):
    """User ke ek scope ka data save karo — server bas store karta hai (last-write-wins)."""
    settings = _auth_settings()
    user = await _require_user(request, settings)
    saved = await usersync.save_user_scope(user.username, scope, req.state, req.updated_at)
    return {"username": user.username, **saved}


@app.delete("/sync/state")
async def sync_delete_state(request: Request, scope: str = "state"):
    """Ek scope delete karo. Agar scope='*' ho toh poora user data wipe."""
    settings = _auth_settings()
    user = await _require_user(request, settings)
    if scope == "*":
        deleted = await usersync.delete_user_all(user.username)
    else:
        deleted = await usersync.delete_user_scope(user.username, scope)
    return {"username": user.username, "scope": scope, "deleted": deleted}


# --------------------------------------------------------------------------
# Admin endpoints
# --------------------------------------------------------------------------
@app.get("/admin/users")
async def admin_users(request: Request):
    settings = _auth_settings()
    user = await _require_user(request, settings)
    if not _is_super_admin(user, settings):
        raise HTTPException(status_code=403, detail="Admin only")

    users = await database.list_users()
    today = database.today_utc()
    out = []
    for u in users:
        row = await database.get_usage_row(u.id, today)
        out.append(
            {
                "id": u.id,
                "username": u.username,
                "role": u.role,
                "api_key": u.api_key[:14] + "...",
                "daily_limit": u.daily_limit,
                "today_requests": row.requests,
                "created_at": u.created_at,
            }
        )
    return {"users": out}


@app.post("/admin/users/{user_id}/limit")
async def admin_set_limit(user_id: int, req: SetLimitRequest, request: Request):
    settings = _auth_settings()
    user = await _require_user(request, settings)
    if not _is_super_admin(user, settings):
        raise HTTPException(status_code=403, detail="Admin only")

    target = await database.set_daily_limit(user_id, req.daily_limit)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True, "username": target.username, "daily_limit": target.daily_limit}


@app.post("/admin/users/{user_id}/role")
async def admin_set_role(user_id: int, req: SetRoleRequest, request: Request):
    """Kisi user ko admin promote/demote karo."""
    await _require_admin(request)

    target = await database.set_role(user_id, req.role)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True, "username": target.username, "role": target.role}


# --------------------------------------------------------------------------
# Model Manager admin endpoints (dashboard Models tab)
# --------------------------------------------------------------------------
def _is_super_admin(user, settings: dict) -> bool:
    """Admin PANEL ka access — sirf env/ADMIN_USERS me naam wale users ko.

    - Agar ADMIN_USERS env/config set hai (Render pe): sirf wahi users panel
      dekhenge. Role se promote kiye hue admins panel NAHI dekh payenge
      (par quota unlimited rahega).
    - Agar env/config khali hai (local dev): role-admin fallback — pehla
      registered user hi admin.
    """
    admins = settings.get("admin_usernames") or set()
    if admins:
        return user.username in admins
    return user.is_admin


def _is_admin(user, settings: dict) -> bool:
    """Koi bhi admin — env wala super admin YA role=admin wala.

    Models panel (group editor + live models) ke liye dono ko access hai.
    """
    return _is_super_admin(user, settings) or user.is_admin


async def _require_admin(request: Request):
    """SIRF super admin (env ADMIN_USERS) — Users & Quotas, providers, enc-key."""
    settings = _auth_settings()
    user = await _require_user(request, settings)
    if not _is_super_admin(user, settings):
        raise HTTPException(status_code=403, detail="Admin only")
    return user


async def _require_any_admin(request: Request):
    """Koi bhi admin — Models panel (super admin + promoted admin)."""
    settings = _auth_settings()
    user = await _require_user(request, settings)
    if not _is_admin(user, settings):
        raise HTTPException(status_code=403, detail="Admin only")
    return user


@app.get("/admin/models")
async def admin_models(request: Request):
    """Dashboard ke liye: catalog + current managed config + provider/key counts."""
    await _require_any_admin(request)

    rotator: Rotator = request.app.state.rotator
    managed = await database.load_managed_config()

    # provider state se: name, type, key count, configured models + LIVE models
    cache = _live_cache(request)
    providers = []
    for st in rotator.providers:
        entry = cache.get(st.cfg.name) or {}
        providers.append(
            {
                "name": st.cfg.name,
                "type": st.cfg.ptype,
                "key_count": len(st.cfg.keys),
                "keys": [
                    {"index": i, "preview": _key_preview(k)}
                    for i, k in enumerate(st.cfg.keys)
                ],
                "configured_models": list(st.cfg.models),
                "live_models": [m["id"] for m in entry.get("models", [])],
                "live_error": entry.get("error"),
            }
        )
    # providers jo config me hain par abhi incomplete (keys/models nahi) — bhi dikhao
    raw_cfg = _load_config().get("providers", [])
    known = {p["name"] for p in providers}
    for p in raw_cfg:
        name = p.get("name", "")
        if name not in known:
            providers.append(
                {
                    "name": name,
                    "type": p.get("type", "openai"),
                    "key_count": 0,
                    "configured_models": [m for m in p.get("models", []) if not str(m).startswith("PASTE_")],
                }
            )

    return {
        "catalog": MODEL_CATALOG,
        "managed": managed,
        "providers": providers,
        "default_model": rotator.default_model,
    }


@app.put("/admin/models")
async def admin_save_models(req: SaveModelsRequest, request: Request):
    """Dashboard se save — models/groups/order apply karo (config.yaml touch nahi)."""
    await _require_any_admin(request)

    # groups ko pydantic → plain dicts (member: provider + models queue + keys)
    groups = []
    for g in req.groups:
        members = []
        for m in g.members:
            member_models = [x for x in (m.models or ([m.model] if m.model else [])) if x]
            if not member_models:
                continue
            members.append(
                {
                    "provider": m.provider,
                    "models": member_models,
                    "keys": [i for i in m.keys if isinstance(i, int) and i >= 0],
                }
            )
        if members:
            groups.append(
                {
                    "id": g.id,
                    "label": g.label or g.id,
                    "enabled": g.enabled,
                    "members": members,
                }
            )
    await database.save_managed_config(
        provider_models=req.provider_models,
        provider_order=req.provider_order,
        groups=groups,
    )

    rotator: Rotator = request.app.state.rotator
    rotator.apply_managed(await database.load_managed_config())
    return {"ok": True, "message": "Models config save + apply ho gayi"}


# --------------------------------------------------------------------------
# Live models cache (provider APIs se fetch — 5 min TTL)
# --------------------------------------------------------------------------
LIVE_MODELS_TTL = 300  # seconds (5 min)


def _key_preview(key: str) -> str:
    """API key ka masked preview — aage ka 6 + piche ka 4 char."""
    if not key:
        return ""
    if len(key) <= 10:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


def _live_cache(request: Request) -> dict:
    cache = getattr(request.app.state, "live_models_cache", None)
    if cache is None:
        cache = {}
        request.app.state.live_models_cache = cache
    return cache


async def _refresh_live_models(request: Request, name: str, force: bool = False) -> dict:
    """Ek provider ke liye live models fetch karo + cache me daalo.

    Provider config rotator (ya store) se milta hai — pehle custom provider
    (dashboard se add), phir config.yaml wala. Returns cache entry.
    """
    cache = _live_cache(request)
    now = time.time()
    entry = cache.get(name)
    if entry and not force and (now - entry.get("fetched_at", 0)) < LIVE_MODELS_TTL:
        return entry

    rotator: Rotator = request.app.state.rotator
    # provider info dhoondo: custom store → rotator state → raw config
    ptype, base_url, api_keys, models = None, None, [], []
    for st in rotator.providers:
        if st.cfg.name == name:
            ptype, base_url, api_keys = st.cfg.ptype, st.cfg.base_url, list(st.cfg.keys)
            models = list(st.cfg.models)
            break
    if ptype is None:
        for p in _load_config().get("providers", []):
            if p.get("name") == name:
                ptype = p.get("type", "openai")
                base_url = p.get("base_url")
                api_keys = [k.strip() for k in p.get("api_keys", []) if not k.startswith("PASTE_")]
                models = [m for m in p.get("models", []) if not str(m).startswith("PASTE_")]
                break
    if ptype is None:
        # custom store me ho sakta hai (rotator me models empty ho toh skip hua)
        cp = await database.get_custom_provider(name)
        if cp:
            ptype = cp.get("type", "openai")
            base_url = cp.get("base_url")
            api_keys = cp.get("api_keys", [])
            models = cp.get("models", [])

    if ptype is None or not api_keys:
        entry = {"fetched_at": now, "models": [], "error": "provider not configured / no keys"}
        cache[name] = entry
        return entry

    try:
        live = await fetch_live_models(name, ptype, base_url, api_keys)
        entry = {
            "fetched_at": now,
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "provider": m.provider,
                    "context_length": m.context_length,
                    "supports_streaming": m.supports_streaming,
                    "supports_vision": m.supports_vision,
                    "supports_reasoning": m.supports_reasoning,
                    "supports_tool_calling": m.supports_tool_calling,
                    "is_free": m.is_free,
                }
                for m in live
            ],
            "error": None,
            "configured_models": models,
        }
    except Exception as exc:  # noqa: BLE001 — cache me error hi rakh do
        entry = {
            "fetched_at": now,
            "models": [],
            "error": str(exc)[:500],
            "configured_models": models,
        }
    cache[name] = entry
    return entry


def _live_models_for_list(request: Request) -> list[dict]:
    """/v1/models ke liye cached live models (TTL check)."""
    cache = _live_cache(request)
    now = time.time()
    out = []
    for name, entry in cache.items():
        if (now - entry.get("fetched_at", 0)) > LIVE_MODELS_TTL:
            continue
        for m in entry.get("models", []):
            out.append({"id": m["id"], "provider": m.get("provider") or name})
    return out


# --------------------------------------------------------------------------
# Custom providers admin endpoints (dashboard se add/remove)
# --------------------------------------------------------------------------
@app.get("/admin/providers")
async def admin_providers(request: Request):
    """Custom providers list + live models cache status."""
    await _require_admin(request)
    custom = await database.list_custom_providers()
    # public view: keys ko mask karo (dashboard pe sirf count dikhega)
    public = []
    for p in custom:
        public.append(
            {
                "name": p.get("name"),
                "type": p.get("type", "openai"),
                "base_url": p.get("base_url", ""),
                "key_count": len(p.get("api_keys", [])),
                "models": p.get("models", []),
                "enabled": p.get("enabled", True),
            }
        )
    cache = _live_cache(request)
    status = []
    for name, entry in cache.items():
        status.append(
            {
                "name": name,
                "fetched_at": entry.get("fetched_at"),
                "count": len(entry.get("models", [])),
                "error": entry.get("error"),
            }
        )
    return {"providers": public, "cache": status}


@app.post("/admin/providers")
async def admin_add_provider(req: CustomProviderInput, request: Request):
    """Naya provider add karo (ya same name pe update) + live apply."""
    await _require_admin(request)

    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Provider name required")
    ptype = req.type.strip().lower()
    if ptype not in ("gemini", "openai"):
        raise HTTPException(status_code=400, detail="type must be 'gemini' or 'openai'")
    if ptype == "openai" and not req.base_url.strip():
        raise HTTPException(status_code=400, detail="openai provider needs base_url")

    keys = [k.strip() for k in req.api_keys if k.strip()]
    if not keys:
        raise HTTPException(status_code=400, detail="At least one API key required")
    models = [m.strip() for m in req.models if m.strip()]

    provider = {
        "name": name,
        "type": ptype,
        "base_url": req.base_url.strip(),
        "api_keys": keys,
        "models": models,
        "enabled": req.enabled,
    }
    await database.upsert_custom_provider(provider)

    # live apply karo
    rotator: Rotator = request.app.state.rotator
    custom = await database.list_custom_providers()
    rotator.apply_custom_providers(custom)

    # naye provider ke liye live models fetch karo (background me nahi — turant)
    try:
        entry = await _refresh_live_models(request, name, force=True)
    except Exception:  # noqa: BLE001
        entry = None

    return {
        "ok": True,
        "message": f"Provider '{name}' add/update ho gaya",
        "live": (entry or {}).get("models", []),
        "live_error": (entry or {}).get("error"),
    }


@app.delete("/admin/providers/{name}")
async def admin_delete_provider(name: str, request: Request):
    """Provider hatao + live apply."""
    await _require_admin(request)
    removed = await database.remove_custom_provider(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")
    rotator: Rotator = request.app.state.rotator
    custom = await database.list_custom_providers()
    rotator.apply_custom_providers(custom)
    _live_cache(request).pop(name, None)
    return {"ok": True, "message": f"Provider '{name}' delete ho gaya"}


@app.post("/admin/providers/{name}/fetch-models")
async def admin_fetch_models(name: str, request: Request):
    """Provider API se LIVE models fetch karo (force refresh + cache update)."""
    await _require_admin(request)
    entry = await _refresh_live_models(request, name, force=True)
    if entry.get("error"):
        raise HTTPException(status_code=502, detail=f"Live fetch failed: {entry['error']}")
    return {
        "ok": True,
        "provider": name,
        "models": entry.get("models", []),
        "configured_models": entry.get("configured_models", []),
    }


@app.post("/admin/providers/refresh-all")
async def admin_refresh_all_models(request: Request):
    """Saare configured providers ke live models refresh karo (force)."""
    await _require_any_admin(request)
    rotator: Rotator = request.app.state.rotator
    names = [st.cfg.name for st in rotator.providers]
    results = []
    for name in names:
        try:
            entry = await _refresh_live_models(request, name, force=True)
            results.append({"name": name, "count": len(entry.get("models", [])), "error": entry.get("error")})
        except Exception as exc:  # noqa: BLE001
            results.append({"name": name, "count": 0, "error": str(exc)})
    return {"ok": True, "results": results}


@app.get("/admin/sync/enc-key")
async def admin_get_enc_key(request: Request):
    """SYNC_ENC_KEY secret dekh lo (yaad rakhne ki zaroorat nahi).

    Secret 3 sources se milta hai:
      - env SYNC_ENC_KEY (Render secret) — sabse high priority
      - file data/.sync-enc-key (auto-generated hidden file) — GitHub sync skip
      - nahi hai toh abhi generate + persist
    Ye secret sirf ADMIN ko dikhta hai.
    """
    await _require_admin(request)
    secret, source = usersync.get_or_create_secret()
    return {
        "set": True,
        "source": source,
        "secret": secret,
        "hint": "Isko Render me SYNC_ENC_KEY secret ke roop me daal sakte ho (optional — file-based bhi persist karta hai).",
    }


@app.post("/admin/sync/enc-key/rotate")
async def admin_rotate_enc_key(request: Request):
    """Naya secret banao aur file me save karo.

    WARNING: purana encrypted data nayi key se unlock nahi hoga — users ko
    phir se login + re-sync karna padega. Sirf tabhi use karo jab zaroori ho.
    """
    await _require_admin(request)
    secret, source = usersync.rotate_secret()
    return {
        "set": True,
        "source": source,
        "secret": secret,
        "warning": "Nayi key file me set ho gayi. Purana encrypted sync data ab unlock nahi hoga — users re-login karein.",
        "env_override": bool(os.environ.get("SYNC_ENC_KEY", "").strip()),
    }


# --------------------------------------------------------------------------
# Main LLM gateway (with per-user quota)
# --------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    rotator: Rotator = request.app.state.rotator
    settings = _auth_settings()

    try:
        messages = [_convert_message(m) for m in req.messages]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- auth + quota (sirf jab auth enabled ho) ---
    user = None
    if settings["enabled"]:
        user = await _authenticate(request, settings)
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="Missing/invalid token. Login karo ya apni sk- API key bhejo.",
            )
        quota_ok = await _reserve_quota(user)
        if not quota_ok:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Aapka daily quota khatam ho gaya. Kal reset hoga."
                ),
            )

    try:
        result = await rotator.chat(
            messages,
            model=req.model or None,
            models=req.models,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            tools=req.tools,
            tool_choice=req.tool_choice,
        )
    except RateLimitError as exc:
        # provider rate-limit — user ko raw message NAHI dikhate
        if user:
            await _refund_quota(user)
        logger.warning("chat: rate limited upstream: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Server abhi bahut load me hai, kuch minute baad try karo.",
        ) from exc
    except AllProvidersExhausted as exc:
        # saare providers ki keys/models fail
        if user:
            await _refund_quota(user)
        logger.error("chat: all providers exhausted: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "Something went wrong — saare AI providers abhi busy/ exhausted hain. "
                "Thodi der baad try karo, ya apna configured provider use karo. "
                "(sahi provider error `/status` ya admin panel me dikhta hai)"
            ),
        ) from exc
    except ProviderError as exc:
        # koi aur provider error (config galat, network, etc.)
        if user:
            await _refund_quota(user)
        logger.error("chat: provider error: %s", exc)
        status_code = exc.status_code if exc.status_code else 502
        raise HTTPException(
            status_code=status_code,
            detail="Server ko AI provider se connect karne me problem aayi. Kuch minute baad try karo.",
        ) from exc

    # record actual token usage
    if user:
        tokens = (
            result.usage.get("totalTokenCount", result.usage.get("total_tokens", 0)) or 0
        )
        await _record_tokens(user, int(tokens))

    return JSONResponse(
        content={
            "id": "chatcmpl-rotator",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result.model,
            "provider": result.provider,
            "key": result.key_label,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.text,
                        **({"tool_calls": result.tool_calls} if result.tool_calls else {}),
                    },
                    "finish_reason": "tool_calls" if result.tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.usage.get("promptTokenCount", result.usage.get("prompt_tokens", 0)),
                "completion_tokens": result.usage.get("candidatesTokenCount", result.usage.get("completion_tokens", 0)),
                "total_tokens": result.usage.get("totalTokenCount", result.usage.get("total_tokens", 0)),
            },
        }
    )


# --------------------------------------------------------------------------
# Auth + quota helpers
# --------------------------------------------------------------------------
async def _authenticate(request: Request, settings: dict):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:].strip()

    if is_user_api_key(token):
        return await database.get_user_by_api_key(token)
    payload = decode_jwt(token, settings["jwt_secret"])
    if payload:
        return await database.get_user_by_id(int(payload["sub"]))
    return None


async def _require_user(request: Request, settings: dict):
    user = await _authenticate(request, settings)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


async def _reserve_quota(user) -> bool:
    """Request reserve karo — quota bacha hai toh True. Admin ke liye unlimited."""
    if user.is_admin:
        return True
    return await database.reserve_quota(user.id, database.today_utc(), user.daily_limit)


async def _refund_quota(user) -> None:
    """Fail ho gaya toh reserved request wapas de do."""
    await database.refund_quota(user.id, database.today_utc())


async def _record_tokens(user, tokens: int) -> None:
    await database.record_tokens(user.id, database.today_utc(), int(tokens))


def _user_public(user) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "daily_limit": user.daily_limit,
        "created_at": user.created_at,
    }


# --------------------------------------------------------------------------
# Message conversion
# --------------------------------------------------------------------------
def _convert_message(m: dict) -> ChatMessage:
    role = m.get("role", "user")
    content = m.get("content")
    images: list[ImageInput] = []
    files: list[ImageInput] = []
    tool_calls = m.get("tool_calls") or []
    tool_call_id = m.get("tool_call_id") or ""
    name = m.get("name") or ""

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                text_parts.append(part.get("text", ""))
            elif ptype == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    try:
                        meta, b64 = url.split(",", 1)
                        mime = meta.split(";")[0].replace("data:", "") or "image/jpeg"
                        images.append(ImageInput(data_base64=b64, mime_type=mime))
                    except ValueError:
                        raise ValueError("malformed data URL in image_url")
                elif url:
                    images.append(ImageInput(url=url))
            elif ptype in ("file", "input_file"):
                # PDF / docx / pptx / xlsx ... — OpenAI `file` content part.
                f = part.get("file") or {}
                file_data = f.get("file_data") or part.get("file_data") or ""
                if isinstance(file_data, str) and file_data.startswith("data:"):
                    try:
                        meta, b64 = file_data.split(",", 1)
                        mime = meta.split(";")[0].replace("data:", "") or "application/octet-stream"
                        files.append(ImageInput(data_base64=b64, mime_type=mime))
                    except ValueError:
                        raise ValueError("malformed data URL in file part")
                elif isinstance(file_data, str) and file_data:
                    files.append(ImageInput(url=file_data))
        text = "\n".join(t for t in text_parts if t)
    else:
        text = ""

    return ChatMessage(
        role=role,
        content=text,
        images=images,
        files=files,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        name=name,
    )


# --------------------------------------------------------------------------
# Dashboard SPA
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartRotator — LLM Gateway</title>
<style>
  :root {
    --bg:#0f1117; --panel:#171a23; --panel2:#1e2230; --border:#2a2f42;
    --text:#e8eaf2; --muted:#8b90a5; --accent:#7c6cff; --accent2:#4fd1c5;
    --green:#38e07a; --red:#ff6b6b; --yellow:#ffd93d;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; min-height:100vh; }
  .wrap { max-width:1100px; margin:0 auto; padding:20px 16px; }
  header { display:flex; align-items:center; justify-content:space-between; padding:8px 0 18px; }
  header h1 { font-size:22px; } header h1 span { color:var(--accent); }
  .btn { background:var(--accent); color:#fff; border:none; border-radius:10px; padding:10px 18px; font-size:14px; cursor:pointer; }
  .btn.sec { background:var(--panel2); color:var(--text); border:1px solid var(--border); }
  .btn.danger { background:var(--red); }
  .btn:hover { filter:brightness(1.15); }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:16px; padding:20px; }
  input, textarea { background:var(--panel2); color:var(--text); border:1px solid var(--border); border-radius:10px; padding:12px; font-size:14px; width:100%; font-family:inherit; }
  input:focus, textarea:focus { outline:none; border-color:var(--accent); }
  label { font-size:13px; color:var(--muted); display:block; margin:12px 0 6px; }
  .tabs { display:flex; gap:4px; margin-bottom:14px; }
  .tab { padding:8px 18px; border-radius:10px 10px 0 0; background:var(--panel2); border:1px solid var(--border); border-bottom:none; cursor:pointer; font-size:14px; color:var(--muted); }
  .tab.active { color:var(--text); background:var(--panel); font-weight:600; }
  .view { display:none; } .view.active { display:block; }
  #messages { display:flex; flex-direction:column; gap:10px; max-height:48vh; overflow-y:auto; padding:4px 2px 12px; }
  .msg { max-width:85%; padding:10px 14px; border-radius:14px; font-size:14px; line-height:1.5; white-space:pre-wrap; word-break:break-word; }
  .msg.user { align-self:flex-end; background:var(--accent); color:#fff; border-bottom-right-radius:4px; }
  .msg.ai { align-self:flex-start; background:var(--panel2); border:1px solid var(--border); border-bottom-left-radius:4px; }
  .msg .meta { font-size:11px; color:var(--accent2); margin-bottom:6px; }
  .msg.user .meta { color:rgba(255,255,255,.7); }
  .row { display:flex; gap:10px; align-items:flex-end; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  .stat { display:flex; justify-content:space-between; padding:10px 0; border-bottom:1px solid var(--border); font-size:14px; }
  .stat:last-child { border-bottom:none; }
  .bar-bg { background:var(--panel2); border-radius:20px; height:12px; overflow:hidden; }
  .bar { background:linear-gradient(90deg,var(--accent),var(--accent2)); height:100%; border-radius:20px; transition:width .4s; }
  .history { display:flex; gap:8px; align-items:flex-end; height:120px; margin-top:14px; }
  .hcol { flex:1; display:flex; flex-direction:column; justify-content:flex-end; align-items:center; gap:4px; }
  .hbar { width:70%; background:linear-gradient(180deg,var(--accent),var(--accent2)); border-radius:6px 6px 0 0; min-height:2px; }
  .hday { font-size:10px; color:var(--muted); }
  .keybox { display:flex; gap:8px; align-items:center; }
  .keybox input { font-family:monospace; flex:1; }
  .model-list { display:flex; flex-wrap:wrap; gap:8px; }
  .chip { display:flex; align-items:center; gap:6px; background:var(--panel2); border:1px solid var(--border); border-radius:20px; padding:6px 12px; font-size:13px; cursor:pointer; }
  .chip input { accent-color:var(--accent); width:14px; height:14px; }
  .chip.checked { border-color:var(--accent); }
  .muted { color:var(--muted); font-size:13px; }
  .tag { font-size:11px; background:var(--panel2); border:1px solid var(--border); border-radius:20px; padding:2px 10px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--border); }
  th { color:var(--muted); font-weight:600; }
  .badge-ok { color:var(--green); } .badge-admin { color:var(--yellow); }
  .err { color:var(--red); font-size:13px; margin-top:8px; }
  .ok { color:var(--green); font-size:13px; margin-top:8px; }
  .modal { position:fixed; inset:0; z-index:100; background:rgba(0,0,0,.65); display:flex; align-items:center; justify-content:center; padding:16px; }
  .modal-box { background:var(--panel); border:1px solid var(--border); border-radius:16px; padding:18px; width:min(620px,100%); max-height:86vh; display:flex; flex-direction:column; }
  .pick-item { display:flex; align-items:center; gap:8px; padding:7px 10px; border-radius:8px; cursor:pointer; border:1px solid var(--border); margin-bottom:6px; background:var(--panel2); font-size:13px; }
  .pick-item.on { border-color:var(--accent); }
  .pick-item input { accent-color:var(--accent); width:15px; height:15px; }
  .pick-item .id { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  @media (max-width:760px){ .grid2{ grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>🔄 Smart<span>Rotator</span></h1>
    <div id="userChip" class="muted"></div>
  </header>

  <!-- AUTH VIEW -->
  <div id="view-auth" class="card" style="max-width:420px;margin:40px auto;">
    <div class="tabs" style="margin-bottom:18px;">
      <div class="tab active" id="tab-login" onclick="authTab('login')">Login</div>
      <div class="tab" id="tab-reg" onclick="authTab('reg')">Register</div>
    </div>
    <div id="auth-login">
      <label>Username</label><input id="login-user" autocomplete="username">
      <label>Password</label><input id="login-pass" type="password" autocomplete="current-password">
      <div class="err" id="login-err"></div>
      <div style="height:14px"></div>
      <button class="btn" style="width:100%" onclick="doLogin()">Login ➤</button>
    </div>
    <div id="auth-reg" style="display:none;">
      <label>Username (min 3)</label><input id="reg-user" autocomplete="username">
      <label>Password (min 6)</label><input id="reg-pass" type="password" autocomplete="new-password">
      <div class="err" id="reg-err"></div>
      <div style="height:14px"></div>
      <button class="btn" style="width:100%" onclick="doRegister()">Create Account ➤</button>
    </div>
    <div class="muted" style="margin-top:16px;text-align:center">
      Login karte hi per-user daily limit apply hoti hai.<br>
      Apps me bhi bas login ka JWT token — koi API key nahi chahiye. 😊
    </div>
  </div>

  <!-- MAIN VIEW -->
  <div id="view-main" style="display:none;">
    <div class="tabs">
      <div class="tab active" onclick="showView('chat')">💬 Chat</div>
      <div class="tab" onclick="showView('usage'); loadMe();">📊 Usage</div>
      <div class="tab" onclick="showView('settings')">⚙️ Settings</div>
      <div class="tab" id="modelsTab" style="display:none" onclick="showView('models'); loadModelsAdmin();">🧠 Models</div>
      <div class="tab" id="adminTab" style="display:none" onclick="showView('admin'); loadAdmin();">🛡 Admin</div>
    </div>

    <!-- CHAT -->
    <div class="view active" id="view-chat">
      <div class="card">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:10px">
          <div class="muted">Models (multi-select — inhi me rotation hogi):</div>
          <button class="btn sec" onclick="openModelPicker()">🎯 Pick Models <span id="pickCount" class="tag" style="color:var(--accent2)">0</span></button>
        </div>
        <div class="model-list" id="modelList"><span class="muted">Loading…</span></div>
        <div id="messages" style="margin-top:16px;"></div>
        <div style="margin-top:8px;display:flex;gap:8px">
          <input type="file" id="fileInput" accept="image/*" hidden onchange="handleFile(this)">
          <button class="btn sec" onclick="document.getElementById('fileInput').click()">🖼 Image</button>
          <div id="thumbs" style="display:flex;gap:6px"></div>
        </div>
        <div class="row" style="margin-top:10px">
          <textarea id="prompt" rows="2" placeholder="Message… (Enter=send, Shift+Enter=newline)"></textarea>
          <button class="btn" id="sendBtn" onclick="send()">Send ➤</button>
        </div>
      </div>
    </div>

    <!-- MODEL PICKER MODAL -->
    <div class="modal" id="modelPicker" style="display:none">
      <div class="modal-box">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
          <h3 style="margin:0">🎯 Pick Models</h3>
          <button class="btn sec" onclick="closeModelPicker()">✕</button>
        </div>
        <input id="pickSearch" class="field" style="width:100%;box-sizing:border-box;margin-bottom:10px" placeholder="🔍 Search models… (jaise gemini-3.5)" oninput="renderPickerList()">
        <div id="pickProviders" style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px"></div>
        <div id="pickList" style="max-height:46vh;overflow:auto;border:1px solid var(--border);border-radius:10px;padding:8px"></div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:12px;gap:8px;flex-wrap:wrap">
          <div class="muted" id="pickStatus"></div>
          <div style="display:flex;gap:8px">
            <button class="btn sec" onclick="pickAll()">All</button>
            <button class="btn sec" onclick="pickNone()">Clear</button>
            <button class="btn" onclick="applyPicks()">✅ Apply</button>
          </div>
        </div>
      </div>
    </div>

    <!-- USAGE -->
    <div class="view" id="view-usage">
      <div class="grid2">
        <div class="card">
          <h3 style="margin-bottom:14px">Today's Usage</h3>
          <div class="stat"><span>Requests used</span><span id="u-used">–</span></div>
          <div class="stat"><span>Daily limit</span><span id="u-limit">–</span></div>
          <div class="stat"><span>Tokens consumed</span><span id="u-tokens">–</span></div>
          <div class="bar-bg" style="margin-top:14px"><div class="bar" id="u-bar" style="width:0%"></div></div>
          <div class="muted" id="u-remaining" style="margin-top:8px"></div>
          <h3 style="margin:20px 0 10px">Last 7 Days</h3>
          <div class="history" id="u-history"></div>
        </div>
        <div class="card">
          <h3 style="margin-bottom:14px">Account</h3>
          <div class="stat"><span>Username</span><span id="u-name">–</span></div>
          <div class="stat"><span>Role</span><span id="u-role">–</span></div>
          <div class="stat"><span>Member since</span><span id="u-created">–</span></div>
        </div>
      </div>
    </div>

    <!-- SETTINGS -->
    <div class="view" id="view-settings">
      <div class="card">
        <h3 style="margin-bottom:14px">App Integration — login based (no API key)</h3>
        <p class="muted" style="margin-bottom:10px">App me login karo, JWT token lo, wahi use karo. Per-user limit apne aap apply hogi:</p>
        <pre id="code-sample" style="background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:12px;font-size:12px;overflow:auto;margin-bottom:14px"></pre>
        <hr style="border:none;border-top:1px solid var(--border);margin:20px 0">
        <h3 style="margin-bottom:10px">API Key (optional — OpenAI SDK ke liye)</h3>
        <p class="muted" style="margin-bottom:10px">Agar OpenAI SDK hi use karni ho toh sk- key bhi chalegi:
        <code style="background:var(--panel2);padding:2px 6px;border-radius:6px">OpenAI(base_url=location.origin+"/v1", api_key="sk-...")</code></p>
        <div class="keybox">
          <input id="api-key" readonly>
          <button class="btn sec" onclick="copyKey()">📋 Copy</button>
          <button class="btn sec" onclick="rotateKey()">🔄 Rotate</button>
        </div>
        <div class="err" id="settings-err"></div>
        <div class="ok" id="settings-ok"></div>
        <hr style="border:none;border-top:1px solid var(--border);margin:20px 0">
        <h3 style="margin-bottom:10px">🔑 Change Password</h3>
        <p class="muted" style="margin-bottom:10px">Apna password badlo — old password daalo + naya password. (Pehli baar set kar rahe ho toh old khali chhodo.)</p>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <input id="pw-old" type="password" placeholder="old password (optional)" style="flex:1;min-width:120px;padding:8px" autocomplete="current-password">
          <input id="pw-new" type="password" placeholder="new password (min 6)" style="flex:1;min-width:120px;padding:8px" autocomplete="new-password">
          <button class="btn sec" onclick="changePassword()">💾 Update</button>
        </div>
        <div style="height:10px"></div>
        <hr style="border:none;border-top:1px solid var(--border);margin:20px 0">
        <button class="btn danger" onclick="logout()">Logout</button>
      </div>
    </div>

    <!-- ADMIN -->
    <div class="view" id="view-admin">
      <div class="card">
        <h3 style="margin-bottom:14px">Users & Quotas</h3>
        <div style="overflow:auto"><table id="admin-table">
          <thead><tr><th>ID</th><th>Username</th><th>Role</th><th>Key</th><th>Today</th><th>Limit</th><th></th></tr></thead>
          <tbody></tbody>
        </table></div>
      </div>
    </div>

    <!-- MODELS ADMIN -->
    <div class="view" id="view-models">
      <div class="card">
        <h3 style="margin-bottom:4px">Virtual Model Groups</h3>
        <div class="muted" style="margin-bottom:10px">Group = ek model id jo multiple (provider, model) ko rotate karta hai. App bas group id bhejega (jaise "levelup"). Model dropdown me LIVE models dikhte hain — provider API se fetch hoke. 🔄</div>
        <div class="err" id="models-err"></div>
        <div class="ok" id="models-ok"></div>
        <div style="margin:10px 0">
          <button class="btn sec" onclick="refreshGroupLive()">🔄 Refresh Live Models</button>
          <span class="muted" id="live-refresh-status" style="margin-left:10px"></span>
        </div>
        <div id="group-editor"></div>
        <button class="btn sec" onclick="addGroup()" style="margin-top:10px">+ Add Group</button>
        <div style="height:18px"></div>
        <button class="btn" onclick="saveModels()">💾 Save Models Config</button>
      </div>
    </div>
  </div>
</div>

<script>
let token = localStorage.getItem('sr_token') || '';
let images = [];

// ---------- helpers ----------
async function api(path, opts = {}) {
  opts.headers = opts.headers || {};
  opts.headers['Content-Type'] = 'application/json';
  if (token) opts.headers['Authorization'] = 'Bearer ' + token;
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  return { res, data };
}

// ---------- auth ----------
function showAuth() {
  document.getElementById('view-auth').style.display = 'block';
  document.getElementById('view-main').style.display = 'none';
  document.getElementById('userChip').textContent = '';
}
function showMain(user, isSuperAdmin) {
  document.getElementById('view-auth').style.display = 'none';
  document.getElementById('view-main').style.display = 'block';
  document.getElementById('userChip').textContent = '👤 ' + user.username + (isSuperAdmin ? ' 👑 super admin' : (user.role === 'admin' ? ' ⭐ admin' : ''));
  // Tabs: user → bas Chat/Usage/Settings. Models → koi bhi admin
  // (env wala YA promote hua). Users & Quotas (Admin) → SIRF env wala.
  const isAnyAdmin = isSuperAdmin || user.role === 'admin';
  document.getElementById('modelsTab').style.display = isAnyAdmin ? '' : 'none';
  document.getElementById('adminTab').style.display = isSuperAdmin ? '' : 'none';
}
function authTab(which) {
  document.getElementById('tab-login').classList.toggle('active', which === 'login');
  document.getElementById('tab-reg').classList.toggle('active', which === 'reg');
  document.getElementById('auth-login').style.display = which === 'login' ? '' : 'none';
  document.getElementById('auth-reg').style.display = which === 'reg' ? '' : 'none';
}
async function doLogin() {
  const { res, data } = await api('/auth/login', { method: 'POST', body: JSON.stringify({ username: document.getElementById('login-user').value, password: document.getElementById('login-pass').value }) });
  if (!res.ok) return document.getElementById('login-err').textContent = data.detail || 'Login failed';
  afterAuth(data);
}
async function doRegister() {
  const { res, data } = await api('/auth/register', { method: 'POST', body: JSON.stringify({ username: document.getElementById('reg-user').value, password: document.getElementById('reg-pass').value }) });
  if (!res.ok) return document.getElementById('reg-err').textContent = data.detail || 'Register failed';
  afterAuth(data);
}
function afterAuth(data) {
  token = data.token; localStorage.setItem('sr_token', token);
  document.getElementById('api-key').value = data.api_key;
  showMain(data.user, data.is_super_admin); showView('chat'); loadModels(); loadMe();
}
function logout() { localStorage.removeItem('sr_token'); token = ''; location.reload(); }

// ---------- init ----------
(async function init() {
  if (token) {
    const { res, data } = await api('/auth/me');
    if (res.ok) {
      showMain(data.user, data.is_super_admin);
      document.getElementById('api-key').value = data.api_key || '';
      updateCodeSample();
      loadModels(); loadMe();
    } else showAuth();
  } else showAuth();
})();

// ---------- chat ----------
function persistPicks() {
  // selection localStorage me yaad rakh — refresh/login pe auto-reset na ho
  try { localStorage.setItem('sr_models', JSON.stringify(selectedModels)); } catch (e) {}
}
async function loadModels() {
  const { res, data } = await api('/v1/models/raw');
  if (!res.ok) return;
  allModels = data.data || [];
  const defaultModel = data.default_model || (allModels[0] && allModels[0].id) || '';
  // saved selection restore karo — nahi toh pehli baar saare models (full rotation)
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem('sr_models') || 'null'); } catch (e) { saved = null; }
  if (Array.isArray(saved) && saved.length) {
    selectedModels = saved.filter(id => allModels.some(m => m.id === id));
    if (!selectedModels.length && defaultModel) selectedModels = [defaultModel];
  } else if (defaultModel) {
    selectedModels = [defaultModel];
  } else {
    selectedModels = allModels.map(m => m.id);
  }
  persistPicks();
  renderModelChips();
  renderPickerProviders();
}
let allModels = [];
let selectedModels = [];

function renderModelChips() {
  const list = document.getElementById('modelList');
  list.innerHTML = '';
  if (selectedModels.length === 0) {
    list.innerHTML = '<span class="muted">Koi model select nahi — 🎯 Pick Models dabao</span>';
  }
  selectedModels.forEach(id => {
    const m = allModels.find(x => x.id === id);
    const chip = document.createElement('label');
    chip.className = 'chip checked';
    chip.innerHTML = `${id}<span class="tag">${m ? m.owned_by : '?'}</span><span style="cursor:pointer" onclick="removePick('${id.replace(/'/g, "\\'")}')" title="hatao">✕</span>`;
    list.appendChild(chip);
  });
  const c = document.getElementById('pickCount');
  if (c) c.textContent = selectedModels.length;
}
async function send() {
  const text = document.getElementById('prompt').value.trim();
  if (!text && images.length === 0) return;
  // pehle hi box clear karo — request slow ho toh bhi message wahi nahi rehta
  document.getElementById('prompt').value = '';
  const sentImages = images.slice();
  addMsg('user', text || '(image only)', sentImages);
  const content = [];
  sentImages.forEach(i => content.push({ type: 'image_url', image_url: { url: i.dataUrl } }));
  if (text) content.push({ type: 'text', text });
  document.getElementById('sendBtn').disabled = true;
  try {
    const { res, data } = await api('/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify({ models: selectedModels, messages: [{ role: 'user', content }] })
    });
    if (!res.ok) addMsg('ai', '⚠️ ' + (data.detail || ('Error ' + res.status)));
    else addMsg('ai', data.choices[0].message.content, [], `⚡ ${data.provider} · ${data.model} · ${data.key}`);
  } catch (err) {
    addMsg('ai', '⚠️ Network error: ' + (err.message || err));
  } finally {
    document.getElementById('sendBtn').disabled = false;
    images = []; document.getElementById('thumbs').innerHTML = '';
  }
}

// ---------- model picker (modal) ----------
let pickFilter = 'all';
function openModelPicker() {
  renderPickerProviders();
  renderPickerList();
  document.getElementById('modelPicker').style.display = 'flex';
}
function closeModelPicker() { document.getElementById('modelPicker').style.display = 'none'; }
function renderPickerProviders() {
  const provs = Array.from(new Set(allModels.map(m => m.owned_by))).sort();
  const box = document.getElementById('pickProviders');
  box.innerHTML = `<button class="btn sec" style="padding:4px 10px;font-size:12px" onclick="setPickFilter('all')">All</button>`;
  provs.forEach(p => {
    box.innerHTML += `<button class="btn sec" style="padding:4px 10px;font-size:12px;${pickFilter === p ? 'border-color:var(--accent);color:var(--accent2)' : ''}" onclick="setPickFilter('${p.replace(/'/g, "\\'")}')">${p}</button>`;
  });
}
function setPickFilter(p) { pickFilter = p; renderPickerProviders(); renderPickerList(); }
function renderPickerList() {
  const q = (document.getElementById('pickSearch').value || '').toLowerCase().trim();
  const list = document.getElementById('pickList');
  list.innerHTML = '';
  allModels.filter(m => (pickFilter === 'all' || m.owned_by === pickFilter) && (!q || m.id.toLowerCase().includes(q)))
    .forEach(m => {
      const on = selectedModels.includes(m.id);
      const div = document.createElement('div');
      div.className = 'pick-item' + (on ? ' on' : '');
      div.innerHTML = `<input type="checkbox" ${on ? 'checked' : ''} onchange="togglePick('${m.id.replace(/'/g, "\\'")}', this.checked)"><span class="id">${m.id}</span><span class="tag">${m.owned_by}</span>`;
      list.appendChild(div);
    });
  const st = document.getElementById('pickStatus');
  st.textContent = selectedModels.length + ' selected';
}
function togglePick(id, on) {
  const i = selectedModels.indexOf(id);
  if (on && i < 0) selectedModels.push(id);
  if (!on && i >= 0) selectedModels.splice(i, 1);
  persistPicks();
  renderPickerList();
  renderModelChips();
}
function pickAll() { selectedModels = allModels.map(m => m.id); persistPicks(); renderPickerList(); renderModelChips(); }
function pickNone() { selectedModels = []; persistPicks(); renderPickerList(); renderModelChips(); }
function applyPicks() { persistPicks(); closeModelPicker(); renderModelChips(); }
function removePick(id) {
  const i = selectedModels.indexOf(id);
  if (i >= 0) selectedModels.splice(i, 1);
  persistPicks();
  renderModelChips(); renderPickerList();
}
function addMsg(role, text, imgs = [], meta = null) {
  const box = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (meta) div.innerHTML = `<div class="meta">${meta}</div>`;
  imgs.forEach(i => { const img = document.createElement('img'); img.src = i.dataUrl; img.style.cssText = 'max-width:180px;border-radius:8px;display:block;margin-bottom:6px'; div.appendChild(img); });
  if (text) div.appendChild(document.createTextNode(text));
  box.appendChild(div); box.scrollTop = box.scrollHeight;
}
function handleFile(input) {
  for (const file of input.files) {
    const r = new FileReader();
    r.onload = e => {
      const idx = images.length; images.push({ dataUrl: e.target.result });
      const t = document.createElement('div'); t.style.position = 'relative';
      t.innerHTML = `<img src="${e.target.result}" style="width:48px;height:48px;object-fit:cover;border-radius:8px;border:1px solid var(--border)">
                     <button onclick="removeImg(${idx})" style="position:absolute;top:-6px;right:-6px;width:18px;height:18px;border-radius:50%;background:var(--red);color:#fff;border:none;font-size:10px;cursor:pointer">✕</button>`;
      document.getElementById('thumbs').appendChild(t);
    };
    r.readAsDataURL(file);
  }
  input.value = '';
}
function removeImg(idx) {
  images.splice(idx, 1);
  const box = document.getElementById('thumbs'); box.innerHTML = '';
  images.forEach((img, i) => { const t = document.createElement('div'); t.innerHTML = `<img src="${img.dataUrl}" style="width:48px;height:48px;object-fit:cover;border-radius:8px">`; box.appendChild(t); });
}
document.getElementById('prompt').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });

// ---------- usage ----------
async function loadMe() {
  const { res, data } = await api('/auth/me');
  if (!res.ok) return;
  const u = data.user, t = data.today;
  document.getElementById('u-used').textContent = t.requests;
  document.getElementById('u-limit').textContent = u.daily_limit;
  document.getElementById('u-tokens').textContent = t.tokens.toLocaleString();
  const pct = Math.min(100, (t.requests / u.daily_limit) * 100);
  document.getElementById('u-bar').style.width = pct + '%';
  document.getElementById('u-remaining').textContent = (u.daily_limit - t.requests) + ' requests remaining today';
  document.getElementById('u-name').textContent = u.username;
  document.getElementById('u-role').textContent = u.role;
  document.getElementById('u-created').textContent = (u.created_at || '').slice(0, 10);
  const hist = document.getElementById('u-history');
  hist.innerHTML = '';
  const max = Math.max(1, ...data.history.map(h => h.requests));
  data.history.forEach(h => {
    const col = document.createElement('div'); col.className = 'hcol';
    const bar = document.createElement('div'); bar.className = 'hbar';
    bar.style.height = Math.max(2, (h.requests / max) * 100) + 'px';
    const day = document.createElement('div'); day.className = 'hday'; day.textContent = h.day.slice(5);
    col.appendChild(bar); col.appendChild(day);
    hist.appendChild(col);
  });
}

// ---------- settings ----------
function copyKey() {
  const k = document.getElementById('api-key').value;
  if (k) { navigator.clipboard.writeText(k); setSettingsMsg('✅ Copied!', true); }
}
async function rotateKey() {
  const { res, data } = await api('/auth/rotate-key', { method: 'POST' });
  if (res.ok) { document.getElementById('api-key').value = data.api_key; setSettingsMsg('✅ New key generated!', true); }
  else setSettingsMsg('❌ ' + (data.detail || 'Failed'), false);
}
async function changePassword() {
  const oldpw = document.getElementById('pw-old').value;
  const newpw = document.getElementById('pw-new').value;
  if (newpw.length < 6) return setSettingsMsg('❌ Naya password kam se kam 6 chars ka ho', false);
  const { res, data } = await api('/auth/change-password', { method: 'POST', body: JSON.stringify({ old_password: oldpw, new_password: newpw }) });
  if (res.ok) { document.getElementById('pw-old').value = ''; document.getElementById('pw-new').value = ''; setSettingsMsg('✅ ' + (data.message || 'Password update!'), true); }
  else setSettingsMsg('❌ ' + (data.detail || 'Failed'), false);
}
function setSettingsMsg(text, ok) {
  const el = document.getElementById(ok ? 'settings-ok' : 'settings-err');
  el.textContent = text; setTimeout(() => el.textContent = '', 3000);
}
function updateCodeSample() {
  const base = location.origin;
  document.getElementById('code-sample').textContent =
`# STEP 1 — login karo, JWT token lo (koi API key nahi chahiye)
import requests

r = requests.post("${base}/auth/login",
                  json={"username": "YOUR_USER", "password": "YOUR_PASS"})
token = r.json()["token"]

# STEP 2 — token se chat karo (per-user quota apne aap apply hogi)
resp = requests.post("${base}/v1/chat/completions",
    headers={"Authorization": "Bearer " + token},
    json={"messages": [{"role": "user", "content": "Hello!"}]})
print(resp.json()["choices"][0]["message"]["content"])`;
}
document.addEventListener('input', updateCodeSample);
setInterval(() => { const k = document.getElementById('api-key').value; if (k && location.origin) updateCodeSample(); }, 500);

// ---------- admin ----------
async function loadAdmin() {
  const { res, data } = await api('/admin/users');
  if (!res.ok) return;
  const tb = document.querySelector('#admin-table tbody');
  tb.innerHTML = '';
  data.users.forEach(u => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${u.id}</td><td>${u.username}</td>
      <td><select class="role-sel" data-id="${u.id}" onchange="setRole(${u.id}, this.value)" style="background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:4px 6px;color:var(--text)">
        <option value="user" ${u.role === 'user' ? 'selected' : ''}>user</option>
        <option value="admin" ${u.role === 'admin' ? 'selected' : ''}>admin</option>
      </select></td>
      <td class="muted">${u.api_key}</td>
      <td>${u.today_requests}</td>
      <td>${u.daily_limit}</td>
      <td><input type="number" value="${u.daily_limit}" min="1" style="width:80px;padding:6px" onchange="setLimit(${u.id}, this.value)"></td>`;
    tb.appendChild(tr);
  });
}
async function setRole(id, role) {
  const { res, data } = await api('/admin/users/' + id + '/role', { method: 'POST', body: JSON.stringify({ role }) });
  if (!res.ok) { alert(data.detail || 'Failed'); loadAdmin(); }
  else setSettingsMsg('✅ ' + data.username + ' → ' + data.role, true);
}
async function setLimit(id, value) {
  const { res, data } = await api('/admin/users/' + id + '/limit', { method: 'POST', body: JSON.stringify({ daily_limit: parseInt(value) }) });
  if (!res.ok) alert(data.detail || 'Failed');
}

// ---------- models admin ----------
let modelsData = null;  // {catalog, managed, providers, default_model}
let modelsDraft = null; // working copy

async function loadModelsAdmin() {
  const { res, data } = await api('/admin/models');
  if (!res.ok) { document.getElementById('models-err').textContent = data.detail || 'Failed'; return; }
  modelsData = data;
  modelsDraft = JSON.parse(JSON.stringify(data.managed));  // deep copy
  if (!modelsDraft.provider_models) modelsDraft.provider_models = {};
  if (!modelsDraft.provider_order) modelsDraft.provider_order = [];
  if (!modelsDraft.groups) modelsDraft.groups = [];
  renderGroupEditor();
}

// ---------- live models (group editor) ----------
async function refreshGroupLive() {
  const st = document.getElementById('live-refresh-status');
  const btn = document.querySelector('button[onclick="refreshGroupLive()"]');
  if (btn) { btn.disabled = true; btn.textContent = '⏳ refreshing…'; }
  if (st) st.textContent = '';
  const { res, data } = await api('/admin/providers/refresh-all', { method: 'POST' });
  if (btn) { btn.disabled = false; btn.textContent = '🔄 Refresh Live Models'; }
  if (!res.ok) { if (st) st.textContent = '❌ ' + (data.detail || 'Failed'); return; }
  const lines = (data.results || []).map(r => `${r.name}: ${r.count}${r.error ? ' (❌ ' + r.error + ')' : ''}`).join(' · ');
  if (st) st.textContent = '✅ ' + lines;
  await loadModelsAdminData();
}

// provider ke saare possible models: LIVE (API se) + catalog + configured
function providerModelOptions(providerName) {
  const pInfo = modelsData.providers.find(p => p.name === providerName);
  const liveModels = (pInfo && pInfo.live_models) || [];
  const catalogModels = (modelsData.catalog && modelsData.catalog[providerName]) || [];
  const configured = (pInfo && pInfo.configured_models) || [];
  return Array.from(new Set(liveModels.concat(catalogModels).concat(configured)));
}

async function loadModelsAdminData() {
  const { res, data } = await api('/admin/models');
  if (!res.ok) return;
  modelsData = data;
  modelsDraft = JSON.parse(JSON.stringify(data.managed));
  if (!modelsDraft.provider_models) modelsDraft.provider_models = {};
  if (!modelsDraft.provider_order) modelsDraft.provider_order = [];
  if (!modelsDraft.groups) modelsDraft.groups = [];
  renderGroupEditor();
}

function renderGroupEditor() {
  const box = document.getElementById('group-editor');
  box.innerHTML = '';
  modelsDraft.groups.forEach((g, gi) => {
    const card = document.createElement('div');
    card.style.cssText = 'background:var(--panel2);border:1px solid var(--border);border-radius:12px;padding:14px;margin-bottom:10px';
    let membersHtml = '';
    g.members.forEach((m, mi) => {
      // backward compat: purana single-model member → models array
      if (!m.models) m.models = m.model ? [m.model] : [];
      if (!m.keys) m.keys = [];
      const provOpts = modelsData.providers.map(p => `<option value="${p.name}" ${p.name === m.provider ? 'selected' : ''}>${p.name}</option>`).join('');
      const availableModels = providerModelOptions(m.provider);
      // models queue chips (order = rotation order)
      const modelChips = m.models.map((mm, mii) =>
        `<span class="chip checked">${mm} <span style="cursor:pointer" onclick="removeMemberModel(${gi}, ${mi}, ${mii})" title="hatao">✕</span></span>`).join('');
      // add-model dropdown — unselected models
      const unselected = availableModels.filter(mm => !m.models.includes(mm));
      const addOpts = unselected.map(mm => `<option value="${mm}">${mm}</option>`).join('');
      const addSel = `<select id="addm-${gi}-${mi}" onchange="addMemberModel(${gi}, ${mi}, this.value); this.value=''" style="flex:2;background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:8px;color:var(--text)"><option value="">+ model add</option>${addOpts}</select>`;
      // keys selection — masked preview + checkbox
      const pInfo = modelsData.providers.find(p => p.name === m.provider);
      const keyOpts = (pInfo && pInfo.keys || []).map(k => {
        const on = m.keys.includes(k.index);
        return `<label class="chip ${on ? 'checked' : ''}" style="cursor:pointer" title="${k.index}"><input type="checkbox" ${on ? 'checked' : ''} onchange="toggleMemberKey(${gi}, ${mi}, ${k.index})">${k.preview}</label>`;
      }).join('');
      const keyNote = m.keys.length
        ? `<span class="muted" style="font-size:11px">✅ ${m.keys.length} key select</span>`
        : '<span class="muted" style="font-size:11px">(kuch nahi select = saari keys)</span>';
      const keyAllBtn = (pInfo && pInfo.keys && pInfo.keys.length)
        ? `<button class="btn sec" style="padding:2px 8px;font-size:11px" onclick="selectAllKeys(${gi}, ${mi})">Select All</button>
           <button class="btn sec" style="padding:2px 8px;font-size:11px" onclick="clearKeys(${gi}, ${mi})">Clear</button>`
        : '';
      membersHtml += `<div style="border:1px solid var(--border);border-radius:10px;padding:10px;margin-top:8px;background:var(--panel)">
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
          <select onchange="updateMemberProvider(${gi}, ${mi}, this.value)" style="flex:1;min-width:120px;background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:8px;color:var(--text)">${provOpts}</select>
          <button class="btn danger" style="padding:4px 10px" onclick="removeMember(${gi}, ${mi})">✕</button>
        </div>
        <div class="muted" style="font-size:11px;margin-bottom:4px">Models queue (upar wala pehle try hota hai):</div>
        <div class="model-list">${modelChips || '<span class="muted" style="font-size:11px">koi model nahi — neeche se add karo</span>'}</div>
        <div style="display:flex;gap:8px;margin-top:6px">${addSel}<button class="btn sec" style="padding:8px 12px" onclick="addMemberModel(${gi}, ${mi}, document.getElementById('addm-${gi}-${mi}').value); document.getElementById('addm-${gi}-${mi}').value=''">+</button></div>
        <div style="display:flex;align-items:center;gap:8px;margin:8px 0 4px">
          <div class="muted" style="font-size:11px">API keys (${pInfo ? pInfo.key_count : 0} available):</div>
          ${keyAllBtn}
        </div>
        <div class="model-list">${keyOpts || '<span class="muted" style="font-size:11px">koi keys nahi</span>'}</div>
        ${keyNote}
      </div>`;
    });
    card.innerHTML = `
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <input value="${g.id}" placeholder="group id (jaise levelup)" style="flex:1;min-width:140px;padding:8px" onchange="updateGroup(${gi}, 'id', this.value)">
        <input value="${g.label || ''}" placeholder="label" style="flex:1;min-width:100px;padding:8px" onchange="updateGroup(${gi}, 'label', this.value)">
        <label style="display:flex;align-items:center;gap:6px;font-size:13px"><input type="checkbox" ${g.enabled ? 'checked' : ''} onchange="updateGroup(${gi}, 'enabled', this.checked)">enabled</label>
        <button class="btn danger" style="padding:6px 12px" onclick="removeGroup(${gi})">🗑 Delete</button>
      </div>
      <div style="margin-top:8px">${membersHtml}</div>
      <button class="btn sec" style="padding:6px 12px;margin-top:10px" onclick="addMember(${gi})">+ Member</button>
      <div class="muted" style="margin-top:6px;font-size:12px">App bhejega: model = "<b>${g.id || 'levelup'}</b>" → server pehle member 1 ke saare models+keys, phir member 2, ... rotate karega</div>`;
    box.appendChild(card);
  });
}

function updateGroup(gi, key, val) { modelsDraft.groups[gi][key] = val; }
function removeGroup(gi) { modelsDraft.groups.splice(gi, 1); renderGroupEditor(); }
function addGroup() {
  modelsDraft.groups.push({ id: 'levelup', label: 'LevelUp', enabled: true, members: [] });
  renderGroupEditor();
}
function removeMember(gi, mi) { modelsDraft.groups[gi].members.splice(mi, 1); renderGroupEditor(); }
function addMember(gi) {
  const prov = modelsData.providers[0] || { name: '' };
  const opts = providerModelOptions(prov.name);
  const pInfo = modelsData.providers.find(p => p.name === prov.name);
  // default: saare models + saari keys select — taaki rotation me fallback mile
  const keyIdx = (pInfo && pInfo.keys || []).map(k => k.index);
  modelsDraft.groups[gi].members.push({ provider: prov.name, models: opts.slice(), keys: keyIdx });
  renderGroupEditor();
}
function selectAllKeys(gi, mi) {
  const mem = modelsDraft.groups[gi].members[mi];
  const pInfo = modelsData.providers.find(p => p.name === mem.provider);
  mem.keys = (pInfo && pInfo.keys || []).map(k => k.index);
  renderGroupEditor();
}
function clearKeys(gi, mi) {
  const mem = modelsDraft.groups[gi].members[mi];
  mem.keys = [];
  renderGroupEditor();
}
function addMemberModel(gi, mi, model) {
  const m = (model || '').trim();
  if (!m) return;
  const mem = modelsDraft.groups[gi].members[mi];
  if (!mem.models) mem.models = [];
  if (!mem.models.includes(m)) mem.models.push(m);
  renderGroupEditor();
}
function removeMemberModel(gi, mi, mii) {
  modelsDraft.groups[gi].members[mi].models.splice(mii, 1);
  renderGroupEditor();
}
function toggleMemberKey(gi, mi, ki) {
  const mem = modelsDraft.groups[gi].members[mi];
  if (!mem.keys) mem.keys = [];
  const i = mem.keys.indexOf(ki);
  if (i >= 0) mem.keys.splice(i, 1); else mem.keys.push(ki);
  mem.keys.sort((a, b) => a - b);
  renderGroupEditor();
}
function updateMemberProvider(gi, mi, val) {
  const mem = modelsDraft.groups[gi].members[mi];
  mem.provider = val;
  mem.models = [];   // provider badla → models queue reset
  mem.keys = [];     // keys bhi reset (doosre provider ki keys hain)
  renderGroupEditor();
}

async function saveModels() {
  // enabled groups ke members se provider_models + provider_order auto-banao
  const provider_models = {};
  const provider_order = [];
  modelsDraft.groups.forEach(g => {
    if (!g.enabled) return;
    g.members.forEach(m => {
      if (!m.provider || !m.models || !m.models.length) return;
      if (!provider_models[m.provider]) provider_models[m.provider] = [];
      m.models.forEach(mm => {
        if (!provider_models[m.provider].includes(mm)) provider_models[m.provider].push(mm);
      });
      if (!provider_order.includes(m.provider)) provider_order.push(m.provider);
    });
  });
  const { res, data } = await api('/admin/models', {
    method: 'PUT',
    body: JSON.stringify({
      provider_models,
      provider_order,
      groups: modelsDraft.groups,
    })
  });
  const errEl = document.getElementById('models-err'), okEl = document.getElementById('models-ok');
  errEl.textContent = ''; okEl.textContent = '';
  if (!res.ok) errEl.textContent = data.detail || 'Save failed';
  else {
    okEl.textContent = '✅ ' + (data.message || 'Saved');
    setTimeout(() => okEl.textContent = '', 3000);
  }
}

// ---------- tabs ----------
function showView(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById('view-' + name).classList.add('active');
  const tabs = document.querySelectorAll('.tab');
  tabs.forEach(t => { if (t.textContent.toLowerCase().includes(name)) t.classList.add('active'); });
}
</script>
</body>
</html>
"""


def main() -> None:
    import uvicorn

    server_cfg = {}
    try:
        raw = _load_config()
        server_cfg = raw.get("server", {})
    except Exception:
        pass

    host = os.environ.get("HOST", server_cfg.get("host", "0.0.0.0"))
    port = int(os.environ.get("PORT", server_cfg.get("port", 8000)))
    uvicorn.run("rotator.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
