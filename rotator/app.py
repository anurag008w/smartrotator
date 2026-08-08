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
  GET  /status                (key/proxy health — admin only)
  GET  /                      (dashboard SPA)

Auth:
  POST /auth/register  POST /auth/login  GET /auth/me
  POST /auth/rotate-key
Admin:
  GET  /admin/users    POST /admin/users/{id}/limit
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
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
    GEMINI_V1,
    ImageInput,
    ProviderError,
    RateLimitError,
    fetch_live_models,
    is_blank_text,
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
# Request logging middleware — production observability
#
# Har request log hoti hai: method, path, status, duration, client IP.
# Sensitive headers (Authorization) kabhi log NAHI hote. Paths me bhi koi
# query string nahi (tokens URL me aa sakte hain).
# --------------------------------------------------------------------------
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "%s %s -> %d (%.1fms) from %s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request.client.host if request.client else "-",
        )
    return response


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


def _load_or_create_jwt_secret() -> str:
    """JWT secret: env `JWT_SECRET` → hidden file `data/.jwt-secret` (auto-gen).

    Pehle ek PUBLIC default secret source me hardcoded tha — koi bhi JWT forge
    kar ke kisi bhi user ka token bana sakta tha (CRITICAL). Ab koi fallback
    secret source me nahi hai: env nahi hai toh random secret generate hota hai
    aur `data/.jwt-secret` (git-ignored) me persist hota hai.
    """
    env_secret = os.environ.get("JWT_SECRET", "").strip()
    if env_secret:
        return env_secret
    path = Path(github_sync.DATA_DIR) / ".jwt-secret"
    try:
        if path.exists():
            val = path.read_text(encoding="utf-8").strip()
            if val:
                return val
        path.parent.mkdir(parents=True, exist_ok=True)
        import secrets

        generated = secrets.token_urlsafe(48)
        path.write_text(generated + "\n", encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        logger.warning(
            "JWT_SECRET env nahi mila — naya secret data/.jwt-secret me generate "
            "hokar persist ho gaya (restart pe tokens invalid ho jayenge, re-login karna padega)"
        )
        return generated
    except OSError:
        # data/ writable nahi (read-only filesystem) — in-memory random secret.
        # Restart pe sab users logout ho jayenge; secure toh hai.
        logger.error(
            "JWT secret file write nahi ho paya (data/ writable nahi?) — "
            "in-memory random secret use ho raha hai, restart pe sab logout"
        )
        import secrets

        return secrets.token_urlsafe(48)


def _auth_settings() -> dict:
    cfg = _load_config().get("auth", {}) or {}
    secret = os.environ.get(cfg.get("jwt_secret_env", "JWT_SECRET"), cfg.get("jwt_secret", ""))
    if not secret:
        secret = _load_or_create_jwt_secret()
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "default_daily_limit": int(cfg.get("default_daily_limit", 30)),
        "default_monthly_limit": int(cfg.get("default_monthly_limit", 1000)),
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
# Rate limiting — in-memory sliding window (brute-force protection).
# Login/register pe unlimited requests = password brute-force + username
# enumeration. Ye limiter per-boot hai (restart pe reset) — kaafi for
# self-hosted scale, koi external dependency nahi chahiye.
# --------------------------------------------------------------------------
class _RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = {}  # key -> [monotonic timestamps]
        self._lock = asyncio.Lock()

    async def hit(self, key: str, limit: int, window: float) -> tuple[bool, int]:
        """Record a hit. Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        async with self._lock:
            bucket = self._hits.setdefault(key, [])
            cutoff = now - window
            bucket[:] = [t for t in bucket if t > cutoff]
            if len(bucket) >= limit:
                oldest = bucket[0] if bucket else now
                retry_after = max(1, int(window - (now - oldest)) + 1)
                return False, retry_after
            bucket.append(now)
            return True, 0

    def _reset(self) -> None:  # tests ke liye
        self._hits.clear()


_rate_limiter = _RateLimiter()


def _client_ip(request: Request) -> str:
    """Real client IP — Render pe nginx x-forwarded-for set karta hai."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _enforce_rate_limit(request: Request, bucket: str, limit: int, window: float) -> None:
    key = f"{bucket}:{_client_ip(request)}"
    allowed, retry_after = await _rate_limiter.hit(key, limit, window)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Bahut saare requests aa rahe hain — thodi der baad try karo.",
            headers={"Retry-After": str(retry_after)},
        )


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------
class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[dict] = Field(..., min_length=1, max_length=512)
    max_tokens: int = Field(8192, ge=1, le=262144)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = False  # accepted for compatibility; returns non-streamed
    models: Optional[list[str]] = Field(None, max_length=50)  # UI multi-select
    tools: Optional[list[dict]] = Field(None, max_length=50)  # function calling
    tool_choice: Optional[Union[str, dict]] = None
    # --- models ki real power: poora OpenAI-compatible surface pass-through ---
    top_p: Optional[float] = None
    stop: Optional[Union[str, list[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    response_format: Optional[dict] = None  # {"type": "json_object"} JSON mode
    seed: Optional[int] = None
    logit_bias: Optional[dict] = None


class EmbeddingsRequest(BaseModel):
    model: str = "gemini-embedding-001"
    input: Union[str, list[str]] = Field(..., min_length=1)
    encoding_format: str = "float"  # float | base64 (base64 abhi support nahi)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    # Login pe password policy (min_length) enforce NAHI karte — galat/chhota
    # password bhi 401 dena chahiye, 422 validation error nahi (policy leak).
    # Sirf size bound rakhna hai taaki unbounded input na aaye.
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class SyncStateRequest(BaseModel):
    """App-side push-authoritative: poora user state server ko bheja jata hai."""
    state: dict = {}
    updated_at: str = ""


class SetLimitRequest(BaseModel):
    daily_limit: Optional[int] = Field(None, ge=1, le=1000000)
    monthly_limit: Optional[int] = Field(None, ge=1, le=10000000)


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
    base_url: str = Field("", max_length=512)
    api_keys: list[str] = Field([], max_length=100)
    models: list[str] = Field([], max_length=100)
    enabled: bool = True
    replace_keys: bool = False    # true = purani keys hatake nayi lagao
    key_base_urls: dict[str, str] = {}   # key -> us ka apna base_url (per-key)
    selected_keys: list[int] = []        # existing keys me se select ki hui (indices)


# --------------------------------------------------------------------------
# Public endpoints
# --------------------------------------------------------------------------
@app.get("/health")
async def health(request: Request):
    """Depth check — sirf process alive nahi, data layer bhi check hoti hai.

    Render ka healthCheckPath isi pe hit hota hai. Agar users.json corrupt
    hai ya rotator load nahi hua, degraded status + 503 dete hain taaki
    Render restart trigger kare.
    """
    checks: list[dict] = []
    ok = True
    try:
        if not hasattr(request.app.state, "rotator") or request.app.state.rotator is None:
            raise RuntimeError("rotator initialized nahi hai")
        checks.append({"name": "rotator", "ok": True})
    except Exception as exc:  # noqa: BLE001
        ok = False
        checks.append({"name": "rotator", "ok": False, "detail": str(exc)})
    try:
        users = await database.count_users()
        if users < 0:
            raise RuntimeError("negative user count")
        checks.append({"name": "store", "ok": True, "users": users})
    except Exception as exc:  # noqa: BLE001
        ok = False
        checks.append({"name": "store", "ok": False, "detail": str(exc)})
    try:
        _load_config()
        checks.append({"name": "config", "ok": True})
    except Exception as exc:  # noqa: BLE001
        ok = False
        checks.append({"name": "config", "ok": False, "detail": str(exc)})
    if not ok:
        return JSONResponse(status_code=503, content={"status": "degraded", "checks": checks})
    return {"status": "ok", "checks": checks}


@app.get("/v1/models")
async def list_models(request: Request):
    rotator: Rotator = request.app.state.rotator
    # SIRF selected/configured models — live merge NAHI hota ab.
    # Dashboard "Exposed Models" tab se admin select karta hai ki /v1/models
    # me kaun se models dikhen (provider_models → apply_managed se cfg.models
    # override ho jata hai). External apps ke liye exactly wahi dikhega.
    data = [
        {"id": m["id"], "object": "model", "owned_by": m["provider"], "type": m["type"]}
        for m in rotator.models()
    ]
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
    """Provider/key health. Admin-only — key previews + cooldowns expose hoti
    hain, public nahi hone chahiye (auth off ho to public rahega)."""
    settings = _auth_settings()
    if settings["enabled"]:
        await _require_any_admin(request)
    rotator: Rotator = request.app.state.rotator
    return {**rotator.status(), "github_sync": github_sync.sync_status()}


# --------------------------------------------------------------------------
# Auth endpoints
# --------------------------------------------------------------------------
@app.post("/auth/register")
async def register(req: RegisterRequest, request: Request):
    settings = _auth_settings()
    if not settings["enabled"]:
        raise HTTPException(status_code=403, detail="Auth disabled")
    # register pe strict rate limit — unlimited public registration = attacker
    # pehle admin-bootstrap race jeet sakta hai + username enumeration.
    # 10/hour/IP — self-hosted scale ke liye kaafi, abuse rokne ke liye bhi.
    await _enforce_rate_limit(request, "register", 10, 3600)

    existing = await database.get_user_by_username(req.username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")

    # role: env ADMIN_USERS me naam ho → admin. Env set hai toh koi bhi
    # random pehla user auto-admin NAHI banega (sirf owner). Env khali
    # (local dev / self-host) pe bootstrap_admin config (default true) hone
    # pe pehla registered user admin hota hai. `bootstrap_admin: false`
    # production ke liye — admin sirf ADMIN_USERS se.
    bootstrap = bool(_load_config().get("auth", {}).get("bootstrap_admin", True))
    if req.username in settings["admin_usernames"]:
        role = "admin"
    elif settings["admin_usernames"]:
        role = "user"
    elif bootstrap:
        count = await database.count_users()
        role = "admin" if count == 0 else "user"
    else:
        role = "user"

    password_hash, salt = await asyncio.to_thread(hash_password, req.password)
    user = await database.create_user(
        username=req.username,
        password_hash=password_hash,
        salt=salt,
        api_key=generate_api_key(),
        daily_limit=settings["default_daily_limit"],
        monthly_limit=settings["default_monthly_limit"],
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
async def login(req: LoginRequest, request: Request):
    settings = _auth_settings()
    if not settings["enabled"]:
        raise HTTPException(status_code=403, detail="Auth disabled")
    # login brute-force protection — IP pe sliding window
    await _enforce_rate_limit(request, "login", 10, 60)

    user = await database.get_user_by_username(req.username)
    if not user or not await asyncio.to_thread(
        verify_password, req.password, user.password_hash, user.salt
    ):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Purane PBKDF2 hash ko scrypt pe migrate karo (login pe ek baar re-hash)
    if not is_scrypt_hash(user.password_hash):
        new_hash, _ = await asyncio.to_thread(hash_password, req.password, user.salt)
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
    month = database.month_utc()
    row = await database.get_usage_row(user.id, today)
    month_row = await database.get_usage_month(user.id, month)
    month_days = await database.get_usage_month_days(user.id, month)
    monthly_totals = await database.get_monthly_totals(user.id, 6)
    history = await database.get_usage_between(user.id, 7)
    return {
        "user": _user_public(user),
        "is_super_admin": _is_super_admin(user, settings),
        "api_key": user.api_key,
        "unlimited": user.is_admin,
        "limits": {
            "daily_limit": user.daily_limit,
            "monthly_limit": user.monthly_limit,
        },
        "today": {"day": today, "requests": row.requests, "tokens": row.tokens},
        "month": {"month": month, "requests": month_row.requests, "tokens": month_row.tokens},
        "month_days": [
            {"day": d.day, "requests": d.requests, "tokens": d.tokens} for d in month_days
        ],
        "monthly_totals": [
            {"month": t.day, "requests": t.requests, "tokens": t.tokens} for t in monthly_totals
        ],
        "history": [
            {"day": h.day, "requests": h.requests, "tokens": h.tokens} for h in history
        ],
    }


@app.post("/auth/rotate-key")
async def rotate_key(request: Request):
    settings = _auth_settings()
    user = await _require_user(request, settings)
    await _enforce_rate_limit(request, "rotate-key", 10, 60)
    db_user = await database.rotate_api_key(user.id, generate_api_key())
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"api_key": db_user.api_key}


@app.post("/auth/change-password")
async def change_password(req: ChangePasswordRequest, request: Request):
    """Apna password badlo — old password verify karke (hamesha).

    Pehle `old_password` khali chhod kar koi bhi valid JWT wala password
    reset kar sakta tha (silent account takeover). Ab old password zaroori
    hai — jo JWT me hai use password nahi pata, wo change nahi kar sakta.
    """
    settings = _auth_settings()
    user = await _require_user(request, settings)
    await _enforce_rate_limit(request, "change-password", 5, 60)

    db_user = await database.get_user_by_id(user.id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if not req.old_password:
        raise HTTPException(status_code=401, detail="Old password zaroori hai")
    if not await asyncio.to_thread(
        verify_password, req.old_password, db_user.password_hash, db_user.salt
    ):
        raise HTTPException(status_code=401, detail="Old password galat hai")

    password_hash, salt = await asyncio.to_thread(hash_password, req.new_password)
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
    # disk-fill protection — bina bound ke state koi bhi bhej dega to disk bharegi
    if len(json.dumps(req.state, ensure_ascii=False)) > 2_000_000:  # ~2 MB
        raise HTTPException(status_code=413, detail="State 2MB se bada hai")
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
    month = database.month_utc()
    out = []
    for u in users:
        row = await database.get_usage_row(u.id, today)
        month_row = await database.get_usage_month(u.id, month)
        out.append(
            {
                "id": u.id,
                "username": u.username,
                "role": u.role,
                "api_key": u.api_key[:14] + "...",
                "daily_limit": u.daily_limit,
                "monthly_limit": u.monthly_limit,
                "today_requests": row.requests,
                "today_tokens": row.tokens,
                "month_requests": month_row.requests,
                "month_tokens": month_row.tokens,
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
    if req.daily_limit is None and req.monthly_limit is None:
        raise HTTPException(status_code=400, detail="daily_limit ya monthly_limit do")

    target = await database.set_limits(
        user_id, daily_limit=req.daily_limit, monthly_limit=req.monthly_limit
    )
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "ok": True,
        "username": target.username,
        "daily_limit": target.daily_limit,
        "monthly_limit": target.monthly_limit,
    }


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

    # SIRF wo providers dikhte hain jinme kam se kam ek API key configured hai.
    # rotator.providers me sirf key-configured providers build hote hain
    # (bina key wale `_build_state_from_cfg` me skip ho jate hain) — isliye
    # yahan raw config fallback nahi chahiye (pehle PASTE_ placeholder wale
    # bhi "key_count: 0" ke saath dikh rahe the).
    # Catalog bhi isi ke hisaab se filter — bina key wale provider ke models
    # kisi bhi tab me nahi dikhne chahiye.
    active_names = {p["name"] for p in providers}
    catalog = {k: v for k, v in MODEL_CATALOG.items() if k in active_names}

    return {
        "catalog": catalog,
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


def _default_base_url(ptype: str) -> str:
    """Provider type ke liye default base_url (UI me empty chhodne pe use hota hai)."""
    if ptype == "gemini":
        return GEMINI_V1
    if ptype == "openai":
        return ""
    return ""


# --------------------------------------------------------------------------
# Provider presets — "+ Add Provider" picker ke liye.
#
# Naya provider add karte waqt UI me sirf ek provider SELECT karna hota hai —
# base_url + models yahi se auto-fill hote hain, aur API keys kabhi type nahi
# karni padti: woh <NAME>_KEYS env var (ya config.yaml) se seedha detect hoti
# hain (dekho _env_keys_for_name aur admin_add_provider ka fallback).
# Ek hi base_url pe jitni bhi keys env var me comma-separated ho, sab
# automatically round-robin rotate hoti hain (router.py KeyRing) — isliye
# "kitne bhi apikeys ek baseurl pe" already built-in hai, UI me kuch extra
# nahi karna padta.
# --------------------------------------------------------------------------
PROVIDER_PRESETS: dict[str, dict] = {
    "gemini": {
        "label": "Google Gemini",
        "icon": "✨",
        "type": "gemini",
        "base_url": GEMINI_V1,
    },
    "groq": {
        "label": "Groq",
        "icon": "⚡",
        "type": "openai",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "openrouter": {
        "label": "OpenRouter",
        "icon": "🌐",
        "type": "openai",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "nvidia": {
        "label": "NVIDIA NIM",
        "icon": "🟩",
        "type": "openai",
        "base_url": "https://integrate.api.nvidia.com/v1",
    },
    "zen": {
        "label": "OpenCode Zen",
        "icon": "🧘",
        "type": "openai",
        "base_url": "https://opencode.ai/zen/v1",
    },
}


def _env_keys_for_name(name: str) -> list[str]:
    """<NAME>_KEYS env var se comma-separated keys nikaalo (PASTE_ placeholders skip).

    Yehi function poore "Add Provider" flow ka core hai — UI kabhi bhi key
    paste karne ka input nahi dikhati, sirf yahan se detect hoti hain.
    """
    env_name = f"{name.strip().upper()}_KEYS"
    raw = os.environ.get(env_name, "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return [k for k in keys if not k.startswith("PASTE_")]


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
    ptype, base_url, api_keys, models, key_base_urls = None, None, [], [], {}
    for st in rotator.providers:
        if st.cfg.name == name:
            ptype, base_url, api_keys = st.cfg.ptype, st.cfg.base_url, list(st.cfg.keys)
            models = list(st.cfg.models)
            key_base_urls = dict(st.cfg.key_base_urls)
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
            key_base_urls = dict(cp.get("key_base_urls") or {})

    if ptype is None or not api_keys:
        entry = {"fetched_at": now, "models": [], "error": "provider not configured / no keys"}
        cache[name] = entry
        return entry

    # top-level base_url khaali ho sakta hai jab har key ka apna gateway ho
    # (per-key base_url) — live-models listing ke liye pehli available
    # per-key base_url use karo, warna default pe fall karo.
    if not base_url and key_base_urls:
        base_url = next((key_base_urls[k] for k in api_keys if k in key_base_urls), None)

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
@app.get("/admin/providers/catalog")
async def admin_providers_catalog(request: Request):
    """"+ Add Provider" picker ke liye known presets.

    Har preset ka base_url + default models bhi milte hain, aur env var
    (`<NAME>_KEYS`) me kitni keys detect hui unka count bhi — taaki UI
    keys ka koi input dikhaye bina "N keys detected" seedha dikha sake.
    """
    await _require_admin(request)
    rotator: Rotator = request.app.state.rotator
    existing_names = {st.cfg.name for st in rotator.providers}
    custom = await database.list_custom_providers()
    existing_names |= {p.get("name") for p in custom if p.get("name")}

    presets = []
    for name, meta in PROVIDER_PRESETS.items():
        env_keys = _env_keys_for_name(name)
        presets.append(
            {
                "name": name,
                "label": meta["label"],
                "icon": meta["icon"],
                "type": meta["type"],
                "base_url": meta["base_url"],
                "models": MODEL_CATALOG.get(name, []),
                "env_var": f"{name.upper()}_KEYS",
                "env_key_count": len(env_keys),
                "already_added": name in existing_names,
            }
        )
    return {"presets": presets}


@app.get("/admin/providers/detect-keys")
async def admin_detect_keys(name: str, request: Request):
    """Custom/manual provider name ke liye env keys live-detect karo.

    "Add Provider" modal me provider select karte hi (ya "Custom / Other"
    me naam type karte hi) yeh <NAME>_KEYS env var check karke har key ka
    masked preview + index deta hai — taaki UI "API Key 1 / API Key 2 / ..."
    jaise individually SELECT karne ka checkbox list dikha sake. Koi bhi
    key kabhi type/paste nahi karni padti.
    """
    await _require_admin(request)
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    keys = _env_keys_for_name(name)
    return {
        "name": name,
        "env_var": f"{name.upper()}_KEYS",
        "key_count": len(keys),
        "previews": [_key_preview(k) for k in keys],
        "keys": [{"index": i, "preview": _key_preview(k)} for i, k in enumerate(keys)],
    }


class TestProviderUrlInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    type: str = "openai"
    base_url: str = Field("", max_length=512)
    key_indices: list[int] = []   # detect-keys ke "index" — empty = saari detected keys me se pehli


@app.post("/admin/providers/test-url")
async def admin_test_provider_url(req: TestProviderUrlInput, request: Request):
    """"Add Provider" modal ka 🧪 Test button — Save se PEHLE base_url +
    (selected) key(s) se ek live call karke confirm karta hai ki base_url
    valid hai ya nahi. Kuch save nahi hota, sirf ek live models fetch try
    hota hai aur result (ok/error + model count) return hota hai.
    """
    await _require_admin(request)
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    ptype = req.type.strip().lower()
    if ptype not in ("gemini", "openai"):
        raise HTTPException(status_code=400, detail="type must be 'gemini' or 'openai'")
    base_url = req.base_url.strip() or None

    # test ke liye keys: env se, warna already-saved (existing custom /
    # runtime) provider ki keys se — jo bhi mile
    candidate_keys = _env_keys_for_name(name)
    if not candidate_keys:
        rotator: Rotator = request.app.state.rotator
        st = rotator._find_provider(name)
        if st is not None:
            candidate_keys = list(st.cfg.keys)
    if not candidate_keys:
        existing = await database.get_custom_provider(name)
        if existing:
            candidate_keys = list(existing.get("api_keys", []))

    if req.key_indices:
        test_keys = [candidate_keys[i] for i in req.key_indices if 0 <= i < len(candidate_keys)]
    else:
        test_keys = candidate_keys

    if not test_keys:
        return {"ok": False, "message": "❌ Test karne ke liye koi key nahi mili"}
    if ptype == "openai" and not base_url:
        return {"ok": False, "message": "❌ openai provider ke liye base_url zaroori hai"}

    try:
        live = await fetch_live_models(name, ptype, base_url, test_keys[:1], timeout=15.0, max_pages=1)
        return {
            "ok": True,
            "message": f"✅ base_url valid — {len(live)} model{'s' if len(live) != 1 else ''} mile",
            "model_count": len(live),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": f"❌ {str(exc)[:300]}"}


@app.post("/admin/providers/{name}/resync-keys")
async def admin_resync_provider_keys(name: str, request: Request):
    """Provider ki keys `<NAME>_KEYS` env var se dobara pull karo.

    Env var me baad me aur keys add karo (comma-separated) toh yahan se
    resync karke woh naya set turant rotation me aa jaata hai — UI me
    kabhi kisi key ko haath se paste nahi karna padta.
    """
    await _require_admin(request)
    name = name.strip()
    env_keys = _env_keys_for_name(name)
    if not env_keys:
        raise HTTPException(
            status_code=404,
            detail=f"'{name.upper()}_KEYS' env var me koi key nahi mili",
        )

    rotator: Rotator = request.app.state.rotator
    existing = await database.get_custom_provider(name)
    if existing:
        provider = dict(existing)
        provider["api_keys"] = env_keys
        provider["selected_keys"] = []  # purane indices ab invalid ho sakte — reset
        # key_base_urls purani keys ko refer karta tha, naye set pe map nahi hoga
        provider["key_base_urls"] = {
            k: v for k, v in (provider.get("key_base_urls") or {}).items() if k in env_keys
        }
        await database.upsert_custom_provider(provider)
        custom = await database.list_custom_providers()
        rotator.apply_custom_providers(custom)
        return {"ok": True, "message": f"'{name}' ki keys env se resync ho gayi", "key_count": len(env_keys)}

    # config.yaml wala provider — uska runtime state boot ke waqt hi env
    # se ban chuka hota hai, isliye alag se sync ki zaroorat nahi.
    return {
        "ok": True,
        "message": f"'{name}' config.yaml se manage hota hai — env keys already applied hain",
        "key_count": len(env_keys),
    }


@app.get("/admin/providers")
async def admin_providers(request: Request):
    """Providers ka editor view — config.yaml wale + custom wale DONO.

    UI ke "🔌 Providers" tab ke liye: har provider ka base_url, models,
    masked key previews aur enabled state milta hai taaki admin bina
    config.yaml chhede sab set kar sake. Keys kabhi plaintext return
    nahi hoti (sirf preview) — POST karte waqt nayi keys bheji jaati hain.
    """
    await _require_admin(request)
    rotator: Rotator = request.app.state.rotator

    # 1) config.yaml wale providers (raw config se — default base_url/models)
    config_providers: dict[str, dict] = {}
    try:
        for p in _load_config().get("providers", []):
            name = (p.get("name") or "").strip()
            if not name:
                continue
            config_providers[name] = {
                "name": name,
                "type": p.get("type", "openai"),
                "base_url": p.get("base_url", ""),
                "models": [m.strip() for m in p.get("models", []) if m.strip()],
                "enabled": True,
                "source": "config",
                "key_count": 0,
                "keys": [],
            }
    except Exception:  # noqa: BLE001
        pass

    # 2) runtime state se env-keys wale base_url/key_count sync karo
    for st in rotator.providers:
        entry = config_providers.get(st.cfg.name)
        if entry is None:
            # runtime me hai par config me nahi (dashboard se add kiya tha)
            entry = {
                "name": st.cfg.name,
                "type": st.cfg.ptype,
                "base_url": st.cfg.base_url or "",
                "models": list(st.cfg.models),
                "enabled": True,
                "source": "runtime",
                "key_count": len(st.cfg.keys),
                "keys": [
                    {
                        "index": i,
                        "preview": _key_preview(k),
                        "base_url": st.cfg.key_base_urls.get(k, ""),
                        "selected": True,
                    }
                    for i, k in enumerate(st.cfg.keys)
                ],
                "key_base_urls": dict(st.cfg.key_base_urls),
            }
            config_providers[st.cfg.name] = entry
        else:
            entry["key_count"] = len(st.cfg.keys)
            entry["keys"] = [
                {
                    "index": i,
                    "preview": _key_preview(k),
                    "base_url": st.cfg.key_base_urls.get(k, ""),
                    "selected": True,
                }
                for i, k in enumerate(st.cfg.keys)
            ]
            entry["key_base_urls"] = dict(st.cfg.key_base_urls)
            if st.cfg.base_url:
                entry["base_url"] = st.cfg.base_url

    # 3) custom providers (dashboard se add) — inki encrypted keys store me hain
    custom = await database.list_custom_providers()
    custom_names = set()
    for p in custom:
        name = (p.get("name") or "").strip()
        if not name:
            continue
        custom_names.add(name)
        # sel_keys empty = router._merge_custom_states saari keys active rakhta
        # hai (koi filter nahi) — UI ko bhi wahi "sab selected" dikhana chahiye,
        # warna admin ko lagta hai kisi key select nahi hui, aur ek-do checkbox
        # tick karke Save karne pe baaki keys hamesha ke liye deselect ho jaati
        # hain (provider ke effective keys ghat jaate hain — group editor me
        # bhi kam keys dikhti hain, "virtual model multiple keys" tootta hai).
        sel_keys = p.get("selected_keys") or []
        config_providers[name] = {
            "name": name,
            "type": p.get("type", "openai"),
            "base_url": p.get("base_url", ""),
            "models": [m.strip() for m in p.get("models", []) if m.strip()],
            "enabled": p.get("enabled", True),
            "source": "custom",
            "key_count": len(p.get("api_keys", [])),
            "key_base_urls": dict(p.get("key_base_urls") or {}),
            "selected_keys": sel_keys,
            "keys": [
                {
                    "index": i,
                    "preview": _key_preview(k),
                    "base_url": (p.get("key_base_urls") or {}).get(k, ""),
                    "selected": (not sel_keys) or (i in sel_keys),
                }
                for i, k in enumerate(p.get("api_keys", []))
            ],
        }

    # consistent order: config order pehle, phir naye custom
    all_names = list(config_providers.keys())
    ordered = [
        config_providers[n] for n in all_names if n not in custom_names
    ] + [config_providers[n] for n in all_names if n in custom_names]

    # sirf dashboard-se-managed providers dikhao — config.yaml ke default
    # placeholder providers (0 keys, PASTE_ wale) hide. Env secrets se keys
    # milne wale config providers (GEMINI_KEYS etc.) MUST show — user ko pata
    # hona chahiye ki unki Render secrets detect hui hain.
    ordered = [p for p in ordered if p.get("source") != "config" or p.get("key_count", 0) > 0]

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
    return {"providers": ordered, "cache": status}


@app.post("/admin/providers")
async def admin_add_provider(req: CustomProviderInput, request: Request):
    """Naya provider add karo (ya same name pe update) + live apply.

    UI "🔌 Providers" tab se base_url / keys / models update karne ke liye:
      - `api_keys` empty bhejo + `replace_keys: false`  → existing keys preserve
      - `api_keys` + `replace_keys: false`             → nayi keys MERGE hoti hain
      - `replace_keys: true`                           → purani keys hatake nayi
      - `base_url` empty bhejo                         → existing preserve
      - `models` empty bhejo                           → existing preserve
    """
    await _require_admin(request)

    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Provider name required")
    ptype = req.type.strip().lower()
    if ptype not in ("gemini", "openai"):
        raise HTTPException(status_code=400, detail="type must be 'gemini' or 'openai'")

    rotator: Rotator = request.app.state.rotator
    existing = await database.get_custom_provider(name)

    # runtime provider state (config.yaml + env keys) — custom update me
    # existing keys preserve karne ke liye (merge bahaviour)
    runtime_st = rotator._find_provider(name)
    runtime_cfg = {"base_url": "", "keys": [], "models": []}
    if runtime_st is not None:
        runtime_cfg = {
            "base_url": runtime_st.cfg.base_url or "",
            "keys": list(runtime_st.cfg.keys),
            "models": list(runtime_st.cfg.models),
        }

    # ---- base_url: empty → existing custom / runtime config preserve ----
    base_url = req.base_url.strip()
    if not base_url:
        if existing and existing.get("base_url"):
            base_url = existing["base_url"]
        elif runtime_cfg["base_url"]:
            base_url = runtime_cfg["base_url"]
        else:
            base_url = _default_base_url(ptype)

    # ---- keys: merge / replace / preserve ----
    new_keys = [k.strip() for k in req.api_keys if k.strip() and not k.startswith("PASTE_")]
    if existing:
        old_keys = list(existing.get("api_keys", []))
    else:
        # custom nahi hai → runtime (config/env) keys preserve karo
        old_keys = [k for k in runtime_cfg["keys"] if not k.startswith("PASTE_")]
        if not old_keys:
            # bilkul naya provider (config.yaml me bhi nahi) — seedha
            # <NAME>_KEYS env var se keys pull karo. "Add Provider" UI
            # kabhi api_keys nahi bhejti, isliye yehi asli source hai.
            old_keys = _env_keys_for_name(name)
    if req.replace_keys:
        keys = new_keys
    elif new_keys:
        # merge — duplicate na ho
        keys = old_keys[:]
        seen = set(keys)
        for k in new_keys:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    else:
        keys = old_keys
    if not keys:
        raise HTTPException(status_code=400, detail="At least one API key required")

    # ---- per-key base_url resolve karo (validation ke liye keys chahiye,
    # isliye yeh keys resolve hone ke baad hi ho sakta hai) ----
    resolved_key_base_urls = _resolve_key_base_urls(req.key_base_urls, keys)

    # top-level base_url zaroori hai SIRF unhi keys ke liye jinka apna
    # per-key base_url set nahi hai — agar har key ka alag gateway hai
    # (jaise Cloudflare Worker per key), top-level blank chhodna valid hai.
    if ptype == "openai" and not base_url:
        keys_missing_base_url = [k for k in keys if k not in resolved_key_base_urls]
        if keys_missing_base_url:
            raise HTTPException(
                status_code=400,
                detail="openai provider needs base_url (ya har key ka apna base_url set karo)",
            )

    # ---- models: empty → existing custom / runtime config preserve ----
    # Models ab yahan MANDATORY nahi — model select karna 🎚 Exposed Models
    # tab ka kaam hai (live-fetch se). Provider bina models ke bhi add ho
    # sakta hai; base_url/keys set hote hi Exposed tab me live models fetch
    # karke expose kar sakte ho, koi "at least one model" gate nahi.
    models = [m.strip() for m in req.models if m.strip()]
    if not models and existing and existing.get("models"):
        models = [m.strip() for m in existing.get("models", []) if m.strip()]
    if not models and runtime_cfg["models"]:
        models = runtime_cfg["models"]

    provider = {
        "name": name,
        "type": ptype,
        "base_url": base_url,
        "api_keys": keys,
        "models": models,
        "enabled": req.enabled,
        # key_base_urls: {key: url} ya {index: url} dono accept — index wale ko resolve
        "key_base_urls": resolved_key_base_urls,
        # selected_keys: sirf ye keys rotation me use hongi (empty = sab selected)
        "selected_keys": [i for i in (req.selected_keys or []) if 0 <= i < len(keys)],
    }
    await database.upsert_custom_provider(provider)

    # live apply karo
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
        "key_count": len(keys),
        "live": (entry or {}).get("models", []),
        "live_error": (entry or {}).get("error"),
    }


def _resolve_key_base_urls(raw: dict, keys: list[str]) -> dict:
    """Per-key base_url map resolve karo — {key: url} ya {index: url} dono chalta hai.
    (UI index-based bhejta hai kyunki full keys plaintext return nahi hoti.)"""
    out: dict[str, str] = {}
    for kk, vv in (raw or {}).items():
        if not vv:
            continue
        if kk.isdigit() and int(kk) < len(keys):
            out[keys[int(kk)]] = vv.strip()
        elif kk in keys:
            out[kk] = vv.strip()
    return out


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
# ------------------------------------------------------------------
# SSE streaming (`stream: true`)
#
# BUG FIX: pehle `stream: true` sirf "accepted for compatibility" tha —
# server hamesha ek normal `application/json` body wapas bhejta tha, kabhi
# `text/event-stream` nahi. Isse do problems hoti thi:
#
#   1. Bade OpenAI-compatible clients (Open WebUI, LibreChat, SillyTavern,
#      Chatbox, TypingMind, ...) `stream: true` bhejte hain aur seedha SSE
#      parse karte hain — unko valid `data: {...}\n\n` frames na milne pe
#      wo request hi fail/hang kar dete the (chahe model normal ho ya
#      virtual group jaisa "levelup").
#   2. LevelUp app khud bhi pehle `stream: true` try karta hai aur SSE
#      parser ko 0 chunks milte the → wo chup-chaap non-stream call se
#      dobara try karta tha (fallback). Matlab HAR message ke liye do
#      real upstream calls (Gemini) ho rahi thi — quota/RPM DOUBLE use ho
#      raha tha 9 keys hone ke bawajood bhi "sab kaam nahi karte" isi wajah
#      se lagta tha.
#
# Fix: rotation/fallback/group logic bilkul same rehta hai (poora result
# pehle hi mil chuka hota hai, jaise non-stream path me) — bas response ko
# proper OpenAI-style SSE frames me chunk karke bhejte hain. Isse dono
# problems fix ho jaati hain, bina rotator/provider logic chhede.
# ------------------------------------------------------------------
def _sse_text_chunks(text: str, words_per_chunk: int = 3) -> list[str]:
    """Text ko typing-effect ke liye chhote pieces me todta hai.
    Whitespace bilkul preserve hota hai (koi word split nahi hota)."""
    if not text:
        return []
    parts = re.findall(r"\S+\s*", text)
    if not parts:
        return [text]
    return ["".join(parts[i : i + words_per_chunk]) for i in range(0, len(parts), words_per_chunk)]


async def _stream_chat_completion(result, req: "ChatCompletionRequest"):
    """Ek complete ChatResult ko OpenAI-compatible SSE chunk stream me convert karta hai."""
    created = int(time.time())
    base = {
        "id": "chatcmpl-rotator",
        "object": "chat.completion.chunk",
        "created": created,
        "model": result.model,
    }

    def frame(delta: dict, finish_reason: Optional[str] = None) -> str:
        payload = {
            **base,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    # 1. role chunk — clients isi se assistant message shuru karte hain
    yield frame({"role": "assistant", "content": ""})

    # 2. tool_calls (agar hain) — ek hi chunk me poora bhej dete hain
    #    (true incremental function-call streaming abhi support nahi hai,
    #    par clients ko poora tool_calls array milte hi kaam ho jata hai)
    if result.tool_calls:
        yield frame({"tool_calls": result.tool_calls})

    # 3. content — chhote-chhote pieces me (typing effect)
    for piece in _sse_text_chunks(result.text):
        yield frame({"content": piece})
        await asyncio.sleep(0.01)

    # 4. reasoning (agar hai)
    if result.reasoning_content:
        yield frame({"reasoning_content": result.reasoning_content})

    # 5. finish chunk
    yield frame({}, finish_reason="tool_calls" if result.tool_calls else "stop")
    yield "data: [DONE]\n\n"


def _sse_error_response(detail: str, code: int = 503) -> StreamingResponse:
    """Streaming request pe error OpenAI-style SSE frame me bhejo.

    LevelUp jaise SSE parsers non-200 response ko sirf "SSE HTTP 503" jaise
    generic text se handle karte hain — actual error message kabhi nahi
    dikhta. Isliye jab client `stream: true` bhejta hai, provider error ko
    `data: {"error": {...}}` + `[DONE]` frame (HTTP 200) me convert karte
    hain. OpenAI-compatible clients ise error ki tarah parse karte hain.
    """
    payload = {
        "error": {
            "message": detail,
            "type": "server_error",
            "code": code,
        }
    }
    return StreamingResponse(
        iter([f"data: {json.dumps(payload, ensure_ascii=False)}\n\ndata: [DONE]\n\n"]),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
            top_p=req.top_p,
            stop=req.stop,
            presence_penalty=req.presence_penalty,
            frequency_penalty=req.frequency_penalty,
            response_format=req.response_format,
            seed=req.seed,
            logit_bias=req.logit_bias,
        )
    except RateLimitError as exc:
        # provider rate-limit — user ko raw message NAHI dikhate
        if user:
            await _refund_quota(user)
        logger.warning("chat: rate limited upstream: %s", exc)
        if req.stream:
            return _sse_error_response(
                "Server abhi bahut load me hai, kuch minute baad try karo.", code=503
            )
        raise HTTPException(
            status_code=503,
            detail="Server abhi bahut load me hai, kuch minute baad try karo.",
        ) from exc
    except AllProvidersExhausted as exc:
        # saare providers ki keys/models fail
        if user:
            await _refund_quota(user)
        logger.error("chat: all providers exhausted: %s", exc)
        detail = (
            "Something went wrong — saare AI providers abhi busy/ exhausted hain. "
            "Thodi der baad try karo, ya apna configured provider use karo. "
            "(sahi provider error `/status` ya admin panel me dikhta hai)"
        )
        if req.stream:
            return _sse_error_response(detail, code=503)
        raise HTTPException(status_code=503, detail=detail) from exc
    except ProviderError as exc:
        # koi aur provider error (config galat, network, etc.)
        if user:
            await _refund_quota(user)
        logger.error("chat: provider error: %s", exc)
        status_code = exc.status_code if exc.status_code else 502
        detail = "Server ko AI provider se connect karne me problem aayi. Kuch minute baad try karo."
        if req.stream:
            return _sse_error_response(detail, code=status_code)
        raise HTTPException(
            status_code=status_code,
            detail=detail,
        ) from exc

    # record actual token usage
    if user:
        tokens = (
            result.usage.get("totalTokenCount", result.usage.get("total_tokens", 0)) or 0
        )
        await _record_tokens(user, int(tokens))

    # Defensive guard — blank reply (empty / whitespace / zero-width only)
    # kabhi bhi user tak na pahunche. Router ise already failure treat karta
    # hai, par belt-and-suspenders: agar kisi path se phir bhi aa jaye toh 502.
    if is_blank_text(result.text) and not result.tool_calls:
        if user:
            await _refund_quota(user)
        logger.warning(
            "chat: BLANK reply router se aa gaya (%s/%s) — zero-width/whitespace",
            result.provider,
            result.model,
        )
        detail = "AI model ne khaali reply diya (saara token budget thinking me chala gaya). Thodi der baad try karo ya max_tokens badhao."
        if req.stream:
            return _sse_error_response(detail, code=502)
        raise HTTPException(
            status_code=502,
            detail=detail,
        )

    # web search indicator — raw Gemini response se grounding check karo.
    # Client ko test karne me asaan ho: `"web_search": true` = search hua,
    # `false` = search tool drop ho gaya (openai-type member) ya koi grounding nahi.
    gm = (result.raw or {}).get("candidates", [{}])[0].get("groundingMetadata") or {}
    ws_used = bool(
        gm.get("groundingChunks") or gm.get("webSearchQueries") or gm.get("searchEntryPoint")
    )
    ws_queries = gm.get("webSearchQueries") or []

    if req.stream:
        return StreamingResponse(
            _stream_chat_completion(result, req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # nginx/reverse-proxy buffering off
            },
        )

    return JSONResponse(
        content={
            "id": "chatcmpl-rotator",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result.model,
            "provider": result.provider,
            "key": result.key_label,
            "web_search": ws_used,
            **({"search_queries": ws_queries} if ws_queries else {}),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.text,
                        **({"tool_calls": result.tool_calls} if result.tool_calls else {}),
                        **({"reasoning_content": result.reasoning_content} if result.reasoning_content else {}),
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


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingsRequest, request: Request):
    """OpenAI-compatible embeddings — Gemini text-embedding models ke through.

    Body: {"model": "text-embedding-004", "input": "text" | ["t1","t2"]}
    Auth + quota: /v1/chat/completions jaisa hi (Bearer JWT ya sk- key).
    """
    rotator: Rotator = request.app.state.rotator
    settings = _auth_settings()

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
                detail="Aapka daily quota khatam ho gaya. Kal reset hoga.",
            )

    # input normalize — OpenAI string ya list dono accept karta hai
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        if user:
            await _refund_quota(user)
        raise HTTPException(status_code=400, detail="input empty hai")

    # Gemini provider + keys chahiye (sirf Gemini embeddings support karta hai)
    gemini_st = next(
        (st for st in rotator.providers if st.cfg.ptype == "gemini" and st.cfg.keys),
        None,
    )
    if gemini_st is None:
        if user:
            await _refund_quota(user)
        logger.error("embeddings: koi Gemini provider/keys configured nahi")
        raise HTTPException(
            status_code=503,
            detail="Embeddings ke liye koi Gemini provider configured nahi hai.",
        )

    gemini_provider = gemini_st.provider
    keys = gemini_st.ring.keys_as_list()

    # keys pe chota failover: pehli available key try karo, fail pe agli.
    # (pura rotation loop router jaisa banana overkill — embeddings rare hai)
    last_err: Optional[Exception] = None
    for idx, key in enumerate(keys[:3]):
        try:
            proxy = None
            if rotator.proxy_pool is not None:
                proxy = rotator.proxy_pool.next()
            result = await gemini_provider.embeddings(
                texts, model=req.model, proxy=proxy, api_key=key
            )
            if user:
                tokens = result.get("usage", {}).get("total_tokens", 0) or 0
                await _record_tokens(user, int(tokens))
            return JSONResponse(content=result)
        except (RateLimitError, ProviderError) as exc:
            last_err = exc
            # rate limit / transient — agli key try karo
            if idx == len(keys[:3]) - 1:
                break
            continue

    if user:
        await _refund_quota(user)
    logger.error("embeddings: sab keys fail — %s", last_err)
    raise HTTPException(
        status_code=503,
        detail="Embeddings generate karne me problem aayi. Kuch minute baad try karo.",
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
        try:
            return await database.get_user_by_id(int(payload["sub"]))
        except (KeyError, ValueError, TypeError):
            # validly-signed token me sub galat type/format — generic 500 na de
            logger.warning("auth: JWT payload me invalid 'sub' (%r)", payload.get("sub"))
            return None
    return None


async def _require_user(request: Request, settings: dict):
    user = await _authenticate(request, settings)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


async def _reserve_quota(user) -> bool:
    """Request reserve — quota bacha hai toh True.

    Admin ke liye unlimited (koi check nahi), PAR usage track hota hai —
    dashboard me admin ka usage bhi dikhta hai (limit me ♾️ symbol).
    Normal user: daily + monthly dono check hota hai.
    """
    if user.is_admin:
        await database.reserve_unlimited(user.id, database.today_utc())
        return True
    return await database.reserve_quota_with_monthly(
        user.id,
        database.today_utc(),
        database.month_utc(),
        user.daily_limit,
        user.monthly_limit,
    )


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
        "monthly_limit": user.monthly_limit,
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
    # DeepSeek round-trip: client assistant message ke saath reasoning_content
    # wapas bhejta hai — drop mat karo, warna agli request pe provider 400 dega.
    reasoning_content = m.get("reasoning_content") or ""

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
        reasoning_content=reasoning_content,
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
      <div class="tab" id="exposedTab" style="display:none" onclick="showView('exposed'); loadExposed();">🎚 Exposed</div>
      <div class="tab" id="providersTab" style="display:none" onclick="showView('providers'); loadProvidersAdmin();">🔌 Providers</div>
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
          <h3 style="margin:20px 0 10px">This Month</h3>
          <div class="stat"><span>Requests used</span><span id="u-mused">–</span></div>
          <div class="stat"><span>Monthly limit</span><span id="u-mlimit">–</span></div>
          <div class="stat"><span>Tokens consumed</span><span id="u-mtokens">–</span></div>
          <h3 style="margin:20px 0 10px">This Month — Daily</h3>
          <div class="history" id="u-mgraph"></div>
          <h3 style="margin:20px 0 10px">Last 6 Months</h3>
          <div class="history" id="u-mtotals"></div>
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
          <thead><tr><th>ID</th><th>Username</th><th>Role</th><th>Key</th><th>Today (req · tok)</th><th>This Month (req · tok)</th><th>Daily limit</th><th>Monthly limit</th></tr></thead>
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

    <!-- EXPOSED MODELS VIEW -->
    <div class="view" id="view-exposed">
      <div class="card">
        <h3 style="margin-bottom:4px">Exposed Models</h3>
        <div class="muted" style="margin-bottom:10px">Har provider ke saare models (live API se fetch) me se select karo — <b>/v1/models</b> me external apps ko SIRF checked wale dikhenge. Routing bhi sirf inhi models me hogi. 🔄 Refresh se naye models aa jate hain. 💡</div>
        <div class="err" id="exposed-err"></div>
        <div class="ok" id="exposed-ok"></div>
        <div style="margin:10px 0;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
          <button class="btn sec" onclick="refreshExposedLive()">🔄 Refresh Live Models</button>
          <button class="btn" onclick="saveExposed()">💾 Save Exposed Models</button>
          <span class="muted" id="exposed-status"></span>
        </div>
        <div id="exposed-list"><span class="muted">Loading…</span></div>
      </div>
    </div>

    <!-- PROVIDERS ADMIN (env keys + base_url UI se set karo) -->
    <div class="view" id="view-providers">
      <div class="card">
        <h3 style="margin-bottom:4px">🔌 Provider Settings</h3>
        <div class="muted" style="margin-bottom:10px">API keys sirf <b>env secrets</b> se aati hain (<b>GEMINI_KEYS</b>, <b>GROQ_KEYS</b>, <b>OPENROUTER_KEYS</b>, <b>NVIDIA_KEYS</b>, <b>ZEN_KEYS</b>, ...) — yahan UI me kabhi key type/paste nahi karni padti. "+ Add Provider" se ek provider select karo, base_url + detected keys apne aap dikh jayenge. Har provider card pe jo keys use karni hain unhi ko select karo — sab selected keys ek hi base_url pe automatically rotate hoti hain. Models yahan set nahi hote — woh <b>🎚 Exposed Models</b> tab se manage karo. 💾</div>
        <div style="margin-bottom:12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <button class="btn" onclick="openAddProviderModal()">➕ Add Provider</button>
          <button class="btn sec" onclick="refreshProvidersLive()">🔄 Refresh Live Models</button>
        </div>
        <div class="err" id="providers-err"></div>
        <div class="ok" id="providers-ok"></div>
        <div id="providers-list"><span class="muted">Loading…</span></div>
      </div>
    </div>
  </div>
</div>

<!-- ADD PROVIDER MODAL -->
<div class="modal" id="addProviderModal" style="display:none">
  <div class="modal-box">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
      <h3 style="margin:0" id="addProviderTitle">➕ Add Provider</h3>
      <button class="btn sec" onclick="closeAddProviderModal()">✕</button>
    </div>

    <!-- STEP 1: pick a provider -->
    <div id="addProviderStep1">
      <div class="muted" style="margin-bottom:10px">Provider select karo — base_url aur configured keys apne aap load ho jayengi.</div>
      <div id="addProviderPresets"><span class="muted">Loading…</span></div>
    </div>

    <!-- STEP 2: confirm + save (no key input anywhere — auto-detected only) -->
    <div id="addProviderStep2" style="display:none">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
        <button class="btn sec" onclick="backToProviderPresets()">‹ Back</button>
        <b id="ap-name-label" style="font-size:15px"></b>
        <span class="badge" id="ap-type-badge" style="background:var(--panel2);padding:2px 8px;border-radius:10px;font-size:12px"></span>
      </div>

      <div id="ap-custom-name-wrap" style="display:none">
        <label>Provider name (env var: <span id="ap-custom-envname" class="muted"></span>)</label>
        <input id="ap-custom-name" type="text" placeholder="jaise: cerebras" oninput="onCustomProviderNameInput()">
        <label>Type</label>
        <select id="ap-custom-type" onchange="onCustomProviderTypeChange()" style="width:100%;background:var(--panel2);color:var(--text);border:1px solid var(--border);border-radius:10px;padding:12px">
          <option value="openai">openai (OpenAI-compatible /chat/completions)</option>
          <option value="gemini">gemini (Google Gemini API)</option>
        </select>
      </div>

      <label>Base URL</label>
      <input id="ap-base-url" type="text" placeholder="https://...">

      <div class="ok" id="ap-keys-status" style="margin-top:12px"></div>
      <div id="ap-keys-list" style="margin-top:6px"></div>
      <div class="err" id="ap-keys-err"></div>

      <div style="margin-top:12px;display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <button class="btn sec" onclick="testAddProviderBaseUrl()">🧪 Test Base URL</button>
        <span id="ap-test-result" class="muted" style="font-size:12px"></span>
      </div>

      <div class="muted" style="margin-top:10px;font-size:12px">Models yahan nahi chunte — provider save hote hi <b>🎚 Exposed Models</b> tab me jaake live models fetch karke expose kar dena.</div>

      <div style="margin-top:12px"><span class="muted" style="font-size:12px"><input id="ap-enabled" type="checkbox" checked> enabled</span></div>

      <div style="display:flex;justify-content:flex-end;gap:8px;margin-top:14px">
        <button class="btn sec" onclick="closeAddProviderModal()">Cancel</button>
        <button class="btn" id="ap-submit-btn" onclick="submitNewProvider()">💾 Save</button>
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
  document.getElementById('exposedTab').style.display = isAnyAdmin ? '' : 'none';
  document.getElementById('providersTab').style.display = isSuperAdmin ? '' : 'none';
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
  const u = data.user, t = data.today, m = data.month || {}, lim = data.limits || {};
  const unlimited = !!data.unlimited;
  document.getElementById('u-used').textContent = t.requests;
  document.getElementById('u-tokens').textContent = (t.tokens || 0).toLocaleString();
  document.getElementById('u-mused').textContent = m.requests;
  document.getElementById('u-mtokens').textContent = (m.tokens || 0).toLocaleString();
  if (unlimited) {
    // admin/super admin — usage dikhta hai, limit me ♾️ unlimited
    document.getElementById('u-limit').textContent = '♾️ unlimited';
    document.getElementById('u-mlimit').textContent = '♾️ unlimited';
    document.getElementById('u-bar').style.width = '0%';
    document.getElementById('u-remaining').textContent = 'Admin hai — koi limit nahi! 🎉 (usage track hota rehta hai)';
  } else {
    document.getElementById('u-limit').textContent = lim.daily_limit;
    document.getElementById('u-mlimit').textContent = lim.monthly_limit;
    const pct = Math.min(100, (t.requests / lim.daily_limit) * 100);
    document.getElementById('u-bar').style.width = pct + '%';
    document.getElementById('u-remaining').textContent =
      (lim.daily_limit - t.requests) + ' requests remaining today · ' +
      Math.max(0, lim.monthly_limit - (m.requests || 0)) + ' remaining this month';
  }
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

  // This Month — daily bars (month_days already sorted by day)
  const mg = document.getElementById('u-mgraph');
  mg.innerHTML = '';
  const mdays = data.month_days || [];
  const mmax = Math.max(1, ...mdays.map(d => d.requests));
  mdays.forEach(d => {
    const col = document.createElement('div'); col.className = 'hcol';
    const bar = document.createElement('div'); bar.className = 'hbar';
    bar.title = 'Day ' + d.day + ': ' + d.requests + ' req · ' + (d.tokens || 0).toLocaleString() + ' tok';
    bar.style.height = Math.max(2, (d.requests / mmax) * 100) + 'px';
    const day = document.createElement('div'); day.className = 'hday'; day.textContent = d.day;
    col.appendChild(bar); col.appendChild(day);
    mg.appendChild(col);
  });
  if (!mdays.length) mg.innerHTML = '<div class="muted">Is month abhi tak koi usage nahi.</div>';

  // Last 6 Months — month totals bars
  const mt = document.getElementById('u-mtotals');
  mt.innerHTML = '';
  const mtot = data.monthly_totals || [];
  const tmax = Math.max(1, ...mtot.map(mo => mo.requests));
  mtot.forEach(mo => {
    const col = document.createElement('div'); col.className = 'hcol';
    const bar = document.createElement('div'); bar.className = 'hbar';
    bar.title = mo.month + ': ' + mo.requests + ' req · ' + (mo.tokens || 0).toLocaleString() + ' tok';
    bar.style.height = Math.max(2, (mo.requests / tmax) * 100) + 'px';
    const day = document.createElement('div'); day.className = 'hday'; day.textContent = mo.month.slice(2);
    col.appendChild(bar); col.appendChild(day);
    mt.appendChild(col);
  });
  if (!mtot.length) mt.innerHTML = '<div class="muted">Abhi koi monthly data nahi.</div>';
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
    const isAdm = u.role === 'admin';
    const limitCell = (val) => isAdm
      ? '<span style="color:var(--accent);font-weight:bold">♾️ unlimited</span>'
      : `<input type="number" value="${val}" min="1" style="width:80px;padding:6px" onchange="setLimit(${u.id}, this.value, 'daily')">`;
    const mlimitCell = (val) => isAdm
      ? '<span style="color:var(--accent);font-weight:bold">♾️ unlimited</span>'
      : `<input type="number" value="${val}" min="1" style="width:90px;padding:6px" onchange="setLimit(${u.id}, this.value, 'monthly')">`;
    tr.innerHTML = `<td>${u.id}</td><td>${u.username}</td>
      <td><select class="role-sel" data-id="${u.id}" onchange="setRole(${u.id}, this.value)" style="background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:4px 6px;color:var(--text)">
        <option value="user" ${u.role === 'user' ? 'selected' : ''}>user</option>
        <option value="admin" ${u.role === 'admin' ? 'selected' : ''}>admin</option>
      </select></td>
      <td class="muted">${u.api_key}</td>
      <td>${u.today_requests}<span class="muted"> · ${(u.today_tokens || 0).toLocaleString()}</span></td>
      <td>${u.month_requests}<span class="muted"> · ${(u.month_tokens || 0).toLocaleString()}</span></td>
      <td>${limitCell(u.daily_limit)}</td>
      <td>${mlimitCell(u.monthly_limit)}</td>`;
    tb.appendChild(tr);
  });
}
async function setRole(id, role) {
  const { res, data } = await api('/admin/users/' + id + '/role', { method: 'POST', body: JSON.stringify({ role }) });
  if (!res.ok) { alert(data.detail || 'Failed'); }
  else setSettingsMsg('✅ ' + data.username + ' → ' + data.role, true);
  loadAdmin();
}
async function setLimit(id, value, which) {
  const payload = which === 'monthly' ? { monthly_limit: parseInt(value) } : { daily_limit: parseInt(value) };
  const { res, data } = await api('/admin/users/' + id + '/limit', { method: 'POST', body: JSON.stringify(payload) });
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

// ---------- exposed models (per-provider selection tab) ----------
let exposedData = null;   // /admin/models response
let exposedSel = {};      // provider -> [selected models]

async function loadExposed() {
  const { res, data } = await api('/admin/models');
  if (!res.ok) { document.getElementById('exposed-err').textContent = data.detail || 'Failed'; return; }
  exposedData = data;
  // selection init: managed.provider_models (saved selection); agar kisi
  // provider ke liye kuch save nahi → configured_models default.
  const pm = (data.managed || {}).provider_models || {};
  exposedSel = {};
  (data.providers || []).forEach(p => {
    exposedSel[p.name] = Array.isArray(pm[p.name]) ? pm[p.name].slice() : (p.configured_models || []).slice();
  });
  renderExposed();
}

// provider ke saare possible models: LIVE (API se) + catalog + configured
function exposedModelOptions(providerName) {
  const p = (exposedData.providers || []).find(x => x.name === providerName);
  const live = (p && p.live_models) || [];
  const cat = (exposedData.catalog && exposedData.catalog[providerName]) || [];
  const conf = (p && p.configured_models) || [];
  return Array.from(new Set(live.concat(cat).concat(conf))).sort();
}

function renderExposed() {
  const box = document.getElementById('exposed-list');
  box.innerHTML = '';
  let shown = 0;
  (exposedData.providers || []).forEach(p => {
    const opts = exposedModelOptions(p.name);
    if (!opts.length) return; // models nahi mile — skip (bina keys/config wale)
    shown++;
    const sel = exposedSel[p.name] || [];
    const card = document.createElement('div');
    card.style.cssText = 'background:var(--panel2);border:1px solid var(--border);border-radius:12px;padding:14px;margin-bottom:12px';
    const chips = opts.map(m => {
      const on = sel.includes(m);
      return `<label class="chip ${on ? 'checked' : ''}" style="cursor:pointer"><input type="checkbox" ${on ? 'checked' : ''} onchange="toggleExposed('${esc(p.name)}','${esc(m)}',this.checked)">${esc(m)}</label>`;
    }).join('');
    card.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px">
        <div><strong>${esc(p.name)}</strong> <span class="tag">${esc(p.type)}</span> <span class="tag">${p.key_count || 0} keys</span></div>
        <div style="display:flex;gap:6px;align-items:center"><span class="tag" style="color:var(--accent2)">${sel.length}/${opts.length}</span>
          <button class="btn sec" style="padding:4px 10px;font-size:12px" onclick="exposedAll('${esc(p.name)}', true)">All</button>
          <button class="btn sec" style="padding:4px 10px;font-size:12px" onclick="exposedAll('${esc(p.name)}', false)">None</button>
        </div>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:8px">${chips}</div>`;
    box.appendChild(card);
  });
  if (!shown) box.innerHTML = '<div class="muted">Koi provider configured nahi (ya models fetch nahi hue) — config.yaml + keys check karo, phir 🔄 Refresh dabao.</div>';
}

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;'); }

function toggleExposed(provider, model, on) {
  const sel = exposedSel[provider] || (exposedSel[provider] = []);
  if (on) { if (!sel.includes(model)) sel.push(model); }
  else exposedSel[provider] = sel.filter(m => m !== model);
  renderExposed();
}

function exposedAll(provider, on) {
  exposedSel[provider] = on ? exposedModelOptions(provider) : [];
  renderExposed();
}

async function saveExposed() {
  const st = document.getElementById('exposed-status');
  const btn = [...document.querySelectorAll('button')].find(b => b.textContent.includes('Save Exposed'));
  if (btn) btn.disabled = true;
  if (st) st.textContent = '⏳ saving…';
  const managed = exposedData.managed || {};
  const payload = {
    provider_models: exposedSel,
    provider_order: managed.provider_order || [],
    groups: managed.groups || [],
  };
  const { res, data } = await api('/admin/models', { method: 'PUT', body: JSON.stringify(payload) });
  if (btn) btn.disabled = false;
  if (!res.ok) { document.getElementById('exposed-err').textContent = data.detail || 'Save failed'; if (st) st.textContent = ''; return; }
  document.getElementById('exposed-err').textContent = '';
  if (st) st.textContent = '✅ ' + (data.message || 'Saved');
  await loadExposed();   // configured_models ab naye selection ke hisaab se
  loadModels();          // chat tab ka model list bhi update
}

async function refreshExposedLive() {
  const st = document.getElementById('exposed-status');
  if (st) st.textContent = '⏳ refreshing…';
  const { res, data } = await api('/admin/providers/refresh-all', { method: 'POST' });
  if (!res.ok) { document.getElementById('exposed-err').textContent = data.detail || 'Refresh failed'; if (st) st.textContent = ''; return; }
  const lines = (data.results || []).map(r => `${r.name}: ${r.count}${r.error ? ' (❌)' : ''}`).join(' · ');
  if (st) st.textContent = '✅ ' + lines;
  await loadExposed();
}

// ---------- providers admin (base_url + keys UI se) ----------
let providersData = null;
async function loadProvidersAdmin() {
  const box = document.getElementById('providers-list');
  box.innerHTML = '<span class="muted">Loading…</span>';
  document.getElementById('providers-err').textContent = '';
  document.getElementById('providers-ok').textContent = '';
  const { res, data } = await api('/admin/providers');
  if (!res.ok) { box.innerHTML = '<span class="err">' + (data.detail || 'Load failed') + '</span>'; return; }
  providersData = data;
  renderProviders();
}
function renderProviders() {
  const box = document.getElementById('providers-list');
  const list = (providersData && providersData.providers) || [];
  if (!list.length) { box.innerHTML = '<div class="muted">Koi provider nahi — "➕ Add Provider" se ek add karo.</div>'; return; }
  box.innerHTML = list.map((p, idx) => {
    const isCustom = p.source === 'custom';
    const keys = p.keys || [];
    const selectedCount = keys.filter(k => k.selected !== false).length;
    const keyRows = keys.map((k, ki) => {
      const on = k.selected !== false;
      return `
      <div style="margin-bottom:8px">
        <label class="chip ${on ? 'checked' : ''}" style="cursor:pointer;width:100%;box-sizing:border-box;justify-content:flex-start">
          <input type="checkbox" id="prov-ksel-${idx}-${ki}" ${on ? 'checked' : ''} onchange="toggleProviderKey(${idx})">
          🔑&nbsp;<code style="background:transparent;font-size:12px">${k.preview}</code>
          <span class="muted" style="font-size:11px;margin-left:auto">${on ? '✅ is base_url pe use ho rahi' : '⏸ use nahi ho rahi'}</span>
        </label>
        <input id="prov-kbase-${idx}-${ki}" type="text" value="${(k.base_url || '').replace(/"/g, '&quot;')}" placeholder="is key ka apna base_url (optional — warna upar wala base_url)" style="width:100%;box-sizing:border-box;margin-top:4px;padding:6px 8px;font-size:12px">
      </div>`;
    }).join('');
    return `
    <div class="provider-card" style="border:1px solid var(--panel2);border-radius:10px;padding:14px;margin-bottom:12px;background:var(--bg)">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px">
        <b>${p.name}</b>
        <span class="badge" style="background:var(--panel2);padding:2px 8px;border-radius:10px;font-size:12px">${p.type}</span>
        ${isCustom ? '<span class="badge" style="background:#ffd70022;color:#e8c34c;padding:2px 8px;border-radius:10px;font-size:12px">custom</span>' : '<span class="badge" style="background:var(--panel2);padding:2px 8px;border-radius:10px;font-size:12px">config</span>'}
        <span class="muted" style="font-size:12px">${selectedCount}/${keys.length} key${keys.length === 1 ? '' : 's'} in use</span>
        ${p.enabled ? '' : '<span class="badge" style="background:#f4433622;color:#f66;padding:2px 8px;border-radius:10px;font-size:12px">disabled</span>'}
      </div>

      <label class="muted" style="font-size:12px">Base URL <span style="font-size:11px">(selected keys sab isi ek URL pe rotate hongi)</span>
        <input id="prov-base-${idx}" type="text" value="${(p.base_url || '').replace(/"/g, '&quot;')}" placeholder="empty = default" style="width:100%;margin-top:4px">
      </label>

      <div class="muted" style="font-size:12px;margin-top:10px;display:flex;align-items:center;gap:8px;flex-wrap:wrap">
        <span>Models: ${(p.models && p.models.length) ? p.models.length + ' exposed' : 'abhi koi expose nahi'}</span>
        <button class="btn sec" style="padding:3px 10px;font-size:12px" onclick="showView('exposed'); loadExposed();">🎚 Manage Models</button>
      </div>

      <div style="margin-top:12px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
        <b style="font-size:13px">🔑 API Keys — is base_url pe rotate karne ke liye select karo</b>
        <button class="btn sec" style="padding:3px 10px;font-size:12px" onclick="selectAllProviderKeys(${idx})">✅ Select all</button>
      </div>
      <div style="margin-top:6px">${keyRows || '<span class="muted">Koi key nahi — 🔁 Sync ya Add Provider se env se pull karo.</span>'}</div>

      <div style="margin-top:10px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
        <span class="muted" style="font-size:12px"><input id="prov-enabled-${idx}" type="checkbox" ${p.enabled ? 'checked' : ''}> enabled</span>
        <button class="btn" onclick="saveProviderAdmin(${idx})">💾 Save</button>
        <button class="btn sec" onclick="resyncProviderKeys(${JSON.stringify(p.name).replace(/"/g, '&quot;')})" title="${p.name.toUpperCase()}_KEYS env var se dobara keys pull karo">🔁 Sync keys from env</button>
        ${isCustom ? `<button class="btn sec" onclick="deleteProviderAdmin(${JSON.stringify(p.name).replace(/"/g, '&quot;')})">🗑 Delete</button>` : ''}
      </div>
    </div>`;
  }).join('');
}
async function toggleProviderKey(idx) {
  // key ko check/uncheck karte hi turant save — alag se "Save" dabane ki zaroorat nahi.
  await saveProviderAdmin(idx);
}
async function selectAllProviderKeys(idx) {
  const p = providersData.providers[idx];
  (p.keys || []).forEach((k, ki) => {
    const cb = document.getElementById('prov-ksel-' + idx + '-' + ki);
    if (cb) cb.checked = true;
  });
  await saveProviderAdmin(idx);
}
async function saveProviderAdmin(idx) {
  const p = providersData.providers[idx];
  document.getElementById('providers-err').textContent = '';
  document.getElementById('providers-ok').textContent = '';
  const keys = (p.keys || []);
  const key_base_urls = {};
  const selected_keys = [];
  keys.forEach((k, ki) => {
    const v = document.getElementById('prov-kbase-' + idx + '-' + ki);
    const cb = document.getElementById('prov-ksel-' + idx + '-' + ki);
    if (v && v.value.trim()) key_base_urls[String(ki)] = v.value.trim();
    if (cb && cb.checked) selected_keys.push(ki);
  });
  const body = {
    name: p.name,
    type: p.type,
    base_url: document.getElementById('prov-base-' + idx).value.trim(),
    enabled: document.getElementById('prov-enabled-' + idx).checked,
    key_base_urls: key_base_urls,
    selected_keys: selected_keys,
  };
  const { res, data } = await api('/admin/providers', { method: 'POST', body: JSON.stringify(body) });
  if (!res.ok) { document.getElementById('providers-err').textContent = data.detail || 'Save failed'; return; }
  const usedCount = selected_keys.length || keys.length;
  document.getElementById('providers-ok').textContent = data.message + ' · ' + usedCount + '/' + keys.length + ' keys in use';
  await loadProvidersAdmin();
}
async function deleteProviderAdmin(name) {
  if (!confirm('Delete provider "' + name + '"?')) return;
  document.getElementById('providers-err').textContent = '';
  document.getElementById('providers-ok').textContent = '';
  const { res, data } = await api('/admin/providers/' + encodeURIComponent(name), { method: 'DELETE' });
  if (!res.ok) { document.getElementById('providers-err').textContent = data.detail || 'Delete failed'; return; }
  document.getElementById('providers-ok').textContent = data.message;
  await loadProvidersAdmin();
}
async function refreshProvidersLive() {
  const { res, data } = await api('/admin/providers/refresh-all', { method: 'POST' });
  if (!res.ok) { document.getElementById('providers-err').textContent = data.detail || 'Refresh failed'; return; }
  document.getElementById('providers-ok').textContent = '✅ Live models refresh ho gaye';
  await loadProvidersAdmin();
}
async function resyncProviderKeys(name) {
  document.getElementById('providers-err').textContent = '';
  document.getElementById('providers-ok').textContent = '';
  const { res, data } = await api('/admin/providers/' + encodeURIComponent(name) + '/resync-keys', { method: 'POST' });
  if (!res.ok) { document.getElementById('providers-err').textContent = data.detail || 'Resync failed'; return; }
  document.getElementById('providers-ok').textContent = '✅ ' + data.message + ' · ' + data.key_count + ' keys';
  await loadProvidersAdmin();
}

// ---------- add provider modal (select provider → base_url + keys auto-detected, no typing) ----------
let apPresets = [];
let apSelected = null;      // { name, type, base_url, env_var, isCustom }
let apDetectedKeys = [];    // [{index, preview}] — abhi ke provider ki detected keys

function openAddProviderModal() {
  document.getElementById('addProviderModal').style.display = 'flex';
  document.getElementById('addProviderStep1').style.display = '';
  document.getElementById('addProviderStep2').style.display = 'none';
  loadProviderCatalogPicker();
}
function closeAddProviderModal() {
  document.getElementById('addProviderModal').style.display = 'none';
}
function backToProviderPresets() {
  document.getElementById('addProviderStep1').style.display = '';
  document.getElementById('addProviderStep2').style.display = 'none';
}
async function loadProviderCatalogPicker() {
  const box = document.getElementById('addProviderPresets');
  box.innerHTML = '<span class="muted">Loading…</span>';
  const { res, data } = await api('/admin/providers/catalog');
  if (!res.ok) { box.innerHTML = '<span class="err">' + (data.detail || 'Load failed') + '</span>'; return; }
  apPresets = data.presets || [];
  box.innerHTML = apPresets.map((p, i) => `
    <div class="pick-item" onclick="selectProviderPreset(${i})">
      <span style="font-size:16px">${p.icon}</span>
      <span class="id">${p.label} <span class="muted">(${p.env_var})</span></span>
      ${p.already_added ? '<span class="tag">already added</span>' : (p.env_key_count > 0 ? `<span class="tag" style="color:var(--accent2)">${p.env_key_count} key${p.env_key_count === 1 ? '' : 's'} detected</span>` : '<span class="tag">no keys yet</span>')}
    </div>`).join('') + `
    <div class="pick-item" onclick="selectCustomProviderPreset()">
      <span style="font-size:16px">🧩</span>
      <span class="id">Custom / Other provider <span class="muted">(OpenAI-compatible gateway)</span></span>
    </div>`;
}
function selectProviderPreset(i) {
  const p = apPresets[i];
  apSelected = { name: p.name, type: p.type, base_url: p.base_url, env_var: p.env_var, isCustom: false };
  document.getElementById('ap-custom-name-wrap').style.display = 'none';
  document.getElementById('ap-name-label').textContent = p.icon + ' ' + p.label;
  document.getElementById('ap-type-badge').textContent = p.type;
  document.getElementById('ap-base-url').value = p.base_url;
  document.getElementById('ap-test-result').textContent = '';
  refreshDetectedKeys(p.name, p.env_var);
  document.getElementById('addProviderStep1').style.display = 'none';
  document.getElementById('addProviderStep2').style.display = '';
}
function selectCustomProviderPreset() {
  apSelected = { name: '', type: 'openai', base_url: '', env_var: '', isCustom: true };
  document.getElementById('ap-custom-name-wrap').style.display = '';
  document.getElementById('ap-custom-name').value = '';
  document.getElementById('ap-custom-type').value = 'openai';
  document.getElementById('ap-name-label').textContent = '🧩 Custom Provider';
  document.getElementById('ap-type-badge').textContent = 'openai';
  document.getElementById('ap-base-url').value = '';
  document.getElementById('ap-custom-envname').textContent = '—';
  document.getElementById('ap-test-result').textContent = '';
  renderApKeyList([]);
  document.getElementById('ap-keys-status').textContent = '';
  document.getElementById('ap-keys-err').textContent = 'Provider ka naam daalo — keys uske env var se auto-detect hongi.';
  document.getElementById('addProviderStep1').style.display = 'none';
  document.getElementById('addProviderStep2').style.display = '';
}
function onCustomProviderTypeChange() {
  document.getElementById('ap-type-badge').textContent = document.getElementById('ap-custom-type').value;
}
let apCustomNameDebounce = null;
function onCustomProviderNameInput() {
  const name = document.getElementById('ap-custom-name').value.trim();
  document.getElementById('ap-custom-envname').textContent = name ? (name.toUpperCase() + '_KEYS') : '—';
  clearTimeout(apCustomNameDebounce);
  if (!name) {
    document.getElementById('ap-keys-status').textContent = '';
    document.getElementById('ap-keys-err').textContent = 'Provider ka naam daalo — keys uske env var se auto-detect hongi.';
    renderApKeyList([]);
    return;
  }
  apCustomNameDebounce = setTimeout(() => refreshDetectedKeys(name, name.toUpperCase() + '_KEYS'), 350);
}
async function refreshDetectedKeys(name, envVar) {
  const statusEl = document.getElementById('ap-keys-status');
  const errEl = document.getElementById('ap-keys-err');
  statusEl.textContent = '🔍 checking ' + envVar + '…';
  errEl.textContent = '';
  renderApKeyList([]);
  const { res, data } = await api('/admin/providers/detect-keys?name=' + encodeURIComponent(name));
  if (!res.ok) { statusEl.textContent = ''; errEl.textContent = data.detail || 'Key detection failed'; return; }
  if (data.key_count > 0) {
    statusEl.textContent = `✅ ${data.key_count} key${data.key_count === 1 ? '' : 's'} detected from ${data.env_var} — jo use karni hain unhi ko select rehne do (default: sab).`;
    renderApKeyList(data.keys);
  } else {
    statusEl.textContent = '';
    errEl.textContent = `⚠️ ${data.env_var} me koi key nahi mili. Host secrets me "${data.env_var}=key1,key2,..." add karke (multiple keys ek saath) dobara try karo.`;
  }
}
function renderApKeyList(keys) {
  apDetectedKeys = keys || [];
  const box = document.getElementById('ap-keys-list');
  box.innerHTML = apDetectedKeys.map(k => `
    <label class="chip checked" style="width:100%;box-sizing:border-box;justify-content:flex-start;margin-bottom:6px;cursor:pointer">
      <input type="checkbox" class="ap-key-cb" value="${k.index}" checked onchange="this.parentElement.classList.toggle('checked', this.checked)">
      API Key ${k.index + 1}: <code style="background:transparent;font-size:12px">${k.preview}</code>
    </label>`).join('');
}
function apCurrentNameType() {
  const name = apSelected.isCustom ? document.getElementById('ap-custom-name').value.trim() : apSelected.name;
  const type = apSelected.isCustom ? document.getElementById('ap-custom-type').value : apSelected.type;
  return { name, type };
}
async function testAddProviderBaseUrl() {
  const errEl = document.getElementById('ap-keys-err');
  const resultEl = document.getElementById('ap-test-result');
  errEl.textContent = '';
  const { name, type } = apCurrentNameType();
  const base_url = document.getElementById('ap-base-url').value.trim();
  if (!name) { errEl.textContent = 'Provider name required'; return; }
  const key_indices = Array.from(document.querySelectorAll('.ap-key-cb:checked')).map(i => parseInt(i.value, 10));
  resultEl.className = 'muted';
  resultEl.style.fontSize = '12px';
  resultEl.textContent = '🧪 testing…';
  const { res, data } = await api('/admin/providers/test-url', { method: 'POST', body: JSON.stringify({ name, type, base_url, key_indices }) });
  if (!res.ok) { resultEl.textContent = ''; errEl.textContent = data.detail || 'Test failed'; return; }
  resultEl.className = data.ok ? 'ok' : 'err';
  resultEl.textContent = data.message;
}
async function submitNewProvider() {
  document.getElementById('ap-keys-err').textContent = '';
  const { name, type } = apCurrentNameType();
  const base_url = document.getElementById('ap-base-url').value.trim();
  if (!name) { document.getElementById('ap-keys-err').textContent = 'Provider name required'; return; }
  const key_indices = Array.from(document.querySelectorAll('.ap-key-cb:checked')).map(i => parseInt(i.value, 10));
  if (apDetectedKeys.length && !key_indices.length) {
    document.getElementById('ap-keys-err').textContent = 'Kam se kam ek API key select karo';
    return;
  }
  const enabled = document.getElementById('ap-enabled').checked;
  const btn = document.getElementById('ap-submit-btn');
  btn.disabled = true; btn.textContent = 'Saving…';
  // NOTE: api_keys jaan-bujhke nahi bheja — backend seedha <NAME>_KEYS env var
  // se saari keys pull karta hai; selected_keys sirf batata hai unme se kaunsi
  // ACTUALLY is base_url pe use karni hain. Koi key kabhi type/paste nahi hoti.
  const { res, data } = await api('/admin/providers', {
    method: 'POST',
    body: JSON.stringify({ name, type, base_url, enabled, selected_keys: key_indices }),
  });
  btn.disabled = false; btn.textContent = '💾 Save';
  if (!res.ok) { document.getElementById('ap-keys-err').textContent = data.detail || 'Save failed'; return; }
  const usedCount = key_indices.length || data.key_count;
  document.getElementById('providers-ok').textContent = '✅ ' + data.message + (data.key_count !== undefined ? ' · ' + usedCount + '/' + data.key_count + ' keys in use' : '');
  closeAddProviderModal();
  await loadProvidersAdmin();
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
