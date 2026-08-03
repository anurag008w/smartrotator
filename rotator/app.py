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

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import db as database
from .auth import (
    create_jwt,
    decode_jwt,
    generate_api_key,
    hash_password,
    is_user_api_key,
    verify_password,
)
from .providers import (
    AllProvidersExhausted,
    ChatMessage,
    ImageInput,
    ProviderError,
    RateLimitError,
)
from .router import Rotator

logger = logging.getLogger("smartrotator")

CONFIG_PATH = os.environ.get("ROTATOR_CONFIG", "config.yaml")


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# --------------------------------------------------------------------------
# App lifecycle
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    rotator = Rotator(config_path=CONFIG_PATH)
    app.state.rotator = rotator
    await database.init_db()
    yield
    await rotator.aclose()
    await database.engine.dispose()


app = FastAPI(
    title="SmartRotator",
    description="Multi-provider LLM gateway with per-user quotas + dashboard.",
    version="0.3.0",
    lifespan=lifespan,
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
    max_tokens: int = 4096
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


class SetLimitRequest(BaseModel):
    daily_limit: int = Field(..., ge=1, le=100000)


# --------------------------------------------------------------------------
# Public endpoints
# --------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(request: Request):
    rotator: Rotator = request.app.state.rotator
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "owned_by": m["provider"], "type": m["type"]}
            for m in rotator.models()
        ],
    }


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

    async with database.get_session() as db:
        existing = await database.get_user_by_username(db, req.username)
        if existing:
            raise HTTPException(status_code=409, detail="Username already taken")

        # first user = admin (self-hosted pattern)
        count = await _count_users(db)
        role = "admin" if count == 0 else "user"
        if req.username in settings["admin_usernames"]:
            role = "admin"

        password_hash, salt = hash_password(req.password)
        user = await database.create_user(
            db,
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
        return {"token": token, "api_key": user.api_key, "user": _user_public(user)}


@app.post("/auth/login")
async def login(req: LoginRequest):
    settings = _auth_settings()
    if not settings["enabled"]:
        raise HTTPException(status_code=403, detail="Auth disabled")

    async with database.get_session() as db:
        user = await database.get_user_by_username(db, req.username)
        if not user or not verify_password(req.password, user.password_hash, user.salt):
            raise HTTPException(status_code=401, detail="Invalid username or password")

        token = create_jwt(
            user.id, user.username, user.role, settings["jwt_secret"], settings["jwt_hours"]
        )
        return {"token": token, "api_key": user.api_key, "user": _user_public(user)}


@app.get("/auth/me")
async def auth_me(request: Request):
    settings = _auth_settings()
    user = await _require_user(request, settings)
    async with database.get_session() as db:
        today = database.today_utc()
        row = await database.get_usage_row(db, user.id, today)
        history = await database.get_usage_between(db, user.id, 7)
        return {
            "user": _user_public(user),
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
    async with database.get_session() as db:
        # naye session me fresh user fetch karo (purana detached ho sakta hai)
        db_user = await database.get_user_by_id(db, user.id)
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        db_user.api_key = generate_api_key()
        await db.commit()
        await db.refresh(db_user)
        return {"api_key": db_user.api_key}


# --------------------------------------------------------------------------
# Admin endpoints
# --------------------------------------------------------------------------
@app.get("/admin/users")
async def admin_users(request: Request):
    settings = _auth_settings()
    user = await _require_user(request, settings)
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")

    async with database.get_session() as db:
        result = await db.execute(database.select(database.User).order_by(database.User.id))
        users = list(result.scalars())
        today = database.today_utc()
        out = []
        for u in users:
            row = await database.get_usage_row(db, u.id, today)
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
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")

    async with database.get_session() as db:
        target = await database.get_user_by_id(db, user_id)
        if not target:
            raise HTTPException(status_code=404, detail="User not found")
        target.daily_limit = req.daily_limit
        await db.commit()
        return {"ok": True, "username": target.username, "daily_limit": target.daily_limit}


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
                    f"Aapka daily quota khatam ho gaya ({user.daily_limit} requests/day). "
                    "Kal reset hoga, ya admin se badhwa lo."
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
                "Thodi der baad try karo, ya apna configured provider use karo."
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

    async with database.get_session() as db:
        if is_user_api_key(token):
            return await database.get_user_by_api_key(db, token)
        payload = decode_jwt(token, settings["jwt_secret"])
        if payload:
            return await database.get_user_by_id(db, int(payload["sub"]))
        return None


async def _require_user(request: Request, settings: dict):
    user = await _authenticate(request, settings)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


async def _count_users(db) -> int:
    result = await db.execute(database.select(database.User.id))
    return len(result.all())


async def _reserve_quota(user) -> bool:
    """Request reserve karo — quota bacha hai toh True."""
    async with database.get_session() as db:
        row = await database.get_usage_row(db, user.id, database.today_utc())
        if row.requests >= user.daily_limit:
            return False
        row.requests += 1
        await db.commit()
        return True


async def _refund_quota(user) -> None:
    """Fail ho gaya toh reserved request wapas de do."""
    async with database.get_session() as db:
        row = await database.get_usage_row(db, user.id, database.today_utc())
        if row.requests > 0:
            row.requests -= 1
            await db.commit()


async def _record_tokens(user, tokens: int) -> None:
    async with database.get_session() as db:
        row = await database.get_usage_row(db, user.id, database.today_utc())
        row.tokens += tokens
        await db.commit()


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
        text = "\n".join(t for t in text_parts if t)
    else:
        text = ""

    return ChatMessage(
        role=role,
        content=text,
        images=images,
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
      <div class="tab" id="adminTab" style="display:none" onclick="showView('admin'); loadAdmin();">🛡 Admin</div>
    </div>

    <!-- CHAT -->
    <div class="view active" id="view-chat">
      <div class="card">
        <div class="muted" style="margin-bottom:10px">Models (multi-select — inhi me rotation hogi):</div>
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
  </div>
</div>

<script>
let token = localStorage.getItem('sr_token') || '';
let images = [];
let models = [];

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
function showMain(user) {
  document.getElementById('view-auth').style.display = 'none';
  document.getElementById('view-main').style.display = 'block';
  document.getElementById('userChip').textContent = '👤 ' + user.username + (user.role === 'admin' ? ' ⭐ admin' : '');
  if (user.role === 'admin') document.getElementById('adminTab').style.display = '';
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
  showMain(data.user); showView('chat'); loadModels(); loadMe();
}
function logout() { localStorage.removeItem('sr_token'); token = ''; location.reload(); }

// ---------- init ----------
(async function init() {
  if (token) {
    const { res, data } = await api('/auth/me');
    if (res.ok) {
      showMain(data.user);
      document.getElementById('api-key').value = data.api_key || '';
      updateCodeSample();
      loadModels(); loadMe();
    } else showAuth();
  } else showAuth();
})();

// ---------- chat ----------
async function loadModels() {
  const { res, data } = await api('/v1/models');
  if (!res.ok) return;
  models = data.data;
  const list = document.getElementById('modelList');
  list.innerHTML = '';
  models.forEach(m => {
    const chip = document.createElement('label');
    chip.className = 'chip checked';
    chip.innerHTML = `<input type="checkbox" class="m-cb" value="${m.id}" checked>${m.id}<span class="tag">${m.owned_by}</span>`;
    list.appendChild(chip);
  });
}
function selectedModels() { return Array.from(document.querySelectorAll('.m-cb:checked')).map(c => c.value); }
async function send() {
  const text = document.getElementById('prompt').value.trim();
  if (!text && images.length === 0) return;
  addMsg('user', text || '(image only)', images);
  const content = [];
  images.forEach(i => content.push({ type: 'image_url', image_url: { url: i.dataUrl } }));
  if (text) content.push({ type: 'text', text });
  document.getElementById('sendBtn').disabled = true;
  const { res, data } = await api('/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({ models: selectedModels(), messages: [{ role: 'user', content }] })
  });
  if (!res.ok) addMsg('ai', '⚠️ ' + (data.detail || ('Error ' + res.status)));
  else addMsg('ai', data.choices[0].message.content, [], `⚡ ${data.provider} · ${data.model} · ${data.key}`);
  document.getElementById('sendBtn').disabled = false;
  document.getElementById('prompt').value = '';
  images = []; document.getElementById('thumbs').innerHTML = '';
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
      <td class="${u.role === 'admin' ? 'badge-admin' : ''}">${u.role}</td>
      <td class="muted">${u.api_key}</td>
      <td>${u.today_requests}</td>
      <td>${u.daily_limit}</td>
      <td><input type="number" value="${u.daily_limit}" min="1" style="width:80px;padding:6px" onchange="setLimit(${u.id}, this.value)"></td>`;
    tb.appendChild(tr);
  });
}
async function setLimit(id, value) {
  const { res, data } = await api('/admin/users/' + id + '/limit', { method: 'POST', body: JSON.stringify({ daily_limit: parseInt(value) }) });
  if (!res.ok) alert(data.detail || 'Failed');
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
