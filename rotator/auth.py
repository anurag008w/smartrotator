"""
auth.py — Password hashing, JWT tokens, user API keys.

- Passwords: PBKDF2-HMAC-SHA256 (per-user salt, 200k iterations)
- Sessions (browser/dashboard): JWT HS256
- App integration: har user ke paas ek `sk-...` API key hoti hai
  jise OpenAI SDK me api_key ki tarah use karo
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

ITERATIONS = 200_000


# --------------------------------------------------------------------------
# Password hashing
# --------------------------------------------------------------------------
def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Returns (hash_hex, salt_hex)."""
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), ITERATIONS
    )
    return digest.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), ITERATIONS
    )
    return hmac.compare_digest(digest.hex(), stored_hash)


# --------------------------------------------------------------------------
# User API keys (apps ke liye — OpenAI SDK compatible)
# --------------------------------------------------------------------------
def generate_api_key() -> str:
    return "sk-" + secrets.token_urlsafe(32)


def is_user_api_key(token: str) -> bool:
    return token.startswith("sk-")


# --------------------------------------------------------------------------
# JWT (dashboard sessions)
# --------------------------------------------------------------------------
def create_jwt(user_id: int, username: str, role: str, secret: str, hours: int = 24) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=hours),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_jwt(token: str, secret: str) -> Optional[dict]:
    try:
        return jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
