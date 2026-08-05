"""
auth.py — Password hashing, JWT tokens, user API keys.

- Passwords: scrypt (memory-hard, GPU/ASIC resistant) — purane PBKDF2
  hashes bhi verify hote hain, login pe automatically scrypt pe migrate
- Sessions (browser/dashboard): JWT HS256
- App integration: har user ke paas ek `sk-...` API key hoti hai
  jise OpenAI SDK me api_key ki tarah use karo

Speed note: PBKDF2-SHA256 200k iterations ~370ms leta tha (Render free
tier pe aur bhi slow). scrypt n=2^14 ~150ms — 2.5x fast + zyada secure.
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

ITERATIONS = 200_000  # legacy PBKDF2 (purane users ke liye verify)

# scrypt params (memory-hard: 128 * r * n bytes ≈ 16 MB)
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_PREFIX = "$scrypt$"


# --------------------------------------------------------------------------
# Password hashing
# --------------------------------------------------------------------------
def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Returns (hash_str, salt_hex). Naye hashes scrypt hote hain.

    Hash string me prefix encode hota hai (`$scrypt$...`), taaki verify
    walay pata kar sake kaunsa algorithm use karna hai. Purane PBKDF2
    hashes (bina prefix) legacy path se verify hote hain.
    """
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=bytes.fromhex(salt),
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=32,
    )
    return SCRYPT_PREFIX + digest.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """scrypt (naya) ya legacy PBKDF2 (purana) — dono verify karta hai."""
    if stored_hash.startswith(SCRYPT_PREFIX):
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt),
            n=SCRYPT_N,
            r=SCRYPT_R,
            p=SCRYPT_P,
            dklen=32,
        )
        return hmac.compare_digest(SCRYPT_PREFIX + digest.hex(), stored_hash)
    # legacy PBKDF2-HMAC-SHA256
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), ITERATIONS
    )
    return hmac.compare_digest(digest.hex(), stored_hash)


def is_scrypt_hash(stored_hash: str) -> bool:
    """Kya ye hash naya scrypt format hai? (lazy migration ke liye)"""
    return stored_hash.startswith(SCRYPT_PREFIX)


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
