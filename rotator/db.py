"""
db.py — Database layer (async SQLAlchemy).

Users + daily usage track karta hai. Local me SQLite, Render pe
PostgreSQL (DATABASE_URL env se). Render free Postgres 30 din ke
baad expire hota hai — production ke liye $7/mo instance
recommended hai (README dekho).

DATABASE_URL examples:
  sqlite+aiosqlite:///./rotator.db        (local dev)
  postgresql+asyncpg://user:pass@host/db  (Render Postgres)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import BigInteger, Integer, String, UniqueConstraint, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./rotator.db")
# Render ka Postgres URL "postgres://" se start hota hai — async driver chahiye
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    salt: Mapped[str] = mapped_column(String(32))
    api_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    role: Mapped[str] = mapped_column(String(16), default="user")  # user | admin
    daily_limit: Mapped[int] = mapped_column(Integer, default=50)  # requests/day
    created_at: Mapped[str] = mapped_column(String(32), default=lambda: _now_utc())

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class Usage(Base):
    __tablename__ = "usage"
    __table_args__ = (UniqueConstraint("user_id", "day", name="uq_user_day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    day: Mapped[str] = mapped_column(String(10), index=True)  # YYYY-MM-DD
    requests: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(BigInteger, default=0)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def get_session() -> AsyncSession:
    """Ek naya session de deta hai (sessionmaker sync hota hai)."""
    return SessionLocal()


# --------------------------------------------------------------------------
# User helpers
# --------------------------------------------------------------------------
async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def get_user_by_api_key(db: AsyncSession, api_key: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.api_key == api_key))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def create_user(
    db: AsyncSession,
    username: str,
    password_hash: str,
    salt: str,
    api_key: str,
    daily_limit: int = 50,
    role: str = "user",
) -> User:
    user = User(
        username=username,
        password_hash=password_hash,
        salt=salt,
        api_key=api_key,
        daily_limit=daily_limit,
        role=role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


# --------------------------------------------------------------------------
# Usage helpers (per-day quota)
# --------------------------------------------------------------------------
async def get_usage_row(db: AsyncSession, user_id: int, day: str) -> Usage:
    """Today's usage row — nahi hai toh bana ke de deta hai."""
    result = await db.execute(
        select(Usage).where(Usage.user_id == user_id, Usage.day == day)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = Usage(user_id=user_id, day=day, requests=0, tokens=0)
        db.add(row)
        await db.commit()
        await db.refresh(row)
    return row


async def get_usage_between(db: AsyncSession, user_id: int, days: int = 7) -> list[Usage]:
    """Last N din ka usage (chart ke liye)."""
    result = await db.execute(
        select(Usage)
        .where(Usage.user_id == user_id, Usage.day >= _days_ago(days))
        .order_by(Usage.day)
    )
    return list(result.scalars())


def _days_ago(n: int) -> str:
    from datetime import timedelta

    return (datetime.now(timezone.utc) - timedelta(days=n - 1)).strftime("%Y-%m-%d")
