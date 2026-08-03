"""
core.py — SmartRotator core primitives.

Round-robin pools, per-key quota tracking, cooldown management
and health-based selection. Yeh engine ka dil hai — yahin se
rotation ki saari magic aati hai.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Generic, Iterator, Optional, TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------
# Quota / limit tracking for a single key
# --------------------------------------------------------------------------
@dataclass
class KeyState:
    """Runtime state for one API key."""

    key: str
    label: str

    # rolling window tracking (requests made in last 60s)
    request_times: list[float] = field(default_factory=list)

    # daily counter (resets at midnight UTC)
    daily_count: int = 0
    daily_day: str = ""

    consecutive_failures: int = 0
    total_success: int = 0
    total_fail: int = 0

    # cooldown bookkeeping
    cooldown_until: float = 0.0
    last_used_at: float = 0.0

    def is_in_cooldown(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return now < self.cooldown_until

    def mark_success(self) -> None:
        self.consecutive_failures = 0
        self.total_success += 1

    def mark_failure(self, cooldown_seconds: float, ban_after: int) -> None:
        self.consecutive_failures += 1
        self.total_fail += 1
        if self.consecutive_failures >= ban_after:
            # temporary ban — longer cooldown
            self.cooldown_until = time.time() + max(cooldown_seconds * 5, 300)
        else:
            self.cooldown_until = time.time() + cooldown_seconds

    def record_request(self, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        self.request_times.append(now)
        # only keep the last 120s of timestamps
        cutoff = now - 120
        self.request_times = [t for t in self.request_times if t > cutoff]
        self.last_used_at = now

    def roll_daily_if_needed(self, today: str) -> None:
        if self.daily_day != today:
            self.daily_day = today
            self.daily_count = 0

    def requests_in_window(self, window_seconds: int = 60, now: float | None = None) -> int:
        now = now if now is not None else time.time()
        cutoff = now - window_seconds
        return sum(1 for t in self.request_times if t > cutoff)


# --------------------------------------------------------------------------
# Generic round-robin pool
# --------------------------------------------------------------------------
class RoundRobinPool(Generic[T]):
    """
    Thread-safe round-robin pool over items of type T.

    Strategies:
      - round_robin : pure cycling
      - random      : shuffle every cycle
      - health      : prefer items with fewer failures (weighted)
    """

    def __init__(self, items: list[T], strategy: str = "round_robin", rng=None):
        self._items: list[T] = items
        self._strategy = strategy
        self._index = 0
        self._lock = threading.Lock()
        self._rng = rng
        if self._rng is None:
            import random

            self._rng = random

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> list[T]:
        return self._items

    def next(self) -> Optional[T]:
        """Return the next item, or None if the pool is empty."""
        if not self._items:
            return None
        with self._lock:
            if self._strategy == "random":
                return self._rng.choice(self._items)
            if self._strategy == "health":
                return self._pick_healthiest()
            item = self._items[self._index % len(self._items)]
            self._index += 1
            return item

    def _pick_healthiest(self) -> Optional[T]:
        """Weighted pick favouring items with fewest failures."""
        if not self._items:
            return None
        # subclasses can override scoring; default = uniform random
        return self._rng.choice(self._items)


# --------------------------------------------------------------------------
# KeyRing — pool of API keys + models for one provider
# --------------------------------------------------------------------------
class KeyRing(RoundRobinPool[KeyState]):
    """
    Manages keys AND models of a single provider with the rotation order:

        Key 1 → model 1, model 2, ...   (models rotate FIRST)
        Key 2 → model 1, model 2, ...
        ...

    So `pick()` returns a (KeyState, model) pair. Jab kisi key ke sab
    (key, model) pairs fail ho jayein, tab poori key ko cooldown de dete
    hain. Daily/RPM limits key-level pe track hote hain.
    """

    def __init__(
        self,
        keys: list[str],
        models: list[str],
        label: str,
        strategy: str = "round_robin",
        cooldown_seconds: float = 60,
        ban_after: int = 3,
        rpm_limit: int = 0,          # 0 = unknown / not enforced
        rpd_limit: int = 0,          # 0 = unknown / not enforced
    ):
        states = [
            KeyState(key=k, label=f"{label}[{i}]")
            for i, k in enumerate(keys)
        ]
        super().__init__(states, strategy=strategy)
        self._label = label
        self._models = list(models)
        self._cooldown_seconds = cooldown_seconds
        self._ban_after = ban_after
        self._rpm_limit = rpm_limit
        self._rpd_limit = rpd_limit
        self._rng = None
        import random

        self._rng = random

        # per (key, model) pair failure / cooldown bookkeeping
        self._pair_failures: dict[tuple[str, str], int] = {}
        self._pair_cooldown: dict[tuple[str, str], float] = {}

    # -- public API --------------------------------------------------------
    @property
    def models(self) -> list[str]:
        return list(self._models)

    def _pairs(self, models: list[str] | None) -> list[tuple[KeyState, str]]:
        """Key-major, model-minor ordered (key, model) pairs."""
        active = models if models is not None else self._models
        return [(s, m) for s in self._items for m in active]

    def _scan(
        self, models: list[str] | None, now: float
    ) -> Optional[tuple[int, KeyState, str]]:
        """Pehla available (key, model) pair dhoondo — index MUTATE NAHI hota.

        Return: (idx_in_pairs, KeyState, model) ya None.
        has_available() isi ko use karta hai taaki check rotation ko
        consume na kare.
        """
        today = self._today_utc()
        pairs = self._pairs(models)
        if not pairs:
            return None

        start = self._index % len(pairs)
        for offset in range(len(pairs)):
            idx = (start + offset) % len(pairs)
            state, model = pairs[idx]
            state.roll_daily_if_needed(today)

            if state.is_in_cooldown(now):
                continue
            if self._rpd_limit and state.daily_count >= self._rpd_limit:
                continue
            if self._rpm_limit and state.requests_in_window(60, now) >= self._rpm_limit:
                continue
            if now < self._pair_cooldown.get((state.key, model), 0):
                continue

            return idx, state, model
        return None

    def pick(self, models: list[str] | None = None) -> Optional[tuple[KeyState, str]]:
        """
        Return the next available (KeyState, model) pair in rotation order.
        Skips: cooldown keys, exhausted keys, cooling (key, model) pairs.
        Returns None when everything is unavailable.
        """
        found = self._scan(models, time.time())
        if found is None:
            return None
        idx, state, model = found
        self._index = idx + 1
        return state, model

    def has_available(self, models: list[str] | None = None) -> bool:
        """Quick check: kya is provider me abhi koi (key, model) available hai?
        (rotation ko consume nahi karta)"""
        return self._scan(models, time.time()) is not None

    def report_success(self, state: KeyState, model: str | None = None) -> None:
        state.mark_success()
        if model is not None:
            self._pair_failures.pop((state.key, model), None)
            self._pair_cooldown.pop((state.key, model), None)

    def report_failure(self, state: KeyState, model: str | None = None) -> None:
        """
        Failure pe (key, model) pair ko chhota cooldown do.
        Agar wahi pair bar-bar fail ho (>= ban_after), toh poori key ko
        lamba cooldown de do — kyunki key hi kharab hai.
        """
        if model is not None:
            pair = (state.key, model)
            self._pair_failures[pair] = self._pair_failures.get(pair, 0) + 1
            if self._pair_failures[pair] >= self._ban_after:
                # whole key gets a long cooldown
                state.cooldown_until = time.time() + max(self._cooldown_seconds * 5, 300)
                for p in [p for p in self._pair_cooldown if p[0] == state.key]:
                    del self._pair_cooldown[p]
            else:
                self._pair_cooldown[pair] = time.time() + self._cooldown_seconds
        else:
            state.mark_failure(self._cooldown_seconds, self._ban_after)

    def record_used(self, state: KeyState) -> None:
        state.record_request()
        state.daily_count += 1

    def status(self) -> dict:
        today = self._today_utc()
        return {
            "label": self._label,
            "models": self._models,
            "total_keys": len(self._items),
            "keys": [
                {
                    "index": i,
                    "key_preview": self._preview(s.key),
                    "in_cooldown": s.is_in_cooldown(),
                    "consecutive_failures": s.consecutive_failures,
                    "success": s.total_success,
                    "fail": s.total_fail,
                    "daily_count": s.daily_count,
                    "rpm_last_60s": s.requests_in_window(60),
                    "daily_left": None
                    if not self._rpd_limit
                    else max(0, self._rpd_limit - s.daily_count),
                }
                for i, s in enumerate(self._items)
            ],
        }

    def keys_as_list(self) -> list[str]:
        return [s.key for s in self._items]

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _preview(key: str) -> str:
        if len(key) <= 10:
            return "***"
        return f"{key[:6]}...{key[-4:]}"

    @staticmethod
    def _today_utc() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())


# --------------------------------------------------------------------------
# ProxyPool — optional IP rotation
# --------------------------------------------------------------------------
class ProxyPool:
    """
    Round-robin pool of proxies (host:port or full URL).

    Keeps a quarantine list for dead proxies and re-validates them
    lazily so a temporarily-down proxy gets a second chance later.
    """

    def __init__(
        self,
        proxies: list[str],
        rotate_every_request: bool = True,
        max_attempts: int = 3,
        quarantine_failures: int = 3,
    ):
        self._all = list(dict.fromkeys(p.strip() for p in proxies if p.strip()))
        self._healthy: list[str] = list(self._all)
        self._quarantined: dict[str, float] = {}
        self._failures: dict[str, int] = {}
        self._lock = threading.Lock()
        self._index = 0
        self._rotate_every_request = rotate_every_request
        self._max_attempts = max_attempts
        self._quarantine_failures = quarantine_failures

    @property
    def enabled(self) -> bool:
        return bool(self._all)

    def next(self) -> Optional[str]:
        """Return the next healthy proxy, or None if none available."""
        with self._lock:
            now = time.time()
            # return quarantined proxies after 60s to give them a retry
            for proxy in list(self._quarantined):
                if now - self._quarantined[proxy] > 60:
                    self._healthy.append(proxy)
                    del self._quarantined[proxy]

            if not self._healthy:
                return None

            proxy = self._healthy[self._index % len(self._healthy)]
            self._index += 1
            if self._rotate_every_request:
                # shuffle the deck each pick to randomise rotation
                import random

                self._rng = getattr(self, "_rng", None) or random
                self._rng.shuffle(self._healthy)
            return proxy

    def report_failure(self, proxy: str) -> None:
        with self._lock:
            self._failures[proxy] = self._failures.get(proxy, 0) + 1
            if self._failures[proxy] >= self._quarantine_failures:
                if proxy in self._healthy:
                    self._healthy.remove(proxy)
                self._quarantined[proxy] = time.time()

    def report_success(self, proxy: str) -> None:
        with self._lock:
            self._failures[proxy] = 0
            if proxy in self._quarantined:
                del self._quarantined[proxy]
            if proxy not in self._healthy and proxy in self._all:
                self._healthy.append(proxy)

    def status(self) -> dict:
        return {
            "total": len(self._all),
            "healthy": len(self._healthy),
            "quarantined": len(self._quarantined),
            "rotate_every_request": self._rotate_every_request,
        }
