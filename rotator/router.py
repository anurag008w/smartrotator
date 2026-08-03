"""
router.py — SmartRotator engine.

Rotation order (user ke requirement ke hisaab se):

    Provider A
      ├─ Key 1 → model 1, model 2, model 3 ...   ← models pehle rotate
      ├─ Key 2 → model 1, model 2, model 3 ...
      └─ ... jab saare keys+models exhaust →
    Provider B ...

Fail (429/5xx/network) pe:
  - us (key, model) pair ko chhota cooldown
  - baar-baar fail ho toh poori key ko lamba cooldown
  - phir agla provider/key/model try

Sab async hai — FastAPI me seedha await karo, CLI me asyncio.run().
"""

from __future__ import annotations

import os
import random
import threading
from dataclasses import dataclass
from typing import Optional

import yaml

from .core import KeyRing, ProxyPool
from .providers import (
    AllProvidersExhausted,
    ChatMessage,
    ChatResult,
    Provider,
    ProviderError,
    RateLimitError,
    build_provider,
)


@dataclass
class ProviderConfig:
    name: str
    ptype: str
    models: list[str]
    keys: list[str]
    base_url: Optional[str] = None
    rpm_limit: int = 0
    rpd_limit: int = 0


class ProviderState:
    """One provider + its key/model ring + health score."""

    def __init__(self, cfg: ProviderConfig, ring: KeyRing, provider: Provider):
        self.cfg = cfg
        self.ring = ring
        self.provider = provider
        self.failures = 0
        self.successes = 0
        self.last_error: Optional[str] = None


class Rotator:
    def __init__(
        self,
        config_path: str = "config.yaml",
        env_prefix: str = "",
    ):
        self.config_path = config_path
        self.env_prefix = env_prefix
        self._lock = threading.Lock()
        self._rng = random.Random()

        self.providers: list[ProviderState] = []
        self._provider_cursor = 0          # sequential drain ke liye
        self.proxy_pool: Optional[ProxyPool] = None
        self.settings: dict = {}
        self.default_model: str = "gemini-2.5-flash"
        self.allow_model_routing: bool = True
        self.provider_strategy: str = "sequential"   # sequential | round_robin

        self.reload()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------
    def reload(self) -> None:
        """(Re)load config from YAML + env var key overrides."""
        with open(self.config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        self.settings = raw.get("rotation", {})
        strategy = self.settings.get("strategy", "round_robin")
        cooldown = float(self.settings.get("cooldown_seconds", 60))
        ban_after = int(self.settings.get("fail_after_attempts", 3))
        self.provider_strategy = self.settings.get("provider_strategy", "sequential")

        server = raw.get("server", {})
        self.default_model = server.get("default_model", "gemini-2.5-flash")
        self.allow_model_routing = bool(server.get("allow_model_routing", True))

        # build provider states
        new_states: list[ProviderState] = []
        for cfg in raw.get("providers", []):
            name = cfg.get("name", "provider")
            ptype = cfg.get("type", "openai")
            models = [m.strip() for m in cfg.get("models", []) if m.strip()]
            base_url = cfg.get("base_url")

            # env var override: <NAME>_KEYS="key1,key2,key3"
            env_name = f"{self.env_prefix}{name.upper()}_KEYS"
            env_keys = os.environ.get(env_name, "")
            if env_keys.strip():
                keys = [k.strip() for k in env_keys.split(",") if k.strip()]
            else:
                keys = [k.strip() for k in cfg.get("api_keys", []) if k.strip()]
                # skip placeholder keys
                keys = [k for k in keys if not k.startswith("PASTE_")]

            if not keys or not models:
                continue  # provider setup incomplete, skip silently

            pcfg = ProviderConfig(
                name=name,
                ptype=ptype,
                models=models,
                keys=keys,
                base_url=base_url,
                rpm_limit=int(cfg.get("rpm_limit", 0)),
                rpd_limit=int(cfg.get("rpd_limit", 0)),
            )
            ring = KeyRing(
                keys=pcfg.keys,
                models=pcfg.models,
                label=name,
                strategy=strategy,
                cooldown_seconds=cooldown,
                ban_after=ban_after,
                rpm_limit=pcfg.rpm_limit,
                rpd_limit=pcfg.rpd_limit,
            )
            provider = build_provider(name, ptype, base_url, models)
            new_states.append(ProviderState(cfg=pcfg, ring=ring, provider=provider))

        self.providers = new_states

        # proxy pool (optional)
        pcfg = raw.get("proxy", {})
        if pcfg.get("enabled", False):
            self.proxy_pool = self._build_proxy_pool(pcfg)
        else:
            self.proxy_pool = None

    def _build_proxy_pool(self, pcfg: dict) -> ProxyPool:
        mode = pcfg.get("mode", "file")
        proxies: list[str] = []

        if mode == "file":
            path = pcfg.get("file", "proxies.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    proxies = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]

        # NOTE: proxifly auto-fetch is handled by CLI 'fetch-proxies' command
        # which downloads proxies into proxies.txt.

        return ProxyPool(
            proxies=proxies,
            rotate_every_request=bool(pcfg.get("rotate_every_request", True)),
            max_attempts=int(pcfg.get("max_proxy_attempts", 3)),
            quarantine_failures=int(pcfg.get("quarantine_failures", 3)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def models(self) -> list[dict]:
        """List all configured models per provider."""
        out = []
        for st in self.providers:
            for m in st.cfg.models:
                out.append({"provider": st.cfg.name, "id": m, "type": st.cfg.ptype})
        return out

    def resolve_model(self, requested: Optional[str]) -> tuple[Optional[ProviderState], str]:
        """
        Find a provider that can serve `requested` model.
        Returns (provider_state, model_id) or (None, "") if nothing available.
        """
        if not self.providers:
            return None, ""
        if requested:
            for st in self.providers:
                if requested in st.cfg.models:
                    return st, requested
            if not self.allow_model_routing:
                return None, ""
        if self.default_model:
            for st in self.providers:
                if self.default_model in st.cfg.models:
                    return st, self.default_model
        st = self.providers[0]
        return st, st.cfg.models[0] if st.cfg.models else ""

    async def chat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        *,
        models: Optional[list[str]] = None,   # UI multi-select: inhi models se rotation
        max_tokens: int = 4096,
        temperature: float = 0.7,
        max_fallback_attempts: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[dict] = None,
    ) -> ChatResult:
        """
        Main entry: send messages, rotating keys+models+providers on failure.

        - `model`  = ek specific model pin karo (keys rotate hoti rahengi)
        - `models` = list of models (multi-select), sab rotate honge
        - dono empty = config ke saare configured models rotate
        - `tools` / `tool_choice` = function calling (OpenAI format) pass-through
        """
        attempts = max_fallback_attempts or int(self.settings.get("max_fallback_attempts", 16))
        last_error: Optional[Exception] = None

        # candidates: list of (ProviderState, active_models)
        candidates: list[tuple[ProviderState, list[str]]] = []

        if model:
            st, resolved = self.resolve_model(model)
            if st:
                candidates.append((st, [resolved]))
        else:
            for st in self.providers:
                active = st.cfg.models
                if models is not None:
                    active = [m for m in active if m in models]
                if active:
                    candidates.append((st, active))

        if not candidates:
            raise ProviderError(
                "No providers configured / no matching model. "
                "config.yaml me keys+models check karo ya env vars set karo.",
                retryable=False,
            )

        for _ in range(attempts):
            provider_pick = self._pick_available_provider(candidates)
            if provider_pick is None:
                break
            st, active_models = provider_pick

            picked = st.ring.pick(active_models)
            if picked is None:
                continue
            state, resolved_model = picked

            proxy = self._pick_proxy()
            try:
                result = await st.provider.chat(
                    messages,
                    resolved_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    proxy=proxy,
                    api_key=state.key,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                st.ring.report_success(state, resolved_model)
                st.ring.record_used(state)
                st.successes += 1
                st.last_error = None
                result.key_label = state.label
                result.model = resolved_model
                if proxy and self.proxy_pool:
                    self.proxy_pool.report_success(proxy)
                return result
            except RateLimitError as exc:
                st.ring.report_failure(state, resolved_model)
                if proxy and self.proxy_pool:
                    self.proxy_pool.report_failure(proxy)
                last_error = exc
            except ProviderError as exc:
                st.ring.report_failure(state, resolved_model)
                if proxy and self.proxy_pool:
                    self.proxy_pool.report_failure(proxy)
                st.failures += 1
                st.last_error = str(exc)
                last_error = exc

        raise AllProvidersExhausted(
            f"All providers exhausted after {attempts} attempts. "
            f"Last error: {last_error}",
            retryable=True,
        ) from last_error

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _pick_available_provider(
        self, candidates: list[tuple[ProviderState, list[str]]]
    ) -> Optional[tuple[ProviderState, list[str]]]:
        """
        sequential: current provider pe tab tak ruko jab tak usme koi
        (key, model) available hai — usse poori tarah drain karo, phir
        agla provider. (user ke requirement: "jab poora provider quota
        khtam ho tab dusra provider")
        """
        if not candidates:
            return None

        if self.provider_strategy == "round_robin":
            for offset in range(len(candidates)):
                st, active = candidates[self._provider_cursor % len(candidates)]
                self._provider_cursor += 1
                if st.ring.has_available(active):
                    return st, active
            return None

        # sequential drain
        for offset in range(len(candidates)):
            st, active = candidates[(self._provider_cursor + offset) % len(candidates)]
            if st.ring.has_available(active):
                if offset > 0:
                    # current provider drain ho chuka — agle available pe move
                    self._provider_cursor = (self._provider_cursor + offset) % len(candidates)
                return st, active
        return None

    def _pick_proxy(self) -> Optional[str]:
        if not self.proxy_pool:
            return None
        return self.proxy_pool.next()

    # ------------------------------------------------------------------
    # Status / diagnostics
    # ------------------------------------------------------------------
    def status(self) -> dict:
        return {
            "config": self.config_path,
            "strategy": self.settings.get("strategy", "round_robin"),
            "provider_strategy": self.provider_strategy,
            "proxy": self.proxy_pool.status() if self.proxy_pool else {"enabled": False},
            "providers": [
                {
                    "name": st.cfg.name,
                    "type": st.cfg.ptype,
                    "successes": st.successes,
                    "failures": st.failures,
                    "last_error": st.last_error,
                    **st.ring.status(),
                }
                for st in self.providers
            ],
        }

    async def aclose(self) -> None:
        for st in self.providers:
            await st.provider.aclose()
