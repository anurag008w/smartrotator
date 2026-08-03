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

        # Model Manager (DB se aata hai — admin dashboard set karta hai)
        self.managed: dict = {}
        self.groups: list[dict] = []
        self._strategy = "round_robin"
        self._cooldown = 60.0
        self._ban_after = 3

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
        self._strategy = strategy
        self._cooldown = cooldown
        self._ban_after = ban_after

        server = raw.get("server", {})
        self.default_model = server.get("default_model", "gemini-2.5-flash")
        self.allow_model_routing = bool(server.get("allow_model_routing", True))

        # build provider states
        new_states: list[ProviderState] = []
        for cfg in raw.get("providers", []):
            st = self._build_state_from_cfg(cfg)
            if st is not None:
                new_states.append(st)

        self.providers = new_states

        # custom providers (dashboard se add kiye) dobara apply karo
        if getattr(self, "_custom_provider_cfgs", None):
            self._merge_custom_states()

        # proxy pool (optional)
        pcfg = raw.get("proxy", {})
        if pcfg.get("enabled", False):
            self.proxy_pool = self._build_proxy_pool(pcfg)
        else:
            self.proxy_pool = None

    def _build_state_from_cfg(self, cfg: dict) -> Optional[ProviderState]:
        """Ek provider ka ProviderState banata hai (config.yaml section se)."""
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
            return None  # provider setup incomplete, skip silently

        return self._build_state(
            ProviderConfig(
                name=name,
                ptype=ptype,
                models=models,
                keys=keys,
                base_url=base_url,
                rpm_limit=int(cfg.get("rpm_limit", 0)),
                rpd_limit=int(cfg.get("rpd_limit", 0)),
            )
        )

    # ------------------------------------------------------------------
    # Custom providers — admin dashboard se add/remove kiye providers
    # ------------------------------------------------------------------
    def apply_custom_providers(self, custom_cfgs: list[dict]) -> None:
        """Dashboard-added providers ko live laga deta hai.

        `custom_cfgs` = store (data/providers.json) se aaye providers:
          [{"name", "type", "base_url", "api_keys", "models", "enabled"}, ...]

        Inhe existing providers ke saath merge karta hai — same name pe
        custom wala win karta hai (config.yaml override nahi hota).
        """
        self._custom_provider_cfgs = list(custom_cfgs)
        self._merge_custom_states()
        # managed config (order/models) phir se apply karo
        if self.managed:
            self.apply_managed(self.managed)

    def _merge_custom_states(self) -> None:
        """Custom provider configs se ProviderState banake merge karo."""
        custom_cfgs = getattr(self, "_custom_provider_cfgs", []) or []
        custom_states: list[ProviderState] = []
        for cfg in custom_cfgs:
            if not cfg.get("enabled", True):
                continue
            name = (cfg.get("name") or "").strip()
            ptype = cfg.get("type", "openai")
            models = [m.strip() for m in cfg.get("models", []) if m.strip()]
            keys = [k.strip() for k in cfg.get("api_keys", []) if k.strip() if not k.startswith("PASTE_")]
            if not name or not keys or not models:
                continue
            custom_states.append(
                self._build_state(
                    ProviderConfig(
                        name=name,
                        ptype=ptype,
                        models=models,
                        keys=keys,
                        base_url=cfg.get("base_url"),
                        rpm_limit=int(cfg.get("rpm_limit", 0)),
                        rpd_limit=int(cfg.get("rpd_limit", 0)),
                    )
                )
            )

        # same name ke custom provider se config.yaml wala override
        custom_names = {st.cfg.name for st in custom_states}
        base_states = [st for st in self.providers if st.cfg.name not in custom_names]
        self.providers = base_states + custom_states

    def _build_state(self, pcfg: ProviderConfig) -> ProviderState:
        """ProviderConfig se ring + provider + state bana deta hai."""
        ring = KeyRing(
            keys=pcfg.keys,
            models=pcfg.models,
            label=pcfg.name,
            strategy=self._strategy,
            cooldown_seconds=self._cooldown,
            ban_after=self._ban_after,
            rpm_limit=pcfg.rpm_limit,
            rpd_limit=pcfg.rpd_limit,
        )
        provider = build_provider(pcfg.name, pcfg.ptype, pcfg.base_url, pcfg.models)
        return ProviderState(cfg=pcfg, ring=ring, provider=provider)

    def _find_provider(self, name: str) -> Optional[ProviderState]:
        for st in self.providers:
            if st.cfg.name == name:
                return st
        return None

    def _find_group(self, group_id: str) -> Optional[dict]:
        for g in self.groups:
            if g["id"] == group_id:
                return g
        return None

    # ------------------------------------------------------------------
    # Model Manager — dashboard se save ki hui config ko live apply karo
    # ------------------------------------------------------------------
    def apply_managed(self, managed: Optional[dict]) -> None:
        """Active models + groups + provider order ko live laga deta hai.

        DB (Postgres) me stored hai isliye redeploy ke baad bhi survive karta hai.
        config.yaml ko touch nahi karta — bas runtime state update hota hai.
        """
        managed = managed or {}
        self.managed = managed

        # 1. groups
        self.groups = []
        for g in managed.get("groups", []):
            gid = (g.get("id") or "").strip()
            members = [
                {"provider": m.get("provider", ""), "model": m.get("model", "")}
                for m in g.get("members", [])
                if m.get("provider") and m.get("model")
            ]
            if gid and members:
                self.groups.append(
                    {
                        "id": gid,
                        "label": g.get("label") or gid,
                        "enabled": bool(g.get("enabled", True)),
                        "members": members,
                    }
                )

        # 2. provider order
        order = [p for p in (managed.get("provider_order") or []) if p]
        if order:
            index = {p: i for i, p in enumerate(order)}
            self.providers.sort(key=lambda st: index.get(st.cfg.name, 10**9))

        # 3. active model override (dashboard se select kiye models)
        overrides = managed.get("provider_models") or {}
        rebuilt: list[ProviderState] = []
        for st in self.providers:
            models = [m for m in (overrides.get(st.cfg.name) or []) if m]
            if models:
                pcfg = ProviderConfig(
                    name=st.cfg.name,
                    ptype=st.cfg.ptype,
                    models=models,
                    keys=st.cfg.keys,
                    base_url=st.cfg.base_url,
                    rpm_limit=st.cfg.rpm_limit,
                    rpd_limit=st.cfg.rpd_limit,
                )
                st = self._build_state(pcfg)
            rebuilt.append(st)
        self.providers = rebuilt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def models(self) -> list[dict]:
        """List all configured models per provider + virtual model groups."""
        out = []
        for st in self.providers:
            for m in st.cfg.models:
                out.append({"provider": st.cfg.name, "id": m, "type": st.cfg.ptype})
        for g in self.groups:
            if g["enabled"]:
                out.append({"provider": "smartrotator", "id": g["id"], "type": "group"})
        return out

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
            # Virtual model group (jaise "levelup")? -> group ke members me rotate
            group = self._find_group(model)
            if group is not None:
                if not group["enabled"]:
                    raise ProviderError(
                        f"Model group '{model}' disabled hai. Dashboard se enable karo.",
                        retryable=False,
                    )
                for member in group["members"]:
                    st = self._find_provider(member["provider"])
                    if st and member["model"] in st.cfg.models:
                        candidates.append((st, [member["model"]]))
                if not candidates:
                    raise ProviderError(
                        f"Model group '{model}' me koi available model nahi hai. "
                        "Dashboard me models select karo.",
                        retryable=False,
                    )
            else:
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
