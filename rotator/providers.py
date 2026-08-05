"""
providers.py — Provider adapters.

Har provider ek OpenAI-compatible ya native API hit karta hai.
Sab async hain (httpx.AsyncClient), taaki FastAPI me bina block
kiye chalein aur CLI me asyncio.run() se chal sakein.

Vision (image input) dono type ke providers me supported hai:
  - Gemini      : inline_data (base64) parts
  - OpenAI-compat: image_url content parts (data URL ya http URL)
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Optional, Union
from urllib.parse import quote

import httpx

GEMINI_V1 = "https://generativelanguage.googleapis.com/v1beta"


# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------
@dataclass
class ImageInput:
    """An image to send to the model."""

    url: Optional[str] = None            # public http(s) URL
    data_base64: Optional[str] = None    # raw base64 payload
    mime_type: str = "image/jpeg"


@dataclass
class ChatMessage:
    role: str = "user"                   # system | user | assistant | tool
    content: str = ""
    images: list[ImageInput] = field(default_factory=list)
    # Generic file parts (PDF/docx/pptx/xlsx...) — OpenAI `file` content parts.
    files: list[ImageInput] = field(default_factory=list)
    # Tool calling (OpenAI format):
    tool_calls: list[dict] = field(default_factory=list)  # assistant → model ke calls
    tool_call_id: str = ""               # tool result message me
    name: str = ""                       # tool result message me (function name)


@dataclass
class ChatResult:
    text: str
    provider: str
    model: str
    key_label: str
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)  # OpenAI format


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class ProviderError(Exception):
    """Base provider error."""

    def __init__(self, message: str, status_code: Optional[int] = None, retryable: bool = True):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class RateLimitError(ProviderError):
    """429 — key ko cooldown me daalo, doosri key try karo."""


class AuthError(ProviderError):
    """401/403 — key invalid hai."""


class AllProvidersExhausted(ProviderError):
    """Saare providers ki keys/models fail — system-level overload.

    User ko sirf generic message dikhega (raw provider error sirf logs me).
    """


# --------------------------------------------------------------------------
# Provider base
# --------------------------------------------------------------------------
class Provider:
    name: str = "base"

    def __init__(self, models: list[str]):
        self.models = models
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        proxy: Optional[str] = None,
        api_key: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> ChatResult:
        raise NotImplementedError

    @staticmethod
    def _proxy_client(proxy: Optional[str]) -> Optional[httpx.AsyncClient]:
        """Build a one-off client with a proxy (proxy rotation per request)."""
        if not proxy:
            return None
        url = proxy if "://" in proxy else f"http://{proxy}"
        return httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0), proxy=url)

    # -- shared response parsing / error mapping -----------------------------
    @staticmethod
    def _parse_json_response(resp: httpx.Response, context: str) -> dict:
        """Response body ko safely JSON me parse karo.

        Kuch platforms 200 pe bhi HTML / plain text / bina JSON body ke
        respond karte hain. Aisi body pe `resp.json()` JSONDecodeError fekta
        hai jo pehle unhandled reh kar 500 ban jata tha — ab ProviderError
        (502) me convert karte hain taaki router doosri key/model try kare.
        """
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(
                f"{context}: invalid JSON response (status {resp.status_code}): {resp.text[:200]}",
                status_code=502,
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError(
                f"{context}: unexpected response body (expected JSON object): {resp.text[:200]}",
                status_code=502,
            )
        return data

    @staticmethod
    def _extract_error_message(data: dict, context: str) -> str:
        """OpenAI-style error body se human-readable message nikalo."""
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or str(err)
            return str(msg)[:300]
        if err:
            return str(err)[:300]
        msg = data.get("message")
        if msg:
            return str(msg)[:300]
        return f"{context}: provider error: {str(data)[:200]}"

    @staticmethod
    def _check_error_body(data: dict, context: str) -> None:
        """200 status pe bhi kuch gateways `{"error": ...}` bhejte hain."""
        if data.get("error") or data.get("message"):
            raise ProviderError(
                f"{context}: {Provider._extract_error_message(data, context)}",
                status_code=502,
            )

    @staticmethod
    def _map_error(exc: httpx.HTTPStatusError, context: str) -> ProviderError:
        code = exc.response.status_code
        body = exc.response.text[:500]
        # JSON error body hai toh human-readable message nikaalo (logs ke liye)
        try:
            data = exc.response.json()
            if isinstance(data, dict):
                body = Provider._extract_error_message(data, context)
        except ValueError:
            pass
        if code == 429:
            return RateLimitError(f"{context}: rate limited (429): {body}", status_code=429)
        if code in (401, 403):
            return AuthError(f"{context}: auth failed ({code}): {body}", status_code=code, retryable=False)
        if code >= 500:
            return ProviderError(f"{context}: server error ({code}): {body}", status_code=code)
        return ProviderError(f"{context}: http {code}: {body}", status_code=code, retryable=False)

    @staticmethod
    def _map_network(exc: Exception, context: str) -> ProviderError:
        return ProviderError(f"{context}: network error: {exc.__class__.__name__}: {exc}")


# --------------------------------------------------------------------------
# OpenAI-compatible provider (Groq, OpenRouter, OpenCode Zen, ...)
# --------------------------------------------------------------------------
class OpenAICompatibleProvider(Provider):
    def __init__(self, name: str, base_url: str, models: list[str]):
        super().__init__(models)
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/chat/completions"

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        proxy: Optional[str] = None,
        api_key: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> ChatResult:
        if not api_key:
            raise AuthError(f"{self.name}: no api key provided", retryable=False)

        payload_messages = [self._to_openai_message(m) for m in messages]
        payload = {
            "model": model,
            "messages": payload_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        client = self._proxy_client(proxy)
        try:
            http = client or self._client
            resp = await http.post(self.endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = self._parse_json_response(resp, self.name)
            # kuch gateways 200 status pe hi error body bhej dete hain
            self._check_error_body(data, self.name)

            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ProviderError(
                    f"{self.name}: unexpected response — no 'choices' in body: {resp.text[:200]}",
                    status_code=502,
                )
            first = choices[0] if isinstance(choices[0], dict) else None
            message = first.get("message") if first else None
            if not isinstance(message, dict):
                raise ProviderError(
                    f"{self.name}: unexpected response — choice has no 'message': {resp.text[:200]}",
                    status_code=502,
                )
            # content ko strip karo — reasoning models aage/peeche whitespace
            # chhodte hain (aur thinking-budget khatam hone pe sirf whitespace
            # bhi aata hai). Clean text hi user tak jaaye; agar strip ke baad
            # kuch na bache toh router empty treat karke agli key/model try
            # karega.
            text = (message.get("content") or "").strip()
            tool_calls = message.get("tool_calls") or []
            usage = data.get("usage", {})
            return ChatResult(
                text=text,
                provider=self.name,
                model=model,
                key_label="openai-compat",
                usage=usage,
                raw=data,
                tool_calls=tool_calls,
            )
        except httpx.HTTPStatusError as exc:
            raise self._map_error(exc, self.name) from exc
        except httpx.HTTPError as exc:
            raise self._map_network(exc, self.name) from exc
        finally:
            if client is not None:
                await client.aclose()

    @staticmethod
    def _to_openai_message(msg: ChatMessage) -> dict:
        if msg.role == "tool":
            # tool ka result — original tool_call_id ke saath wapas
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            }

        if msg.tool_calls:
            # assistant ne tools call kiye the
            out: dict = {"role": "assistant", "content": msg.content or None}
            out["tool_calls"] = msg.tool_calls
            return out

        if not msg.images and not msg.files:
            return {"role": msg.role, "content": msg.content}

        content: list[dict] = [{"type": "text", "text": msg.content}]
        for img in msg.images:
            if img.data_base64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img.mime_type};base64,{img.data_base64}"},
                    }
                )
            elif img.url:
                content.append({"type": "image_url", "image_url": {"url": img.url}})
        for f in msg.files:
            # OpenAI-compatible gateways jo `file` parts samajhte hain unhe
            # passthrough karo (image_url nahi — yeh PDF/docx ho sakte hain).
            if f.data_base64:
                content.append(
                    {
                        "type": "file",
                        "file": {
                            "file_data": f"data:{f.mime_type};base64,{f.data_base64}",
                        },
                    }
                )
            elif f.url:
                content.append({"type": "file", "file": {"file_url": f.url}})
        return {"role": msg.role, "content": content}


# --------------------------------------------------------------------------
# Google Gemini provider (native API, free tier 1500 RPD)
# --------------------------------------------------------------------------
class GeminiProvider(Provider):
    def __init__(self, models: list[str], base_url: str = GEMINI_V1):
        super().__init__(models)
        self.name = "gemini"
        self.base_url = base_url.rstrip("/")

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        proxy: Optional[str] = None,
        api_key: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> ChatResult:
        if not api_key:
            raise AuthError("gemini: no api key provided", retryable=False)

        contents, system_parts = self._to_gemini_contents(messages)
        body: dict = {"contents": contents}
        if system_parts:
            body["system_instruction"] = {"parts": [{"text": t} for t in system_parts]}
        body["generationConfig"] = {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = self._to_gemini_tools(tools)
        # Gemini me tool_choice ka native equivalent nahi hai —
        # "any" chahiye toh function_calling_config use hota hai
        if tool_choice and tools:
            body["toolConfig"] = {
                "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": []}
            }

        url = f"{self.base_url}/models/{model}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": api_key}

        client = self._proxy_client(proxy)
        try:
            http = client or self._client
            resp = await http.post(url, headers=headers, params=params, json=body)
            resp.raise_for_status()
            data = self._parse_json_response(resp, self.name)
            self._check_error_body(data, self.name)
            text = self._extract_text(data)
            tool_calls = self._extract_tool_calls(data)
            usage = data.get("usageMetadata", {})
            thoughts = usage.get("thoughtsTokenCount", 0) or 0
            finish_reason = ""
            try:
                finish_reason = data["candidates"][0].get("finishReason", "")
            except (KeyError, IndexError):
                pass
            # Reasoning model (gemini-3.x-flash etc.) ne saara token budget
            # thinking me kha liya → reply empty ya truncated aata hai
            # (thoughtsTokenCount bada, text "" ya MAX_TOKENS pe ruk gaya).
            # thinkingBudget: 0 se thinking band karke seedha answer milta hai —
            # bas kuch models (3.6-flash) is config ko 400 dete hain, isliye
            # sirf is case me try karo aur 400 aaye to original reply chhodo.
            if (not text.strip() and not tool_calls) or (thoughts > 0 and finish_reason == "MAX_TOKENS"):
                try:
                    body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
                    resp2 = await http.post(
                        url, headers=headers, params=params, json=body
                    )
                    resp2.raise_for_status()
                    data = self._parse_json_response(resp2, self.name)
                    self._check_error_body(data, self.name)
                    text = self._extract_text(data)
                    tool_calls = self._extract_tool_calls(data)
                    usage = data.get("usageMetadata", {})
                except httpx.HTTPStatusError as exc2:
                    # model thinkingConfig support nahi karta — original reply
                    # hi rakh lo; router empty text ko failure treat karke agli
                    # key/model try karega
                    pass
            return ChatResult(
                text=text,
                provider=self.name,
                model=model,
                key_label="gemini",
                usage=usage,
                raw=data,
                tool_calls=tool_calls,
            )
        except httpx.HTTPStatusError as exc:
            raise self._map_error(exc, self.name) from exc
        except httpx.HTTPError as exc:
            raise self._map_network(exc, self.name) from exc
        finally:
            if client is not None:
                await client.aclose()

    @staticmethod
    def _to_gemini_tools(tools: list[dict]) -> list[dict]:
        """OpenAI tools → Gemini functionDeclarations."""
        declarations = []
        for t in tools:
            fn = (t.get("function") or {}) if isinstance(t, dict) else {}
            declarations.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        return [{"functionDeclarations": declarations}]

    @staticmethod
    def _to_gemini_contents(messages: list[ChatMessage]) -> tuple[list[dict], list[str]]:
        """Convert ChatMessages to Gemini contents. System msgs become parts."""
        contents: list[dict] = []
        system_parts: list[str] = []
        for msg in messages:
            parts: list[dict] = []
            if msg.role == "tool":
                # tool ka result → functionResponse
                response = msg.content
                try:
                    response = json.loads(msg.content)  # already JSON hai toh rakh lo
                except (json.JSONDecodeError, TypeError):
                    pass
                parts.append(
                    {
                        "functionResponse": {
                            "name": msg.name,
                            "response": {"result": response},
                        }
                    }
                )
                contents.append({"role": "user", "parts": parts})
                continue

            if msg.tool_calls:
                # assistant ne function calls kiye the → functionCall parts
                for tc in msg.tool_calls:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    parts.append(
                        {"functionCall": {"name": fn.get("name", ""), "args": args}}
                    )
                contents.append({"role": "model", "parts": parts})
                continue

            if msg.content.strip():
                parts.append({"text": msg.content})
            for img in msg.images:
                if img.data_base64:
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": img.mime_type,
                                "data": img.data_base64,
                            }
                        }
                    )
                elif img.url:
                    parts.append({"file_data": {"mime_type": img.mime_type, "file_uri": img.url}})
            for f in msg.files:
                # PDF/docx inline raw bytes — Gemini `inline_data` (fileData
                # requires an already-uploaded Files-API URI, so base64 is the
                # way for direct requests).
                if f.data_base64:
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": f.mime_type,
                                "data": f.data_base64,
                            }
                        }
                    )
                elif f.url:
                    parts.append({"file_data": {"mime_type": f.mime_type, "file_uri": f.url}})
            if msg.role == "system":
                if msg.content.strip():
                    system_parts.append(msg.content)
                continue
            if not parts:
                parts.append({"text": ""})
            gemini_role = "model" if msg.role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": parts})
        return contents, system_parts

    @staticmethod
    def _extract_text(data: dict) -> str:
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError):
            return ""

    @staticmethod
    def _extract_tool_calls(data: dict) -> list[dict]:
        """Gemini functionCall parts → OpenAI tool_calls format."""
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            return []
        calls = []
        for p in parts:
            fc = p.get("functionCall")
            if not fc:
                continue
            name = fc.get("name", "")
            args = fc.get("args") or {}
            calls.append(
                {
                    "id": f"call_{name}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            )
        return calls


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def build_provider(name: str, ptype: str, base_url: str | None, models: list[str]) -> Provider:
    if ptype == "gemini":
        return GeminiProvider(models)
    if ptype == "openai":
        if not base_url:
            raise ValueError(f"provider '{name}': openai type needs base_url")
        return OpenAICompatibleProvider(name, base_url, models)
    raise ValueError(f"provider '{name}': unknown type '{ptype}'")


# --------------------------------------------------------------------------
# Live models fetch (mobile app jaisa — provider APIs se real models)
# --------------------------------------------------------------------------
@dataclass
class LiveModel:
    """Provider API se fetch kiya hua model (mobile app ke ModelInfo jaisa)."""

    id: str
    name: str = ""
    provider: str = ""
    context_length: Optional[int] = None
    supports_streaming: Optional[bool] = None
    supports_vision: Optional[bool] = None
    supports_reasoning: Optional[bool] = None
    supports_tool_calling: Optional[bool] = None
    is_free: bool = False
    fetched_at: float = 0.0


def _map_gemini_model(raw: dict, provider: str, fetched_at: float) -> Optional[LiveModel]:
    name = raw.get("name", "")
    if not isinstance(name, str):
        return None
    mid = name.replace("models/", "")
    if not mid:
        return None
    display = raw.get("displayName") or mid
    desc = str(raw.get("description") or "").lower()
    methods = raw.get("supportedGenerationMethods") or []
    return LiveModel(
        id=mid,
        name=display,
        provider=provider,
        context_length=raw.get("inputTokenLimit"),
        supports_streaming="streamGenerateContent" in methods,
        supports_vision=True if ("vision" in desc or "image" in desc or "multimodal" in desc) else None,
        supports_reasoning=True if ("reasoning" in desc) else None,
        supports_tool_calling=True if ("function calling" in desc or "tools" in desc) else None,
        fetched_at=fetched_at,
    )


def _map_openai_model(raw: dict, provider: str, fetched_at: float) -> Optional[LiveModel]:
    mid = raw.get("id")
    if not isinstance(mid, str) or not mid:
        return None
    params = raw.get("supported_parameters") or []
    input_mods = raw.get("input_modalities") or ["text"]
    pricing = raw.get("pricing") or {}
    prompt = pricing.get("prompt") if isinstance(pricing, dict) else None
    completion = pricing.get("completion") if isinstance(pricing, dict) else None
    is_free = prompt in (0, 0.0, "0") and completion in (0, 0.0, "0")
    return LiveModel(
        id=mid,
        name=raw.get("name") or mid,
        provider=provider,
        context_length=raw.get("context_length") or raw.get("context_window"),
        supports_streaming="streaming" in params if params else None,
        supports_vision=True if "image" in input_mods else ("vision" in params if params else None),
        supports_reasoning="reasoning" in params if params else None,
        supports_tool_calling="tools" in params if params else None,
        is_free=is_free,
        fetched_at=fetched_at,
    )


async def fetch_live_models(
    name: str,
    ptype: str,
    base_url: Optional[str],
    api_keys: list[str],
    timeout: float = 20.0,
    max_pages: int = 5,
) -> list[LiveModel]:
    """Provider API se live models fetch karo (har key try, first success wins).

    - gemini : GET {base}/v1beta/models?pageSize=200 (+ pageToken)
    - openai : GET {base}/models                (+ after / last_id)

    Mobile app (levelup) ke fetchModels() jaisa hi pattern.
    """
    if ptype == "gemini":
        return await _fetch_live_gemini(name, base_url, api_keys, timeout, max_pages)
    if ptype == "openai":
        return await _fetch_live_openai(name, base_url, api_keys, timeout, max_pages)
    raise ValueError(f"provider '{name}': unknown type '{ptype}'")


async def _fetch_live_gemini(
    name: str,
    base_url: Optional[str],
    api_keys: list[str],
    timeout: float,
    max_pages: int,
) -> list[LiveModel]:
    if not api_keys:
        raise AuthError(f"{name}: no api keys provided", retryable=False)
    if base_url:
        base = base_url.rstrip("/")
        # base_url ya to root hai (isliye /v1beta/models) ya already /v1beta
        if base.endswith("/v1beta") or base.endswith("/v1"):
            models_url = f"{base}/models"
        else:
            models_url = f"{base}/v1beta/models"
    else:
        models_url = f"{GEMINI_V1}/models"

    last_err: Optional[Exception] = None
    for key in api_keys:
        try:
            models: list[LiveModel] = []
            page_token: Optional[str] = None
            fetched_at = __import__("time").time()
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
                for _ in range(max_pages):
                    url = f"{models_url}?pageSize=200"
                    if page_token:
                        url += f"&pageToken={quote(page_token)}"
                    resp = await client.get(url, params={"key": key})
                    resp.raise_for_status()
                    data = Provider._parse_json_response(resp, name)
                    for raw in data.get("models", []):
                        m = _map_gemini_model(raw, name, fetched_at)
                        if m:
                            models.append(m)
                    page_token = data.get("nextPageToken")
                    if not page_token:
                        break
            return models
        except httpx.HTTPStatusError as exc:
            last_err = exc
            if exc.response.status_code in (401, 403):
                continue  # is key se nai — next key try
            raise self_map_error_live(exc, name) from exc
        except httpx.HTTPError as exc:
            last_err = exc
            continue
    raise ProviderError(f"{name}: live models fetch failed: {last_err}") from last_err


async def _fetch_live_openai(
    name: str,
    base_url: Optional[str],
    api_keys: list[str],
    timeout: float,
    max_pages: int,
) -> list[LiveModel]:
    if not api_keys:
        raise AuthError(f"{name}: no api keys provided", retryable=False)
    if not base_url:
        raise ValueError(f"provider '{name}': openai type needs base_url")
    base = base_url.rstrip("/")
    models_url = f"{base}/models"

    last_err: Optional[Exception] = None
    for key in api_keys:
        try:
            models: list[LiveModel] = []
            after: Optional[str] = None
            fetched_at = __import__("time").time()
            headers = {"Authorization": f"Bearer {key}"}
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
                for _ in range(max_pages):
                    url = models_url
                    if after:
                        url += f"?after={quote(after)}"
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    data = Provider._parse_json_response(resp, name)
                    items = data.get("data", []) if isinstance(data, dict) else []
                    for raw in items:
                        m = _map_openai_model(raw, name, fetched_at)
                        if m:
                            models.append(m)
                    # pagination: OpenAI /models pe last_id hota hai
                    last_id = data.get("last_id") if isinstance(data, dict) else None
                    if not last_id and items:
                        last_id = items[-1].get("id")
                    if not last_id or len(items) < 20:
                        break
                    after = last_id
            return models
        except httpx.HTTPStatusError as exc:
            last_err = exc
            if exc.response.status_code in (401, 403):
                continue
            raise self_map_error_live(exc, name) from exc
        except httpx.HTTPError as exc:
            last_err = exc
            continue
    raise ProviderError(f"{name}: live models fetch failed: {last_err}") from last_err


def self_map_error_live(exc: httpx.HTTPStatusError, context: str) -> ProviderError:
    """Live-fetch errors ke liye chhota error mapper (reuse provider._map_error)."""
    return Provider._map_error(exc, context)
