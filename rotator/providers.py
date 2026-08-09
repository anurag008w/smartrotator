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

# Zero-width / invisible Unicode characters — ye Python ke str.strip() se
# nahi hattе (isspace() False), par user ko blank reply dikhata hai.
# Kuch models (khaas kar reasoning/formatting wale) ye emit karte hain.
_INVISIBLE_CHARS = (
    "\u200b\u200c\u200d\u200e\u200f"   # ZWSP ZWNJ ZWJ LRM RLM
    "\u2060\u2061\u2062\u2063\u2064"   # word joiner + invisible operators
    "\ufeff\u00ad\u180e"               # BOM/ZWNBSP, soft hyphen, MVS
    "\u202a\u202b\u202c\u202d\u202e"   # bidi embedding controls
    "\u061c"                           # arabic letter mark
)


def clean_text(text: str) -> str:
    """Whitespace + zero-width invisible chars dono ends se strip karo.

    `.strip()` zero-width ko nahi hataata isliye blank replies user tak
    pahunch jaate the. Ab clean text hi aage jata hai.
    """
    if not text:
        return ""
    out = text
    # whitespace aur invisible alag char classes hain — ek alternate loop
    # dono ko expose karke strip karta hai (max 3 passes kaafi hain).
    for _ in range(3):
        nxt = out.strip(_INVISIBLE_CHARS).strip()
        if nxt == out:
            break
        out = nxt
    return out


def is_blank_text(text: str) -> bool:
    """Empty / whitespace-only / zero-width-only → True (blank reply)."""
    return not clean_text(text)


def is_web_search_tool(tool) -> bool:
    """Kya ye tool web search hai? OpenAI-style `{"type": "web_search"}` ya
    `{"type": "function", "function": {"name": "web_search"}}` dono handle karo."""
    if not isinstance(tool, dict):
        return False
    if str(tool.get("type", "")).lower() == "web_search":
        return True
    fn = tool.get("function") or {}
    if isinstance(fn, dict) and str(fn.get("name", "")).lower() in (
        "web_search", "websearch", "search_internet", "google_search",
    ):
        return True
    return False


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
    # Reasoning models (DeepSeek R1 etc.) ka thinking output — DeepSeek API
    # ko tool-call ke baad wale requests me `reasoning_content` WAPAS bhejna
    # MUST hai (warna 400). Isliye ise round-trip preserve karna zaroori.
    reasoning_content: str = ""


@dataclass
class ChatResult:
    text: str
    provider: str
    model: str
    key_label: str
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)  # OpenAI format
    reasoning_content: str = ""          # thinking output (alag rakho — content me mix nahi)


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
        top_p: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        logit_bias: Optional[dict] = None,
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
    def __init__(
        self,
        name: str,
        base_url: str,
        models: list[str],
        web_search_passthrough: bool = False,
        auth_bearer: bool = True,
    ):
        super().__init__(models)
        self.name = name
        # top-level base_url optional ho sakta hai — jab har key ka apna
        # gateway ho (per-key base_url), koi provider-wide default zaroori
        # nahi. Yahan hard-fail nahi karte; agar koi key request-time pe bhi
        # apna base_url na de aur yeh bhi khaali ho, tabhi chat() error dega.
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.endpoint = f"{self.base_url}/chat/completions" if self.base_url else None
        self.web_search_passthrough = web_search_passthrough
        # kuch gateways (OpenCode Zen) `Bearer` prefix reject karte hain —
        # sirf raw key accept karte hain. config me `auth_bearer: false`
        # laga ke unke liye plain Authorization header bhejo.
        self.auth_bearer = auth_bearer

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
        top_p: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        logit_bias: Optional[dict] = None,
        base_url: Optional[str] = None,   # per-key base_url override (optional)
        reasoning_effort: Optional[str] = None,
    ) -> ChatResult:
        if not api_key:
            raise AuthError(f"{self.name}: no api key provided", retryable=False)

        # per-key base_url override — is key ke liye alag gateway/proxy ho toh
        endpoint = self.endpoint
        if base_url:
            endpoint = base_url.rstrip("/") + "/chat/completions"
        if not endpoint:
            raise ProviderError(
                f"{self.name}: base_url configured nahi hai (na provider-wide, na is key ka apna)",
                retryable=False,
            )

        payload_messages = [self._to_openai_message(m) for m in messages]
        payload = {
            "model": model,
            "messages": payload_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # models ki real power — optional params sirf tab bhejo jab diye hon,
        # taaki platforms jo inhe support nahi karte unpe bhi kaam kare
        if top_p is not None:
            payload["top_p"] = top_p
        if stop is not None:
            payload["stop"] = stop
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if response_format is not None:
            payload["response_format"] = response_format
        if seed is not None:
            payload["seed"] = seed
        if logit_bias:
            payload["logit_bias"] = logit_bias
        if reasoning_effort:
            # reasoning effort OpenAI-compatible upstreams pe pass-through —
            # kuch gateways (zen etc.) isse support karte hain
            payload["reasoning_effort"] = reasoning_effort
        if tools:
            # web_search tools most OpenAI-compat platforms pe support nahi hote
            # (OpenRouter/zen unknown type pe 400 dete hain) — default filter karo.
            # Providers jinpe native web search hai (OpenAI, Groq compound, custom
            # gateways) config me `web_search_passthrough: true` laga ke web_search
            # tool ko upstream tak bhej sakte hain.
            function_tools = tools if self.web_search_passthrough else [t for t in tools if not is_web_search_tool(t)]
            if function_tools:
                payload["tools"] = function_tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        headers = {
            "Authorization": f"Bearer {api_key}" if self.auth_bearer else api_key,
            "Content-Type": "application/json",
        }

        client = self._proxy_client(proxy)
        try:
            http = client or self._client
            resp = await http.post(endpoint, headers=headers, json=payload)
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
            # content ko clean karo — reasoning models aage/peeche whitespace
            # aur zero-width invisible chars chhodte hain (aur thinking-budget
            # khatam hone pe sirf ye hi aata hai). Clean text hi user tak
            # jaaye; agar clean ke baad kuch na bache toh router empty treat
            # karke agli key/model try karega.
            text = clean_text(message.get("content") or "")
            # DeepSeek R1 jaise reasoning models content ke alawa
            # `reasoning_content` bhejte hain (thinking output). Isse visible
            # content me KABHI mix mat karo — alag field me rakho taaki client
            # ise wapas pass kar sake (tool-loop round-trip ke liye MUST).
            reasoning = (message.get("reasoning_content") or "").strip()
            tool_calls = message.get("tool_calls") or []
            # kuch reasoning models (zen/big-pickle, DeepSeek R1 via some
            # gateways) poora answer sirf `reasoning_content` me dete hain aur
            # `content` empty chhod dete hain. Router empty text ko failure
            # treat karta hai — isliye content blank ho toh reasoning ko
            # fallback text banao (reasoning_content field bhi preserve rakho
            # taaki round-trip na toote).
            if not text and reasoning and not tool_calls:
                text = reasoning
            usage = data.get("usage", {})
            return ChatResult(
                text=text,
                provider=self.name,
                model=model,
                key_label="openai-compat",
                usage=usage,
                raw=data,
                tool_calls=tool_calls,
                reasoning_content=reasoning,
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
            # assistant ne tools call kiye the — DeepSeek ko ise wapas bhejte
            # waqt `reasoning_content` bhi chahiye (warna 400). preserve karo.
            out: dict = {"role": "assistant", "content": msg.content or None}
            out["tool_calls"] = msg.tool_calls
            if msg.reasoning_content:
                out["reasoning_content"] = msg.reasoning_content
            return out

        if not msg.images and not msg.files:
            out: dict = {"role": msg.role, "content": msg.content}
            if msg.role == "assistant" and msg.reasoning_content:
                out["reasoning_content"] = msg.reasoning_content
            return out

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
        top_p: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        logit_bias: Optional[dict] = None,
        base_url: Optional[str] = None,   # per-key base_url override (optional)
        reasoning_effort: Optional[str] = None,
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
        # Gemini thinking/reasoning output (`thought: true` parts) default me
        # response me NAHI aata — `includeThoughts: true` karna padta hai.
        # Isi se opencode/client ko `reasoning_content` (thinking) milta hai.
        # User/client `reasoning_effort` (low|medium|high) se thinking budget
        # bhi control kar sakta hai — OpenAI-compatible standard field:
        #   low    → chhota budget (fast, halka thinking)
        #   medium → default-ish budget
        #   high   → bada budget (deep thinking)
        #   None   → includeThoughts true, budget model apne default pe
        # Kuch models thinkingConfig ko 400 de sakte hain — HTTPStatusError
        # handler me bina thinkingConfig ke retry hota hai (neeche dekho).
        thinking_cfg: dict = {"includeThoughts": True}
        effort = (reasoning_effort or "").strip().lower()
        if effort == "low":
            # chat replies + tool decisions (default toolThinking)
            thinking_cfg["thinkingBudget"] = 2048
        elif effort == "medium":
            # background memory summaries (default)
            thinking_cfg["thinkingBudget"] = 4096
        elif effort == "high":
            # user ne high select kiya
            thinking_cfg["thinkingBudget"] = 8192
        elif effort == "max":
            # maximum deep thinking
            thinking_cfg["thinkingBudget"] = 16384
        body["generationConfig"]["thinkingConfig"] = thinking_cfg
        # Gemini mapping — models ki real power (jo params Gemini support karta hai)
        if top_p is not None:
            body["generationConfig"]["topP"] = top_p
        if stop is not None:
            stops = [stop] if isinstance(stop, str) else stop
            body["generationConfig"]["stopSequences"] = stops
        if response_format and str(response_format.get("type")) == "json_object":
            body["generationConfig"]["responseMimeType"] = "application/json"
        if tools:
            # Web search tools → Google Search grounding (Gemini native).
            # Function declarations me SIRF function tools jaate hain.
            function_tools = [t for t in tools if not is_web_search_tool(t)]
            search_tools = [t for t in tools if is_web_search_tool(t)]
            body["tools"] = []
            if search_tools:
                body["tools"].append({"google_search": {}})
            if function_tools:
                body["tools"].extend(self._to_gemini_tools(function_tools))
            if not body["tools"]:
                body.pop("tools", None)
            elif search_tools and not function_tools and not tool_choice:
                # sirf search grounding — koi functionConfig mat bhejo
                tool_choice = None
        # Gemini me tool_choice ka native equivalent —
        # OpenAI: "auto" | "required" | "none" (string) ya {type: function}
        # Gemini:  AUTO  | ANY        | NONE   (functionCallingConfig mode)
        # IMPORTANT: opencode/AI-SDK default `tool_choice: "auto"` bhejta
        # hai. Isse mode=ANY banana GALAT hai — wo model ko FORCE karta hai
        # har baar koi na koi tool call karne ke liye (chahe zarurat ho ya
        # nahi). "auto" → AUTO (model khud decide kare) hona chahiye.
        if tool_choice and any(not is_web_search_tool(t) for t in (tools or [])):
            mode = "AUTO"
            allowed: list = []
            if isinstance(tool_choice, str):
                tc = tool_choice.lower()
                if tc == "none":
                    mode = "NONE"
                elif tc in ("required", "any"):
                    mode = "ANY"
                elif tc == "auto":
                    mode = "AUTO"
            elif isinstance(tool_choice, dict):
                # OpenAI style: {"type": "function", "function": {"name": "foo"}}
                fn = (tool_choice.get("function") or {}).get("name")
                if fn:
                    mode = "ANY"
                    allowed = [fn]
                else:
                    mode = "AUTO"
            body["toolConfig"] = {
                "functionCallingConfig": {"mode": mode, "allowedFunctionNames": allowed}
            }

        url = f"{self.base_url}/models/{model}:generateContent"
        if base_url:
            url = base_url.rstrip("/") + f"/models/{model}:generateContent"
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
            reasoning = self._extract_thoughts(data)
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
            # User ko reasoning_content (thinking) bhi chahiye, isliye thinking
            # band karne ke bajaye budget KAM karke retry karo — answer aayega
            # AUR reasoning bhi preserve rahegi. Kuch models (3.6-flash)
            # thinkingConfig hi 400 dete hain → tab bina thinkingConfig retry.
            if (not text.strip() and not tool_calls) or (thoughts > 0 and finish_reason == "MAX_TOKENS"):
                original_reasoning = reasoning
                try:
                    tc = body["generationConfig"].setdefault("thinkingConfig", {})
                    current_budget = tc.get("thinkingBudget") or 4096
                    # thinking kam karke answer ke liye jagah banao
                    tc["thinkingBudget"] = min(current_budget, 2048)
                    resp2 = await http.post(
                        url, headers=headers, params=params, json=body
                    )
                    resp2.raise_for_status()
                    data = self._parse_json_response(resp2, self.name)
                    self._check_error_body(data, self.name)
                    text = self._extract_text(data)
                    tool_calls = self._extract_tool_calls(data)
                    reasoning = self._extract_thoughts(data) or original_reasoning
                    usage = data.get("usageMetadata", {})
                except httpx.HTTPStatusError as exc2:
                    if exc2.response.status_code == 400:
                        # model thinkingConfig support nahi karta — bina
                        # thinkingConfig ke retry (reasoning miss hoga, content
                        # milega); original reasoning bhi saath rakho
                        body["generationConfig"].pop("thinkingConfig", None)
                        try:
                            resp3 = await http.post(
                                url, headers=headers, params=params, json=body
                            )
                            resp3.raise_for_status()
                            data = self._parse_json_response(resp3, self.name)
                            self._check_error_body(data, self.name)
                            text = self._extract_text(data)
                            tool_calls = self._extract_tool_calls(data)
                            reasoning = self._extract_thoughts(data) or original_reasoning
                            usage = data.get("usageMetadata", {})
                        except httpx.HTTPStatusError as exc3:
                            raise self._map_error(exc3, self.name) from exc3
                    else:
                        raise self._map_error(exc2, self.name) from exc2
            return ChatResult(
                text=text,
                provider=self.name,
                model=model,
                key_label="gemini",
                usage=usage,
                raw=data,
                tool_calls=tool_calls,
                reasoning_content=reasoning,
            )
        except httpx.HTTPStatusError as exc:
            # kuch Gemini models `thinkingConfig.includeThoughts` support nahi
            # karte (400 "Invalid JSON payload" / "unknown field") — bina
            # thinkingConfig ke ek baar retry karo taaki model kaam kare
            # (sirf reasoning_content miss hoga, content milega).
            if (
                exc.response.status_code == 400
                and body.get("generationConfig", {}).get("thinkingConfig", {}).get("includeThoughts")
            ):
                body["generationConfig"].pop("thinkingConfig", None)
                try:
                    resp = await http.post(url, headers=headers, params=params, json=body)
                    resp.raise_for_status()
                    data = self._parse_json_response(resp, self.name)
                    self._check_error_body(data, self.name)
                    return ChatResult(
                        text=self._extract_text(data),
                        provider=self.name,
                        model=model,
                        key_label="gemini",
                        usage=data.get("usageMetadata", {}),
                        raw=data,
                        tool_calls=self._extract_tool_calls(data),
                        reasoning_content=self._extract_thoughts(data),
                    )
                except httpx.HTTPStatusError as exc2:
                    raise self._map_error(exc2, self.name) from exc2
            raise self._map_error(exc, self.name) from exc
        except httpx.HTTPError as exc:
            raise self._map_network(exc, self.name) from exc
        finally:
            if client is not None:
                await client.aclose()

    # ------------------------------------------------------------------
    # Embeddings — OpenAI-compatible text → vector (Gemini batchEmbedContents)
    # ------------------------------------------------------------------
    async def embeddings(
        self,
        inputs: list[str],
        model: str = "gemini-embedding-001",
        *,
        proxy: Optional[str] = None,
        api_key: Optional[str] = None,
        chunk_size: int = 100,
    ) -> dict:
        """Texts ka vector banao.

        Gemini ke batchEmbedContents ko OpenAI-compatible response me wrap
        karta hai:
          {"object":"list","data":[{"object":"embedding","embedding":[...],"index":i}],
           "model":..., "usage": {...}}
        Gemini batch limit ~100 requests/call — chunk me bhejo.
        """
        if not api_key:
            raise AuthError("gemini: no api key provided", retryable=False)
        if not inputs:
            raise ProviderError("gemini: embeddings empty input", status_code=400, retryable=False)

        gemini_model = self._map_embedding_model(model)
        url = f"{self.base_url}/models/{gemini_model}:batchEmbedContents"
        headers = {"Content-Type": "application/json"}
        params = {"key": api_key}
        client = self._proxy_client(proxy)
        all_embeddings: list[list[float]] = []
        try:
            http = client or self._client
            for start in range(0, len(inputs), chunk_size):
                chunk = inputs[start:start + chunk_size]
                body = {
                    "requests": [
                        {"model": f"models/{gemini_model}", "content": {"parts": [{"text": t}]}}
                        for t in chunk
                    ]
                }
                resp = await http.post(url, headers=headers, params=params, json=body)
                resp.raise_for_status()
                data = self._parse_json_response(resp, self.name)
                self._check_error_body(data, self.name)
                for e in data.get("embeddings", []):
                    all_embeddings.append(e.get("values") or [])
        except httpx.HTTPStatusError as exc:
            raise self._map_error(exc, self.name) from exc
        except httpx.HTTPError as exc:
            raise self._map_network(exc, self.name) from exc
        finally:
            if client is not None:
                await client.aclose()

        # token estimate: ~4 chars/token (Gemini doesn't return usage for
        # batchEmbedContents — ek sane estimate daal do)
        est = sum(max(1, len(t) // 4) for t in inputs)
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vec, "index": i}
                for i, vec in enumerate(all_embeddings)
            ],
            "model": model,
            "usage": {"prompt_tokens": est, "total_tokens": est},
        }

    @staticmethod
    def _map_embedding_model(model: str) -> str:
        """OpenAI embedding model names → Gemini native names.

        Gemini ke available embedding models (ListModels se confirmed):
          gemini-embedding-001   (3072-dim, modern keys pe stable)
          gemini-embedding-2     / gemini-embedding-2-preview
          text-embedding-004 / 001 / text-multilingual-embedding-002
            (purane free models — kuch keys pe ab available nahi)
        Inhe passthrough karo; baaki (text-embedding-3-small, ada-002,
        unknown...) → default gemini-embedding-001.
        """
        native = {
            "text-embedding-004",
            "text-embedding-001",
            "text-multilingual-embedding-002",
            "gemini-embedding-001",
            "gemini-embedding-2",
            "gemini-embedding-2-preview",
        }
        m = (model or "").strip()
        return m if m in native else "gemini-embedding-001"

    @staticmethod
    def _sanitize_gemini_schema(schema):
        """Pydantic/OpenAI/AI-SDK-style JSON Schema → Gemini-compatible schema.

        Gemini function parameters me JSON Schema ka SUBSET support karta hai.
        AI SDK (opencode), Pydantic, aur tools saare tarah ke keywords bhejte
        hain — Gemini inhe dekhte hi poora request 400 karke reject kar deta
        hai ("Cannot find field" / "property is not defined"). Isliye:
          - whitelist approach: sirf Gemini-supported keywords rakho
          - `$ref`/`$defs` resolve karo (Pydantic schemas ka common pattern)
          - `allOf` merge karo, `anyOf`/`oneOf` nullable-simplify karo
          - `required` me undefined property filter karo
        """
        ALLOWED = {
            "type", "properties", "items", "required", "enum",
            "description", "minimum", "maximum",
        }

        def _sanitize(s: object, defs: dict, depth: int = 0) -> object:
            if depth > 8:
                # self-referencing schemas (recursive $ref loops) — guard
                return {}
            if isinstance(s, list):
                return [_sanitize(x, defs, depth + 1) for x in s]
            if not isinstance(s, dict):
                return s

            # $defs/definitions collect karo (root/any level) — $ref resolve ke liye
            for k in ("$defs", "definitions"):
                v = s.get(k)
                if isinstance(v, dict):
                    defs.update(v)

            out: dict = {}

            # `allOf` merge — Pydantic/OpenAI common: allOf[{a},{b}]
            allof = s.get("allOf")
            if isinstance(allof, list):
                merged_props: dict = {}
                merged_required: list = []
                merged_type: object = None
                for sub in allof:
                    if not isinstance(sub, dict):
                        continue
                    sub2 = _sanitize(sub, defs, depth + 1)
                    if not isinstance(sub2, dict):
                        continue
                    if isinstance(sub2.get("properties"), dict):
                        merged_props.update(sub2["properties"])
                    for r in sub2.get("required") or []:
                        if r not in merged_required:
                            merged_required.append(r)
                    merged_type = merged_type or sub2.get("type")
                if merged_props:
                    out["properties"] = merged_props
                if merged_required:
                    out["required"] = merged_required
                if merged_type:
                    out["type"] = merged_type

            for key, value in s.items():
                # unsupported keywords — Gemini reject karta hai
                if key in (
                    "$schema", "$defs", "definitions", "allOf", "if", "then", "else",
                    "not", "title", "default", "examples", "const", "multipleOf",
                    "additionalProperties", "patternProperties", "pattern",
                    "minLength", "maxLength", "minItems", "maxItems",
                    "minProperties", "maxProperties", "uniqueItems", "contains",
                    "propertyNames", "format", "deprecated", "readOnly", "writeOnly",
                ):
                    continue
                if key == "$ref":
                    # Pydantic `$ref: "#/$defs/Options"` → defs se inline karo
                    name = str(value).rsplit("/", 1)[-1] if isinstance(value, str) else ""
                    if name in defs:
                        return _sanitize(defs[name], defs, depth + 1)
                    continue  # unresolved ref → drop (schemas optional ho jaate hain)
                if key == "exclusiveMinimum":
                    # Gemini sirf minimum/maximum jaanta hai — exclusive ko
                    # approximate karo (integer schemas ke liye +1/-1 exact hai)
                    try:
                        bound = int(value)
                    except (TypeError, ValueError):
                        continue
                    if "minimum" not in out:
                        out["minimum"] = bound + 1
                    continue
                if key == "exclusiveMaximum":
                    try:
                        bound = int(value)
                    except (TypeError, ValueError):
                        continue
                    if "maximum" not in out:
                        out["maximum"] = bound - 1
                    continue
                if key in ("anyOf", "oneOf"):
                    # Pydantic nullable = anyOf[{T}, {null}] → base type rakho
                    variants = [v for v in (value or []) if isinstance(v, dict)]
                    non_null = [v for v in variants if v.get("type") != "null"]
                    if len(non_null) == 1:
                        merged = dict(non_null[0])
                        out.update(_sanitize(merged, defs, depth + 1))
                        continue
                    # complex union — Gemini support nahi karta, optional banao
                    out["type"] = "string"
                    out["description"] = "union type (simplified)"
                    continue
                # container keywords — schema-like values, special handling
                if key == "properties":
                    if isinstance(value, dict):
                        out["properties"] = {
                            name: _sanitize(sub, defs, depth + 1)
                            for name, sub in value.items()
                            if isinstance(sub, (dict, list)) or sub is None
                        }
                    continue
                if key == "items":
                    if isinstance(value, dict):
                        out["items"] = _sanitize(value, defs, depth + 1)
                    elif isinstance(value, list):
                        # tuple validation (rare) — pehle wala schema le lo
                        out["items"] = _sanitize(value[0], defs, depth + 1) if value else {}
                    continue
                # whitelist: sirf Gemini-supported keywords
                if key not in ALLOWED:
                    continue
                out[key] = value

            # required me sirf defined properties rakho — Gemini "property is
            # not defined" 400 deta hai undefined reference pe (opencode/AI SDK
            # ka common pattern: required me property jo properties me nahi).
            props = out.get("properties")
            if isinstance(props, dict) and isinstance(out.get("required"), list):
                out["required"] = [r for r in out["required"] if r in props]
                if not out["required"]:
                    out.pop("required", None)

            # khaali schema → Gemini ko at least type chahiye
            if not out:
                out = {"type": "object"}
            return out

        return _sanitize(schema, {})

    @staticmethod
    def _to_gemini_tools(tools: list[dict]) -> list[dict]:
        """OpenAI tools → Gemini functionDeclarations."""
        declarations = []
        for t in tools:
            fn = (t.get("function") or {}) if isinstance(t, dict) else {}
            parameters = fn.get("parameters") or {"type": "object", "properties": {}}
            params = GeminiProvider._sanitize_gemini_schema(parameters)
            # OpenAI/AI SDK (opencode) ke schemas me `required` me aisi
            # property ho sakti hai jo `properties` me defined NAHI hai.
            # Gemini ise 400 deta hai ("required[0]: property is not
            # defined") — filter karo, warna saare providers fail dikhte hain.
            props = params.get("properties") or {}
            if isinstance(props, dict) and isinstance(params.get("required"), list):
                params["required"] = [r for r in params["required"] if r in props]
                if not params["required"]:
                    params.pop("required", None)
            declarations.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    # Pydantic schemas ($schema, exclusiveMinimum...) ko
                    # Gemini-compatible banao — warna 400 + sab providers fail.
                    "parameters": params,
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
                # Gemini ko functionResponse.name REQUIRED hai. OpenAI clients
                # (opencode/AI SDK, curl, ChatGPT-style) tool message me `name`
                # nahi bhejte — sirf tool_call_id. Toh name ko isse pehle ke
                # assistant functionCall se map karo, warna Gemini 400 dega
                # ("Name cannot be empty") aur sab providers exhausted dikhenge.
                fname = msg.name
                if not fname:
                    for prev in contents:
                        for part in prev.get("parts", []):
                            fc = part.get("functionCall")
                            if fc and fc.get("name"):
                                fname = fc["name"]
                                break
                        if fname:
                            break
                if not fname:
                    fname = "tool_call"
                parts.append(
                    {
                        "functionResponse": {
                            "name": fname,
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
        # `thought: true` parts model ki thinking hoti hai — use visible content
        # me KABHI mix mat karo. Sirf asli answer join hota hai.
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            return ""
        return clean_text("".join(p.get("text", "") for p in parts if not p.get("thought")))

    @staticmethod
    def _extract_thoughts(data: dict) -> str:
        """Gemini thinking parts (`thought: true`) → reasoning_content."""
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            return ""
        return "\n".join(p.get("text", "") for p in parts if p.get("thought")).strip()

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
def build_provider(
    name: str,
    ptype: str,
    base_url: str | None,
    models: list[str],
    web_search_passthrough: bool = False,
    auth_bearer: bool = True,
) -> Provider:
    if ptype == "gemini":
        # custom base_url (Cloudflare Worker gateway / proxy / alt endpoint)
        # allowed — default Google endpoint tab use hota hai jab koi nahi diya
        return GeminiProvider(models, base_url or GEMINI_V1)
    if ptype == "openai":
        # base_url yahan mandatory nahi — agar har key ka apna per-key
        # base_url hai (Cloudflare Worker jaisa gateway), provider-wide
        # default ki zaroorat nahi. Missing endpoint ka actual check
        # request-time pe OpenAICompatibleProvider.chat() karta hai.
        return OpenAICompatibleProvider(
            name,
            base_url or "",
            models,
            web_search_passthrough=web_search_passthrough,
            auth_bearer=auth_bearer,
        )
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
            seen: set[str] = set()
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
                    new_count = 0
                    for raw in items:
                        m = _map_openai_model(raw, name, fetched_at)
                        if m and m.id not in seen:
                            seen.add(m.id)
                            models.append(m)
                            new_count += 1
                    # pagination: OpenAI /models pe last_id hota hai
                    last_id = data.get("last_id") if isinstance(data, dict) else None
                    if not last_id and items:
                        last_id = items[-1].get("id")
                    # server ne poora catalog ek page me de diya (last_id nahi)
                    # ya page < 20 items ka hai → pagination khatam.
                    if not last_id or len(items) < 20:
                        break
                    # NVIDIA NIM jaise providers `after` param IGNORE karke wahi
                    # list dobara bhejte hain — naya model na aaye toh ruko
                    # (warna 5x duplicate models milte).
                    if new_count == 0:
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
