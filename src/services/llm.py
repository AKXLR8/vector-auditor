"""LLM client for OpenAI-compatible APIs with circuit breaker + retry.
Supports multiple providers with model-specific tuning profiles."""
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import httpx

from .circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, retry_with_backoff
from .secrets import get_secret

logger = logging.getLogger("rga_auditor.llm")


class LLMError(RuntimeError):
    pass


# ── Provider profiles ────────────────────────────────────────────────────────
# Each profile maps the modes (black_box, white_box, analyze, verify, gaps) to
# model-specific parameters. Add new profiles here or via env overrides.

@dataclass
class _ModeParams:
    temperature: float = 0.5
    max_tokens: int = 4096
    extra: dict = field(default_factory=dict)


@dataclass
class _Profile:
    name: str
    base_url: str
    model: str
    modes: dict[str, _ModeParams] = field(default_factory=lambda: {
        "black_box": _ModeParams(temperature=0.5, max_tokens=4096,
            extra={"reasoning_effort": "medium", "reasoning_summary": False}),
        "white_box": _ModeParams(temperature=0.3, max_tokens=5120,
            extra={"reasoning_effort": "high", "reasoning_summary": True, "reasoning_summary_wait": True}),
        "analyze":  _ModeParams(temperature=0.3, max_tokens=4096,
            extra={"reasoning_effort": "high", "reasoning_summary": True, "reasoning_summary_wait": True}),
        "verify":  _ModeParams(temperature=0.5, max_tokens=4096,
            extra={"reasoning_effort": "high", "reasoning_summary": False}),
        "gaps":   _ModeParams(temperature=0.5, max_tokens=4096,
            extra={"reasoning_effort": "medium", "reasoning_summary": False}),
    })


PROFILES: dict[str, _Profile] = {
    "mercury": _Profile(
        name="Mercury 2",
        base_url="https://api.inceptionlabs.ai/v1",
        model="mercury-2",
        modes={
            "black_box": _ModeParams(temperature=0.5, max_tokens=4096,
                extra={"reasoning_effort": "medium", "reasoning_summary": False}),
            "white_box": _ModeParams(temperature=0.3, max_tokens=5120,
                extra={"reasoning_effort": "high", "reasoning_summary": True, "reasoning_summary_wait": True}),
            "analyze":  _ModeParams(temperature=0.3, max_tokens=4096,
                extra={"reasoning_effort": "high", "reasoning_summary": True, "reasoning_summary_wait": True, "response_format": {"type": "json_object"}}),
            "verify":  _ModeParams(temperature=0.5, max_tokens=4096,
                extra={"reasoning_effort": "high", "reasoning_summary": False}),
            "gaps":   _ModeParams(temperature=0.5, max_tokens=4096,
                extra={"reasoning_effort": "medium", "reasoning_summary": False}),
        },
    ),
    "gpt5-mini": _Profile(
        name="GPT-5 Mini",
        base_url="https://api.openai.com/v1",
        model="gpt-5-mini",
        modes={
            "black_box": _ModeParams(temperature=0.3, max_tokens=4096),
            "white_box": _ModeParams(temperature=0.3, max_tokens=8192),
            "analyze":  _ModeParams(temperature=0.3, max_tokens=8192,
                extra={"response_format": {"type": "json_object"}}),
            "verify":  _ModeParams(temperature=0.3, max_tokens=4096),
            "gaps":   _ModeParams(temperature=0.3, max_tokens=4096),
        },
    ),
    "gpt5": _Profile(
        name="GPT-5",
        base_url="https://api.openai.com/v1",
        model="gpt-5",
        modes={
            "black_box": _ModeParams(temperature=0.3, max_tokens=4096),
            "white_box": _ModeParams(temperature=0.3, max_tokens=16384),
            "analyze":  _ModeParams(temperature=0.3, max_tokens=16384,
                extra={"response_format": {"type": "json_object"}}),
            "verify":  _ModeParams(temperature=0.3, max_tokens=4096),
            "gaps":   _ModeParams(temperature=0.3, max_tokens=4096),
        },
    ),
    "claude-haiku": _Profile(
        name="Claude Haiku",
        base_url="https://api.anthropic.com/v1",
        model="claude-3.5-haiku-latest",
        modes={
            "black_box": _ModeParams(temperature=0.3, max_tokens=4096),
            "white_box": _ModeParams(temperature=0.3, max_tokens=8192),
            "analyze":  _ModeParams(temperature=0.3, max_tokens=8192),
            "verify":  _ModeParams(temperature=0.3, max_tokens=4096),
            "gaps":   _ModeParams(temperature=0.3, max_tokens=4096),
        },
    ),
    "claude-opus": _Profile(
        name="Claude Opus",
        base_url="https://api.anthropic.com/v1",
        model="claude-opus-4-5-latest",
        modes={
            "black_box": _ModeParams(temperature=0.3, max_tokens=4096),
            "white_box": _ModeParams(temperature=0.3, max_tokens=16384),
            "analyze":  _ModeParams(temperature=0.3, max_tokens=16384),
            "verify":  _ModeParams(temperature=0.3, max_tokens=4096),
            "gaps":   _ModeParams(temperature=0.3, max_tokens=4096),
        },
    ),
    "openrouter": _Profile(
        name="OpenRouter (auto)",
        base_url="https://openrouter.ai/api/v1",
        model="openrouter/auto",
        modes={
            "black_box": _ModeParams(temperature=0.3, max_tokens=4096),
            "white_box": _ModeParams(temperature=0.3, max_tokens=8192),
            "analyze":  _ModeParams(temperature=0.3, max_tokens=8192),
            "verify":  _ModeParams(temperature=0.3, max_tokens=4096),
            "gaps":   _ModeParams(temperature=0.3, max_tokens=4096),
        },
    ),
    "minimax": _Profile(
        name="Minimax M3 (NVIDIA)",
        base_url="https://integrate.api.nvidia.com/v1",
        model="minimaxai/minimax-m3",
        modes={
            "black_box": _ModeParams(temperature=0.1, max_tokens=4096,
                extra={"thinking": {"type": "enabled"}}),
            "white_box": _ModeParams(temperature=0.1, max_tokens=16384,
                extra={"thinking": {"type": "enabled"}}),
            "analyze":  _ModeParams(temperature=0.1, max_tokens=8192,
                extra={"thinking": {"type": "enabled"}, "response_format": {"type": "json_object"}}),
            "verify":  _ModeParams(temperature=0.1, max_tokens=4096,
                extra={"thinking": {"type": "enabled"}}),
            "gaps":   _ModeParams(temperature=0.1, max_tokens=4096,
                extra={"thinking": {"type": "enabled"}}),
        },
    ),
}

_DEFAULT_PROFILE = os.getenv("LLM_PROVIDER", "custom" if os.getenv("LLM_MODEL") else "mercury")


def _get_profile(name: Optional[str] = None) -> _Profile:
    key = name or _DEFAULT_PROFILE
    p = PROFILES.get(key)
    if p:
        return p
    if key == "custom":
        model = os.getenv("LLM_MODEL") or "unknown"
        raw_base = os.getenv("LLM_BASE_URL") or "https://integrate.api.nvidia.com/v1"
        base_url = raw_base.rstrip("/")
        suffix = "/chat/completions"
        if base_url.endswith(suffix):
            base_url = base_url[:-len(suffix)]
        logger.info("Building custom profile: model=%s base_url=%s", model, base_url)
        return _Profile(
            name=f"Custom ({model})",
            base_url=base_url,
            model=model,
            modes={
                "black_box": _ModeParams(temperature=0.1, max_tokens=4096,
                    extra={"thinking": {"type": "enabled"}}),
                "white_box": _ModeParams(temperature=0.1, max_tokens=16384,
                    extra={"thinking": {"type": "enabled"}}),
                "analyze":  _ModeParams(temperature=0.1, max_tokens=8192,
                    extra={"thinking": {"type": "enabled"}, "response_format": {"type": "json_object"}}),
                "verify":  _ModeParams(temperature=0.1, max_tokens=4096,
                    extra={"thinking": {"type": "enabled"}}),
                "gaps":   _ModeParams(temperature=0.1, max_tokens=4096,
                    extra={"thinking": {"type": "enabled"}}),
            },
        )
    logger.warning("Unknown LLM profile %r, falling back to mercury", key)
    return PROFILES["mercury"]


# ── LLM client ───────────────────────────────────────────────────────────────

class LLM:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> None:
        # Resolve profile (may come from LLM_PROVIDER env or LLM_MODEL → "custom")
        self.profile = _get_profile(profile)
        # Env var overrides take priority over profile values
        self.model = os.getenv("LLM_MODEL") or model or self.profile.model
        raw = os.getenv("LLM_BASE_URL") or base_url or self.profile.base_url
        self.base_url = raw.rstrip("/")
        # Allow LLM_BASE_URL to contain the full path; strip /chat/completions
        # since callers append it.
        suffix = "/chat/completions"
        if self.base_url.endswith(suffix):
            self.base_url = self.base_url[:-len(suffix)]
        self.api_key = api_key or get_secret("LLM_API_KEY") or get_secret("INCEPTION_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("INCEPTION_API_KEY")
        if not self.api_key:
            raise RuntimeError("No LLM API key found. Set LLM_API_KEY or INCEPTION_API_KEY in .env")
        self._cb = CircuitBreaker(name="llm", failure_threshold=5, recovery_timeout_s=30.0)

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def is_available(self) -> bool:
        return self._cb.is_available()

    def _params_for(self, mode: str, override_temperature: Optional[float] = None, override_max_tokens: Optional[int] = None) -> dict:
        p = self.profile.modes.get(mode, self.profile.modes["black_box"])
        params = {
            "temperature": override_temperature if override_temperature is not None else p.temperature,
            "max_tokens": override_max_tokens or p.max_tokens,
        }
        params.update(p.extra)
        return params

    def _build_payload(self, prompt: str, system: Optional[str], mode: str,
                       temperature: Optional[float], max_tokens: Optional[int],
                       **overrides) -> dict:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        params = self._params_for(mode, temperature, max_tokens)
        # Apply any per-call overrides
        # Supported overrides: model (switch model name for this call)
        call_model = overrides.pop("model", self.model)
        params.update(overrides)

        payload = {
            "model": call_model,
            "messages": messages,
            **params,
        }
        logger.info("LLM.%s: model=%s mode=%s temp=%s max_tokens=%s extra=%s prompt_len=%d",
                     mode, self.model, mode,
                     params.get("temperature"), params.get("max_tokens"),
                     {k: v for k, v in params.items() if k not in ("temperature", "max_tokens")},
                     len(prompt))
        return payload

    async def chat(self, prompt: str, system: Optional[str] = None, mode: str = "black_box",
                   temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                   **overrides) -> str:
        return await self._cb.call(
            self._do_chat_with_retry, prompt, system=system, mode=mode,
            temperature=temperature, max_tokens=max_tokens, **overrides
        )

    @retry_with_backoff(max_retries=2, base_delay_s=0.5, retryable_exceptions=(httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError))
    async def _do_chat_with_retry(self, prompt: str, system: Optional[str] = None, mode: str = "black_box",
                                   temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                                   **overrides) -> str:
        return await self._do_chat(prompt, system=system, mode=mode,
                                    temperature=temperature, max_tokens=max_tokens, **overrides)

    async def _do_chat(self, prompt: str, system: Optional[str] = None, mode: str = "black_box",
                        temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                        **overrides) -> str:
        payload = self._build_payload(prompt, system, mode, temperature, max_tokens, **overrides)
        timeout = 300.0
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers)
            if r.status_code != 200:
                body = (await r.aread())[:500].decode("utf-8", "ignore")
                logger.error("LLM.chat: HTTP %d: %s", r.status_code, body)
                raise LLMError(f"{self.profile.name} {r.status_code}: {body}")
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            logger.info("LLM.chat: response length=%d, preview=%.200s", len(content), content)
            return content

    async def astream(self, prompt: str, system: Optional[str] = None, mode: str = "black_box",
                       temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                       **overrides) -> AsyncIterator[str]:
        if not self._cb.is_available():
            logger.warning("LLM circuit is OPEN — streaming unavailable")
            yield "The LLM service is temporarily unavailable. Please try again shortly."
            return
        payload = self._build_payload(prompt, system, mode, temperature, max_tokens, **overrides)
        payload["stream"] = True
        timeout = 600.0
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers) as r:
                    if r.status_code != 200:
                        body = await r.aread()
                        raise LLMError(f"{self.profile.name} {r.status_code}: {body[:500].decode('utf-8', 'ignore')}")
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        chunk = line[6:].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            data = json.loads(chunk)
                            delta = data["choices"][0].get("delta", {}).get("content")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except Exception as e:
            logger.error("LLM.astream failed: %s", e)
            await self._cb.call(lambda: (_ for _ in ()).throw(e))
            raise


_llm: Optional[LLM] = None


def get_llm(profile: Optional[str] = None) -> LLM:
    global _llm
    if _llm is None or profile:
        _llm = LLM(profile=profile)
    return _llm


def list_profiles() -> list[dict]:
    out = [{"key": k, "name": p.name, "model": p.model, "base_url": p.base_url} for k, p in PROFILES.items()]
    # Include env-based custom profile if configured
    if os.getenv("LLM_MODEL"):
        out.append({
            "key": "custom",
            "name": f"Custom ({os.getenv('LLM_MODEL')})",
            "model": os.getenv("LLM_MODEL"),
            "base_url": os.getenv("LLM_BASE_URL") or "(env LLM_BASE_URL not set)",
        })
    return out
