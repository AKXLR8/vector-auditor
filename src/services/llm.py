"""LLM client for OpenAI-compatible APIs."""
import json
import logging
import os
from typing import AsyncIterator, Optional

import httpx

from .secrets import get_secret

logger = logging.getLogger("rga_auditor.llm")

DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("INCEPTION_BASE_URL") or "https://api.inceptionlabs.ai/v1"
DEFAULT_MODEL = os.getenv("LLM_MODEL") or os.getenv("MERCURY_MODEL") or "mercury-2"


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        api_key = api_key or get_secret("LLM_API_KEY") or get_secret("INCEPTION_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("INCEPTION_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No LLM API key found. Set LLM_API_KEY or INCEPTION_API_KEY in .env"
            )
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.max_tokens = int(max_tokens) if max_tokens is not None else int(os.getenv("LLM_MAX_TOKENS", "2048"))
        self.temperature = float(temperature if temperature is not None else os.getenv("LLM_TEMPERATURE", "0"))

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers)
                if r.status_code != 200:
                    body = (await r.aread())[:500].decode("utf-8", "ignore")
                    raise LLMError(f"Inception {r.status_code}: {body}")
                data = r.json()
                return data["choices"][0]["message"]["content"]
        except httpx.ConnectError as e:
            raise LLMError(f"Cannot reach {self.base_url}: {e}. Check INCEPTION_API_KEY and network.") from e

    async def astream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers
            ) as r:
                if r.status_code != 200:
                    body = await r.aread()
                    raise LLMError(f"Inception {r.status_code}: {body[:500].decode('utf-8', 'ignore')}")
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


_llm: Optional[LLM] = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm
