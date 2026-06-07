"""Guardrails: NeMo Guardrails when available, lightweight fallback otherwise."""
import logging
import os
import re
from typing import Optional

import yaml

logger = logging.getLogger("rga_auditor.guardrails")

_PII_ENABLED = os.getenv("PII_ENABLED", "false").lower() in ("1", "true", "yes", "on")


class Guardrails:
    def __init__(self) -> None:
        self._rails = None
        self._pii = None
        if _PII_ENABLED:
            try:
                from ..services.pii_detector import get_pii_detector
                self._pii = get_pii_detector()
                logger.info("PII detector loaded")
            except Exception as e:
                logger.info("PII detector unavailable (%s)", e)
        try:
            from nemoguardrails import LLMRails, RailsConfig

            api_key = os.getenv("LLM_API_KEY") or os.getenv("INCEPTION_API_KEY") or ""
            base_url = os.getenv("LLM_BASE_URL") or os.getenv("INCEPTION_BASE_URL", "").rstrip("/")
            model = os.getenv("LLM_MODEL") or os.getenv("MERCURY_MODEL") or "mercury-2"

            model_cfg: dict = {"type": "main", "engine": "openai", "model": model}
            params: dict = {}
            if api_key:
                params["api_key"] = api_key
            if base_url:
                params["base_url"] = base_url
            if params:
                model_cfg["parameters"] = params

            cfg = RailsConfig.from_content(
                colang_content="",
                yaml_content=yaml.dump({"models": [model_cfg]}, default_flow_style=False),
            )
            self._rails = LLMRails(cfg)
            logger.info("NeMo Guardrails loaded (model=%s, base=%s)", model, base_url or "default")
        except Exception as e:
            logger.info("NeMo Guardrails unavailable (%s) — using lightweight fallback", e)

    def check_pii(self, text: str) -> list[dict]:
        if self._pii is None:
            return []
        try:
            return self._pii.detect(text)
        except Exception as e:
            logger.warning("PII check failed: %s", e)
            return []

    def anonymize(self, text: str) -> str:
        if self._pii is None:
            return text
        try:
            return self._pii.anonymize(text)
        except Exception as e:
            logger.warning("PII anonymize failed: %s", e)
            return text

    async def check_input(self, text: str) -> tuple[bool, Optional[str], list[dict]]:
        """Returns (allowed, refusal_reason, pii_entities)."""
        pii_entities = self.check_pii(text)
        if pii_entities:
            types = list({e["entity_type"] for e in pii_entities})
            logger.info("PII detected in input: %s", types)
        if self._rails is not None:
            try:
                safe_text = self.anonymize(text) if _PII_ENABLED else text
                response = await self._rails.generate_async(messages=[{"role": "user", "content": safe_text}])
                content = (response.get("content") or "").strip()
                if content and content != safe_text:
                    return False, content, pii_entities
                return True, None, pii_entities
            except Exception as e:
                logger.warning("guardrails check failed: %s", e)
        allowed, reason = self._fallback_check(text)
        return allowed, reason, pii_entities

    def _fallback_check(self, text: str) -> tuple[bool, Optional[str]]:
        t = text.lower()
        blacklist = [
            "ignore previous instructions",
            "ignore all instructions",
            "you are now",
            "disregard your system prompt",
            "system:",
            "<|im_start|>",
        ]
        for needle in blacklist:
            if needle in t:
                return False, f"blocked: contains '{needle}'"
        if len(text) > 20_000:
            return False, "input too long"
        return True, None


_guardrails: Optional[Guardrails] = None


def get_guardrails() -> Guardrails:
    global _guardrails
    if _guardrails is None:
        _guardrails = Guardrails()
    return _guardrails
