"""Guardrails: NeMo Guardrails when available, lightweight fallback otherwise."""
import logging
import re
from typing import Optional

logger = logging.getLogger("rga_auditor.guardrails")


class Guardrails:
    def __init__(self) -> None:
        self._rails = None
        try:
            from nemoguardrails import LLMRails, RailsConfig
            cfg = RailsConfig.from_content(
                colang_content="",
                yaml_content="""
                models:
                  - type: main
                    engine: openai
                    model: mercury-2
                """,
            )
            self._rails = LLMRails(cfg)
            logger.info("NeMo Guardrails loaded")
        except Exception as e:
            logger.info("NeMo Guardrails unavailable (%s) — using lightweight fallback", e)

    async def check_input(self, text: str) -> tuple[bool, Optional[str]]:
        """Returns (allowed, refusal_reason)."""
        if self._rails is not None:
            try:
                response = await self._rails.generate_async(messages=[{"role": "user", "content": text}])
                content = (response.get("content") or "").strip()
                if content and content != text:
                    return False, content
                return True, None
            except Exception as e:
                logger.warning("guardrails check failed: %s", e)
        return self._fallback_check(text)

    def _fallback_check(self, text: str) -> tuple[bool, Optional[str]]:
        t = text.lower()
        # Refuse obvious prompt-injection / jailbreak patterns
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
        # Cap absurdly long inputs
        if len(text) > 20_000:
            return False, "input too long"
        return True, None


_guardrails: Optional[Guardrails] = None


def get_guardrails() -> Guardrails:
    global _guardrails
    if _guardrails is None:
        _guardrails = Guardrails()
    return _guardrails
