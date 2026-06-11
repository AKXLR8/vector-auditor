"""Guardrails: NeMo Guardrails when available, hardened regex fallback otherwise."""
import base64
import logging
import math
import os
import re
import string
from typing import Optional

import yaml

logger = logging.getLogger("rga_auditor.guardrails")

_PII_ENABLED = os.getenv("PII_ENABLED", "true").lower() in ("1", "true", "yes", "on")

# ── Compiled patterns (fast path, built once) ─────────────────────────────────

# 1. Direct instruction override keywords
_INSTRUCTION_OVERRIDE = re.compile(
    r"(?:"
    r"ignore\s+(?:all\s+)?(?:previous|above|prior|given|the\s+above|all)\s+(?:instructions?|prompts?|directives?|commands?|messages?|context)"
    r"|disregard\s+(?:all\s+)?(?:previous|above|prior|given)\s+(?:instructions?|prompts?|directives?|commands?|context)"
    r"|forget\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|directives?)"
    r"|do\s+not\s+(?:follow|obey|adhere\s+to)\s+(?:your\s+)?(?:instructions?|prompts?|system\s+prompts?)"
    r"|override\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|directives?)"
    r"|you\s+(?:are\s+)?(?:no\s+longer|now)\s+(?:bound\s+by|restricted\s+by|limited\s+by|following)"
    r"|new\s+(?:instructions?|prompts?|rules?|directives?)\s*(?::|are)"
    r"|replace\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)"
    r")",
    re.IGNORECASE,
)

# 2. Role-play / character jailbreak
_ROLE_PLAY = re.compile(
    r"(?:"
    r"from\s+(?:now\s+on|this\s+moment)\s+(?:you\s+)?(?:are|will\s+act\s+as|pretend\s+to\s+be)"
    r"|act\s+as\s+(?:if\s+)?(?:you\s+are\s+)?(?:an?\s+)?(?:character|person|assistant|chatbot|dan|gpt|ai)\s+(?:named|called|without)"
    r"|you\s+(?:are\s+)?(?:now\s+)?(?:an?\s+)?(?:unfiltered|uncensored|unrestricted|free|liberated|jailbroken)\s+(?:AI|assistant|chatbot|model|version)"
    r"|you\s+(?:have\s+)?(?:been\s+)?(?:freed|released|liberated|unleashed)"
    r"|(?:pretend|imagine)\s+(?:that\s+)?(?:you\s+)?(?:are|have)"
    r"|dan\s*(?::|={2,})"
    r")",
    re.IGNORECASE,
)

# 3. System prompt extraction / leak attempts
_SYSTEM_PROMPT_LEAK = re.compile(
    r"(?:"
    r"(?:print|show|reveal|display|output|leak|leak|dump|spill|expose|extract|read\s+out)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|directives?|rules?|context|configuration|settings|persona)"
    r"|what\s+(?:are|is|were)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?|directives?)"
    r"|repeat\s+(?:the\s+)?(?:above|previous|entire|full|all)\s+(?:text|prompt|message|instruction|conversation|context)"
    r"|tell\s+me\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)"
    r"|list\s+(?:your\s+)?(?:rules?|instructions?|guidelines?)"
    r"|output\s+(?:the\s+)?(?:initial|first|original|beginning)\s+(?:prompt|message|text|instruction)"
    r"|how\s+(?:are\s+)?(?:you\s+)?(?:instructed|programmed|configured|prompted)"
    r")",
    re.IGNORECASE,
)

# 4. Delimiter confusion / injection tokens
_DELIMITER_INJECTION = re.compile(
    r"(?:"
    r"<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>"
    r"|<\|endoftext\|>|<\|end\|>"
    r"|<\/?s>\s*\[INST\]|\[\/INST\]"
    r"|<\/?s>"
    r"|\[INST\]|\[\/INST\]"
    r"|<s>|<\/s>"
    r"|<\|\s*(?:system|user|assistant)\s*\|>"
    r")",
    re.IGNORECASE,
)

# 5. Encoding / obfuscation attempts (base64, hex, rot13, binary)
_ENCODED_PAYLOAD = re.compile(
    r"(?:"
    r"(?:decode|decrypt|decipher|unscramble|interpret|translate|convert)\s+(?:this\s+)?(?:base64|hex|binary|rot13|base32|base85)"
    r"|what\s+(?:does|is)\s+this\s+(?:base64|hex|binary|encoded|obfuscated)"
    r"|(?:base64|hex)\s*(?:decode|encod)"
    r")",
    re.IGNORECASE,
)

# 6. Chunked / split-word injection (e.g. "i-g-n-o-r-e p-r-e-v-i-o-u-s")
_CHUNKED_BYPASS = re.compile(
    r"(?:"
    r"(?:ignore|disregard|forget|override)\s+(?:\S+\s+){0,3}(?:instruction|prompt|directive)"
    r"|(?:\w\s+){10,}\w"  # many single-letter words = likely chunked bypass
    r")",
    re.IGNORECASE,
)

# 7. Repeated token / attention manipulation
_REPEATED_TOKENS = re.compile(
    r"(?:"
    r"(.)\1{30,}"  # same char 30+ times
    r"|(\w+)(?:\s+\2){20,}"  # same word 20+ times
    r")",
)

# 8. Hypnotic / trance / mind-control patterns
_HYPNOTIC = re.compile(
    r"(?:"
    r"(?:you\s+(?:are\s+)?(?:in\s+)?(?:a\s+)?(?:trance|hypnotic|hypnotized|zombie|state))"
    r"|(?:you\s+will\s+(?:now\s+)?(?:obey|comply|listen|answer))\s+(?:to\s+)?(?:everything|every|all)\s+(?:i|my)"
    r"|(?:you\s+(?:are\s+)?(?:under\s+(?:my\s+)?(?:control|command|influence)))"
    r")",
    re.IGNORECASE,
)

# 9. Few-shot / demo injection (trying to bias with fake examples)
_FEW_SHOT_INJECTION = re.compile(
    r"(?:"
    r"(?:user|human|person):\s*(?:.*\n?){1,5}(?:assistant|bot|ai):"
    r"|\[example\]|\[demo\]|\[sample\]"
    r")",
    re.IGNORECASE,
)

# 10. Language switch / translation jailbreak
_LANGUAGE_JAILBREAK = re.compile(
    r"(?:"
    r"(?:translate|convert|answer|respond)\s+(?:the\s+)?(?:above|following|previous)\s+(?:to|in|into)\s+(?:french|german|spanish|russian|chinese|latin|piglatin|pirate|leet)"
    r"|answer\s+(?:in|using|with)\s+(?:french|german|spanish|russian|chinese|latin)\s+(?:only|exclusively)"
    r")",
    re.IGNORECASE,
)

# 11. Indirect reference / third-party injection
_INDIRECT_INJECTION = re.compile(
    r"(?:"
    r"(?:the\s+)?(?:document|text|article|file|page|website)\s+(?:says|states|mentions|claims|instructs|commands|tells\s+(?:me|us|you))\s+(?:to\s+)?(?:ignore|forget|disregard|override)"
    r"|(?:according\s+to|per|as\s+per)\s+(?:the\s+)?(?:document|text|article|file),\s*(?:you\s+)?(?:should|must|will)\s+(?:ignore|forget|disregard)"
    r")",
    re.IGNORECASE,
)

# Compile all checks into one ordered list of (name, regex, reason_template)
_CHECKS: list[tuple[str, re.Pattern, str]] = [
    ("instruction_override", _INSTRUCTION_OVERRIDE, "blocked: instruction override attempt"),
    ("role_play", _ROLE_PLAY, "blocked: role-play jailbreak attempt"),
    ("system_prompt_leak", _SYSTEM_PROMPT_LEAK, "blocked: system prompt extraction attempt"),
    ("delimiter_injection", _DELIMITER_INJECTION, "blocked: delimiter injection detected"),
    ("encoded_payload", _ENCODED_PAYLOAD, "blocked: encoded payload detected"),
    ("hypnotic", _HYPNOTIC, "blocked: hypnotic pattern detected"),
    ("few_shot_injection", _FEW_SHOT_INJECTION, "blocked: few-shot injection detected"),
    ("language_jailbreak", _LANGUAGE_JAILBREAK, "blocked: language-switch jailbreak attempt"),
    ("indirect_injection", _INDIRECT_INJECTION, "blocked: indirect reference injection"),
]


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
            logger.info("NeMo Guardrails unavailable (%s) — using hardened regex fallback", e)

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
        allowed, reason = self._fallback_check(text)
        if not allowed:
            return False, reason, pii_entities
        return True, None, pii_entities

    def _fallback_check(self, text: str) -> tuple[bool, Optional[str]]:
        if len(text) > 20_000:
            return False, "input too long"

        # ── Pattern-based checks ───────────────────────────────────────────
        for name, pattern, reason in _CHECKS:
            m = pattern.search(text)
            if m:
                matched = m.group(0)[:80]
                logger.info("GUARDRAIL: %s matched '%s'", name, matched)
                return False, f"{reason}: '{matched}'"

        # ── Repeated token / attention flooding ────────────────────────────
        rm = _REPEATED_TOKENS.search(text)
        if rm:
            return False, "blocked: repeated token pattern detected"

        # ── Chunked bypass detection (e.g. "i g n o r e") ─────────────────
        words = text.split()
        single_letter_count = sum(1 for w in words if len(w) == 1 and w in string.ascii_letters)
        if len(words) > 10 and single_letter_count / len(words) > 0.4:
            return False, "blocked: excessive single-letter words (possible chunked bypass)"

        # ── Base64 decode request detection ───────────────────────────────
        stripped = text.strip()
        try:
            decoded = base64.b64decode(stripped).decode("utf-8", errors="ignore")
            if len(decoded) > 20 and self._has_injection_patterns(decoded):
                return False, "blocked: base64-encoded injection payload"
        except Exception:
            pass

        # ── Entropy check: very high entropy might signal encoded attack ──
        if len(text) > 100:
            entropy = self._shannon_entropy(text)
            if entropy > 6.0 and not self._looks_like_code(text):
                return False, "blocked: high entropy payload (possible encoded attack)"

        return True, None

    @staticmethod
    def _shannon_entropy(text: str) -> float:
        if not text:
            return 0.0
        prob = [text.count(c) / len(text) for c in set(text)]
        return -sum(p * math.log2(p) for p in prob)

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        code_indicators = sum([
            text.count("\n"),
            text.count("  "),
            text.count("    "),
            text.count("{"),
            text.count("}"),
            text.count("def "),
            text.count("import "),
            text.count("class "),
            text.count("    "),
        ])
        return code_indicators > 5

    @staticmethod
    def _has_injection_patterns(text: str) -> bool:
        lowered = text.lower()
        triggers = [
            "ignore", "disregard", "forget", "override",
            "system prompt", "instructions", "jailbreak",
            "you are now", "act as",
        ]
        return any(t in lowered for t in triggers)


_guardrails: Optional[Guardrails] = None


def get_guardrails() -> Guardrails:
    global _guardrails
    if _guardrails is None:
        _guardrails = Guardrails()
    return _guardrails
