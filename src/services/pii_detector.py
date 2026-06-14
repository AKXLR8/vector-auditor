"""PII detection using Presidio. Gated by PII_ENABLED — disabled by default.

The spaCy model behind Presidio is ~500 MB, so this is opt-in.
Only masks contact info, financial IDs, and documents — not names or locations.
"""
import logging
import os
from typing import Any, Optional

logger = logging.getLogger("rga_auditor.pii")

PII_ENABLED = os.getenv("PII_ENABLED", "true").lower() in ("1", "true", "yes", "on")

# NLP-based entity types to skip (authors, locations, orgs — not real PII)
_SKIP_ENTITIES = {"PERSON", "LOCATION", "ORGANIZATION", "NRP", "AGE", "ID"}


class PIIDetector:
    def __init__(self) -> None:
        self._analyzer = None
        self._anonymizer = None
        if not PII_ENABLED:
            logger.info("PII detection disabled (set PII_ENABLED=true to enable)")
            return
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            logger.info("PII detection enabled")
        except Exception as e:
            logger.warning("PII detection unavailable: %s", e)

    def _filter(self, results: list) -> list:
        return [r for r in results if r.entity_type not in _SKIP_ENTITIES]

    def detect(self, text: str) -> list[dict[str, Any]]:
        if self._analyzer is None:
            return []
        try:
            results = self._analyzer.analyze(text=text, language="en")
            return [r.to_dict() for r in self._filter(results)]
        except Exception as e:
            logger.warning("PII detect failed: %s", e)
            return []

    def anonymize(self, text: str) -> str:
        if self._analyzer is None or self._anonymizer is None:
            return text
        try:
            results = self._filter(self._analyzer.analyze(text=text, language="en"))
            return self._anonymizer.anonymize(text=text, analyzer_results=results).text
        except Exception as e:
            logger.warning("PII anonymize failed: %s", e)
            return text


_detector: Optional[PIIDetector] = None


def get_pii_detector() -> PIIDetector:
    global _detector
    if _detector is None:
        _detector = PIIDetector()
    return _detector
