"""
rule_based.py

Local ML moderation backend — no API calls, no cost, works fully offline.

Uses `detoxify` (Unitary), a transformer-based toxicity classifier that
returns per-category confidence scores in 0.0–1.0.

Categories produced:
  toxicity, severe_toxicity, obscene, threat, insult, identity_attack

On first use the model weights (~200 MB) are downloaded from HuggingFace
and cached in ~/.cache/huggingface (or $TRANSFORMERS_CACHE).  Subsequent
startups load from cache and are fast.

Fallback: if detoxify or torch is not installed the backend logs a warning
and always returns verdict="safe" with confidence=0.0 so the rest of the
pipeline keeps running (useful in CI / lightweight deployments).
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any

from app.models.schemas import ContentType, ModerationResult
from app.services.backends.base import ModerationBackend

logger = logging.getLogger(__name__)

# Map detoxify keys → our schema category names
_CATEGORY_MAP: dict[str, str] = {
    "toxicity":           "toxicity",
    "severe_toxicity":    "hate_speech",
    "obscene":            "adult_content",
    "threat":             "violence",
    "insult":             "harassment",
    "identity_attack":    "hate_speech",
}


@lru_cache(maxsize=1)
def _load_model() -> Any | None:
    """Load detoxify once and cache it for the process lifetime."""
    try:
        from detoxify import Detoxify  # type: ignore
        model = Detoxify("original")
        logger.info("RuleBasedBackend: detoxify model loaded")
        return model
    except ImportError:
        logger.warning(
            "RuleBasedBackend: detoxify / torch not installed — "
            "backend will always return safe. "
            "Install with: pip install detoxify"
        )
        return None


class RuleBasedBackend(ModerationBackend):
    """
    Fast, offline, cost-free moderation via detoxify.

    Because detoxify's predict() is CPU/GPU-bound (not I/O-bound) we run it
    in a thread pool so it doesn't block the event loop.
    """

    def __init__(self) -> None:
        self._model = _load_model()

    async def analyse(
        self,
        content_id:   str,
        content:      str,
        content_type: ContentType = "comment",
        author_id:    str         = "",
    ) -> ModerationResult:
        try:
            if self._model is None:
                # detoxify not available — pass through as safe
                return ModerationResult(
                    id=content_id,
                    verdict="safe",
                    confidence=0.0,
                    explanation="Rule-based backend unavailable (detoxify not installed).",
                )

            # Run the CPU-bound inference in a thread pool
            loop   = asyncio.get_event_loop()
            scores = await loop.run_in_executor(None, self._model.predict, content)

            # scores = {"toxicity": 0.03, "severe_toxicity": 0.00, ...}
            confidence    = float(max(scores.values()))
            is_problematic = confidence >= 0.30  # lower bar — let the verdict thresholds decide

            # Collect categories where score exceeds a meaningful signal threshold
            categories: list[str] = []
            seen: set[str] = set()
            for key, score in scores.items():
                if score >= 0.25:
                    mapped = _CATEGORY_MAP.get(key, key)
                    if mapped not in seen:
                        categories.append(mapped)
                        seen.add(mapped)

            verdict = self._verdict(is_problematic, confidence)

            return ModerationResult(
                id=content_id,
                verdict=verdict,
                confidence=round(confidence, 4),
                categories=categories,
                explanation=self._explain(confidence, categories),
                flaggedPhrases=[],  # detoxify doesn't do phrase-level attribution
            )

        except Exception as exc:
            logger.exception("RuleBasedBackend error for %s", content_id)
            return self._error_result(content_id, exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _explain(confidence: float, categories: list[str]) -> str:
        if not categories:
            return "No policy violations detected by local classifier."
        cat_str = ", ".join(categories)
        pct     = int(round(confidence * 100))
        return f"Local classifier flagged: {cat_str} (confidence {pct}%)."
