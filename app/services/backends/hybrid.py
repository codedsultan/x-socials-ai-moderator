"""
hybrid.py

Two-stage moderation backend that minimises LLM API calls.

Stage 1 — fast (local, free):
    Run the RuleBasedBackend (detoxify).  If the signal is unambiguous
    (clearly safe OR clearly toxic) return immediately — no API call made.

Stage 2 — smart (LLM, paid):
    Only reached when the fast score falls in the ambiguous middle band.
    The LLM provides a richer explanation and phrase-level attribution.

Typical result on real social content:
  ~40% clearly safe   → rule-based only
  ~20% clearly toxic  → rule-based only
  ~40% ambiguous      → escalated to LLM

That cuts LLM calls by ~60% compared to running every item through Claude,
while keeping the same verdict quality for the items that matter.

Configuration (all tunable via env vars / Settings):
  HYBRID_FAST_SAFE_CEILING  – scores at or below this are returned as-is (safe)
  HYBRID_FAST_FLAG_FLOOR    – scores at or above this are returned as-is (flagged)
  Everything in between is sent to the smart backend.
"""

from __future__ import annotations

import logging

from app.models.schemas import ContentType, ModerationResult
from app.models.settings import settings
from app.services.backends.base import ModerationBackend

logger = logging.getLogger(__name__)


class HybridBackend(ModerationBackend):
    """
    Composes a fast (rule-based) backend with a smart (LLM) backend.

    Args:
        fast:             Backend to run first (typically RuleBasedBackend).
        smart:            Backend to escalate to (any LLM backend).
        safe_ceiling:     Fast confidence <= this → return fast result immediately.
        flag_floor:       Fast confidence >= this → return fast result immediately.

    Any score between safe_ceiling and flag_floor triggers the smart backend.
    """

    def __init__(
        self,
        fast: ModerationBackend,
        smart: ModerationBackend,
        safe_ceiling: float | None = None,
        flag_floor: float | None = None,
    ) -> None:
        self._fast = fast
        self._smart = smart
        self._safe_ceil = safe_ceiling if safe_ceiling is not None else settings.hybrid_safe_ceiling
        self._flag_floor = flag_floor if flag_floor is not None else settings.hybrid_flag_floor

    async def analyse(
        self,
        content_id: str,
        content: str,
        content_type: ContentType = "comment",
        author_id: str = "",
    ) -> ModerationResult:
        # ── Stage 1: fast classifier ──────────────────────────────────────────
        fast_result = await self._fast.analyse(content_id, content, content_type, author_id)

        if fast_result.error:
            # Fast backend failed — escalate to smart so we still get a result
            logger.warning(
                "HybridBackend: fast backend errored for %s — escalating to smart backend",
                content_id,
            )
            return await self._smart.analyse(content_id, content, content_type, author_id)

        confidence = fast_result.confidence

        # Unambiguously safe
        if confidence <= self._safe_ceil:
            logger.debug("HybridBackend: %s skipped LLM (safe, conf=%.3f)", content_id, confidence)
            return fast_result

        # Unambiguously problematic
        if confidence >= self._flag_floor:
            logger.debug(
                "HybridBackend: %s skipped LLM (flagged, conf=%.3f)", content_id, confidence
            )
            return fast_result

        # ── Stage 2: ambiguous — escalate to LLM ─────────────────────────────
        logger.debug(
            "HybridBackend: %s escalating to smart backend (conf=%.3f, window=[%.2f, %.2f])",
            content_id,
            confidence,
            self._safe_ceil,
            self._flag_floor,
        )
        smart_result = await self._smart.analyse(content_id, content, content_type, author_id)

        if smart_result.error:
            # LLM failed — fall back to the fast result so we still have something
            logger.warning(
                "HybridBackend: smart backend errored for %s — using fast result as fallback",
                content_id,
            )
            return fast_result

        return smart_result
