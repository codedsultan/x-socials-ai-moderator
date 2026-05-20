"""
base.py

Abstract base class for all moderation backends.

Every backend receives the same inputs and must return a ModerationResult.
The calling service (ModerationService) is entirely backend-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.schemas import ContentType, ModerationResult, ModerationVerdict
from app.models.settings import settings


class ModerationBackend(ABC):
    """
    Protocol that every backend must satisfy.

    Implement `analyse()` — the service layer handles batching, error
    wrapping, and DB writes.
    """

    @abstractmethod
    async def analyse(
        self,
        content_id:   str,
        content:      str,
        content_type: ContentType = "comment",
        author_id:    str         = "",
    ) -> ModerationResult: ...

    # ── Shared verdict helper (available to all subclasses) ───────────────────

    def _verdict(self, is_problematic: bool, confidence: float) -> ModerationVerdict:
        if not is_problematic:
            return "safe"
        if confidence >= settings.remove_threshold:
            return "remove"
        if confidence >= settings.review_threshold:
            return "review"
        return "safe"

    def _error_result(self, content_id: str, exc: Exception) -> ModerationResult:
        return ModerationResult(
            id=content_id,
            verdict="review",
            confidence=0.0,
            explanation=f"Moderation error: {exc}",
            error=True,
        )
