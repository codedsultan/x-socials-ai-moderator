"""
anthropic_backend.py

Moderation backend backed by Anthropic's Claude API.

This is the original Claude logic, refactored into the ModerationBackend
protocol so it's swappable like any other backend.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from app.models.schemas import ContentType, ModerationResult
from app.models.settings import settings
from app.services.backends._prompts import COMMENT_PROMPT, POST_PROMPT
from app.services.backends.base import ModerationBackend

logger = logging.getLogger(__name__)


class AnthropicBackend(ModerationBackend):
    """
    Uses Claude via the official Anthropic SDK.

    model defaults to settings.moderator_model (claude-haiku-3-5-20251001).
    Pass a different model string at construction time for higher-quality analysis.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or settings.moderator_model
        self._client: anthropic.AsyncAnthropic | None = None

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._client

    async def analyse(
        self,
        content_id: str,
        content: str,
        content_type: ContentType = "comment",
        author_id: str = "",
    ) -> ModerationResult:
        try:
            raw = await self._call_claude(content, content_type)
            return self._parse(content_id, raw)
        except Exception as exc:
            logger.exception("AnthropicBackend error for %s", content_id)
            return self._error_result(content_id, exc)

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _call_claude(self, content: str, content_type: ContentType) -> dict[str, Any]:
        system = POST_PROMPT if content_type == "post" else COMMENT_PROMPT
        label = "Post" if content_type == "post" else "Comment"
        message = f"{label} to moderate:\n\n{content}"

        response = await self.client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": message}],
        )

        first = response.content[0] if response.content else None
        text = first.text if isinstance(first, anthropic.types.TextBlock) else ""
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Claude returned invalid JSON: {text[:200]}") from exc

    def _parse(self, content_id: str, raw: dict[str, Any]) -> ModerationResult:
        is_problematic = bool(raw.get("is_problematic", False))
        confidence = float(raw.get("confidence", 0.0))
        categories = raw.get("categories", [])
        explanation = raw.get("explanation", "No explanation provided.")
        flagged_phrases = raw.get("flagged_phrases", [])
        verdict = self._verdict(is_problematic, confidence)

        return ModerationResult(
            id=content_id,
            verdict=verdict,
            confidence=confidence,
            categories=categories,
            explanation=explanation,
            flaggedPhrases=flagged_phrases,
        )
