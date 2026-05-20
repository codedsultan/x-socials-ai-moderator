"""
moderation_service.py

Thin orchestration layer — backend-agnostic.

The backend is selected once at startup from MODERATION_MODE (settings):

  rule       → RuleBasedBackend (detoxify, local, free)
  anthropic  → AnthropicBackend (Claude)
  openai     → OpenAICompatBackend (OpenAI / Groq / Together / Mistral)
  ollama     → OpenAICompatBackend pointed at local Ollama
  hybrid     → HybridBackend (rule-based pre-filter + LLM escalation)

All batch concurrency, error handling, and DB writes live in the calling
service (scan_service) or the routes layer — this class stays thin.
"""
from __future__ import annotations

import asyncio
import logging

from app.models.schemas import ContentType, ModerationResult
from app.models.settings import settings
from app.services.backends.base import ModerationBackend

logger = logging.getLogger(__name__)


def _build_backend() -> ModerationBackend:
    """Factory — reads settings and wires up the right backend graph."""
    from app.services.backends.rule_based import RuleBasedBackend
    from app.services.backends.anthropic_backend import AnthropicBackend
    from app.services.backends.openai_compat import OpenAICompatBackend
    from app.services.backends.hybrid import HybridBackend

    mode = settings.moderation_mode
    logger.info("Building moderation backend: mode=%s", mode)

    if mode == "rule":
        return RuleBasedBackend()

    if mode == "anthropic":
        return AnthropicBackend()

    if mode == "openai":
        return OpenAICompatBackend(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )

    if mode == "ollama":
        return OpenAICompatBackend(
            base_url=settings.ollama_base_url,
            api_key="ollama",          # Ollama ignores the key but the client requires one
            model=settings.ollama_model,
        )

    if mode == "hybrid":
        fast = RuleBasedBackend()
        smart_mode = settings.hybrid_smart_backend

        if smart_mode == "anthropic":
            smart: ModerationBackend = AnthropicBackend()
        elif smart_mode == "openai":
            smart = OpenAICompatBackend(
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
                model=settings.openai_model,
            )
        else:  # ollama
            smart = OpenAICompatBackend(
                base_url=settings.ollama_base_url,
                api_key="ollama",
                model=settings.ollama_model,
            )

        return HybridBackend(fast=fast, smart=smart)

    raise ValueError(f"Unknown MODERATION_MODE: {mode!r}")


class ModerationService:
    """
    Thin dispatcher — delegates everything to the active backend.

    Handles:
      - single-item moderation (with error wrapping)
      - concurrent batch moderation
    """

    def __init__(self, backend: ModerationBackend | None = None) -> None:
        self._backend: ModerationBackend = backend or _build_backend()

    @property
    def backend(self) -> ModerationBackend:
        return self._backend

    # ── Public API ────────────────────────────────────────────────────────────

    async def moderate(
        self,
        content_id:   str,
        content:      str,
        author_id:    str         = "",
        model:        str | None  = None,   # kept for scan_service compatibility
        content_type: ContentType = "comment",
    ) -> ModerationResult:
        """Analyse a single comment or post."""
        # `model` override: if a per-request model is specified and we're on
        # an LLM backend, swap to a one-shot AnthropicBackend with that model.
        backend = self._override_backend(model) if model else self._backend
        return await backend.analyse(
            content_id=content_id,
            content=content,
            content_type=content_type,
            author_id=author_id,
        )

    async def moderate_batch(
        self,
        items:        list[dict[str, str]],
        model:        str | None  = None,
        content_type: ContentType = "comment",
    ) -> list[ModerationResult]:
        """Analyse a list of items concurrently."""
        tasks = [
            self.moderate(
                content_id=c["id"],
                content=c["content"],
                author_id=c.get("authorId", ""),
                model=model,
                content_type=content_type,
            )
            for c in items
        ]
        return list(await asyncio.gather(*tasks))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _override_backend(self, model: str) -> ModerationBackend:
        """Return a one-shot AnthropicBackend using the requested model."""
        from app.services.backends.anthropic_backend import AnthropicBackend
        return AnthropicBackend(model=model)

    # ── Legacy compat (tests patch these directly) ────────────────────────────

    def _parse(self, content_id: str, raw: dict) -> ModerationResult:
        """Exposed for backward-compat with existing tests that patch _parse."""
        from app.services.backends.anthropic_backend import AnthropicBackend
        return AnthropicBackend()._parse(content_id, raw)

    def _verdict(self, is_problematic: bool, confidence: float):
        from app.services.backends.base import ModerationBackend as _B
        return self._backend._verdict(is_problematic, confidence)


# Singleton — created once at import time; backend is wired from settings.
moderation_service = ModerationService()
