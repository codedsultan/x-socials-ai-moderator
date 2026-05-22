"""
moderation_service.py

Thin orchestration layer — backend-agnostic.

The backend is selected once at startup from MODERATION_MODE (settings):

  rule        → RuleBasedBackend (detoxify, local, free)
  anthropic   → AnthropicBackend (Claude direct SDK)
  openai      → OpenAICompatBackend (OpenAI / Groq / Together / Mistral)
  ollama      → OpenAICompatBackend pointed at local Ollama
  openrouter  → OpenRouterBackend (200+ models, unified billing, auto-fallback)
  hybrid      → HybridBackend (rule-based pre-filter + LLM escalation)
"""
from __future__ import annotations

import asyncio
import logging

from app.models.schemas import ContentType, ModerationResult
from app.models.settings import settings
from app.services.backends.anthropic_backend import AnthropicBackend
from app.services.backends.base import ModerationBackend
from app.services.backends.openrouter_backend import OpenRouterBackend

logger = logging.getLogger(__name__)


def _build_backend() -> ModerationBackend:
    """Factory — reads settings and wires up the right backend graph."""
    from app.services.backends.anthropic_backend import AnthropicBackend
    from app.services.backends.hybrid import HybridBackend
    from app.services.backends.openai_compat import OpenAICompatBackend
    from app.services.backends.openrouter_backend import OpenRouterBackend
    from app.services.backends.rule_based import RuleBasedBackend

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
            api_key="ollama",
            model=settings.ollama_model,
        )

    if mode == "openrouter":
        return OpenRouterBackend()

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
        elif smart_mode == "openrouter":
            # [Fix 10] OpenRouter is now a valid hybrid smart backend.
            # Default — gives provider portability + automatic failover.
            smart = OpenRouterBackend()
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
      - concurrent batch moderation with semaphore-controlled parallelism
    """

    def __init__(self, backend: ModerationBackend | None = None) -> None:
        # [Fix 9] Always initialise _backend in __init__ so the object is
        # never in a partially constructed state. Previously _active_scans
        # in ScanService (same pattern) was lazily set in trigger() which
        # made shutdown() fragile. Same fix applied here for consistency.
        self._backend: ModerationBackend = backend or _build_backend()

        # [Fix 6] Semaphore caps concurrent LLM calls in moderate_batch().
        # Created here so it is tied to this service instance's lifecycle
        # and shared across all batch calls (not re-created per batch).
        self._llm_semaphore = asyncio.Semaphore(settings.batch_concurrency)

    @property
    def backend(self) -> ModerationBackend:
        return self._backend

    # ── Public API ────────────────────────────────────────────────────────────

    async def moderate(
        self,
        content_id:   str,
        content:      str,
        author_id:    str         = "",
        model:        str | None  = None,
        content_type: ContentType = "comment",
    ) -> ModerationResult:
        """Analyse a single comment or post."""
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
        """
        Analyse a list of items with bounded concurrency.

        [Fix 6] Previously used asyncio.gather(*tasks) with no limit, which
        fires all N requests simultaneously. For a batch of 50 comments this
        could trigger 50 concurrent Claude API calls, saturating the rate
        limit (RPM) and causing 429 errors that corrupt the entire batch.

        The semaphore (default 10) allows up to batch_concurrency calls
        in-flight at once. Remaining tasks wait in the asyncio queue —
        no threads, no overhead, pure cooperative scheduling.

        Throughput: with 10 concurrent calls at ~300ms each, 50 items
        complete in ~1.5s. With no semaphore, 50 simultaneous calls
        trigger rate limiting and the same 50 items take 5-10s with
        retry overhead — slower AND more expensive.
        """
        async def _bounded(item: dict[str, str]) -> ModerationResult:
            async with self._llm_semaphore:
                return await self.moderate(
                    content_id=item["id"],
                    content=item["content"],
                    author_id=item.get("authorId", ""),
                    model=model,
                    content_type=content_type,
                )

        return list(await asyncio.gather(*[_bounded(c) for c in items]))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _override_backend(self, model: str) -> ModerationBackend:
        """
        Return a one-shot backend using the requested model.

        If the model string looks like an OpenRouter model (contains '/'),
        use OpenRouterBackend. Otherwise fall back to AnthropicBackend for
        direct SDK access (supports prompt caching on future upgrade).
        """
        if "/" in model:
            return OpenRouterBackend(model=model)
        return AnthropicBackend(model=model)

    # ── Legacy compat (tests patch these directly) ────────────────────────────

    def _parse(self, content_id: str, raw: dict) -> ModerationResult:
        return AnthropicBackend()._parse(content_id, raw)

    def _verdict(self, is_problematic: bool, confidence: float) -> str:
        return self._backend._verdict(is_problematic, confidence)


# Singleton — created once at import time; backend is wired from settings.
moderation_service = ModerationService()
