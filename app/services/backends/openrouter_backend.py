"""
openrouter_backend.py

OpenRouter moderation backend.

OpenRouter (https://openrouter.ai) is an API aggregator that exposes 200+
models — Claude, GPT-4o, Gemini, Mistral, Llama, and more — through a
single OpenAI-compatible endpoint with a single API key.

Why this exists alongside AnthropicBackend:
  - Unified billing: one invoice, one key, usage dashboard for all providers.
  - Automatic fallback: OpenRouter retries across providers when one is down
    or rate-limited. Your scan keeps running during an Anthropic outage.
  - Model diversity: different models have different moderation sensitivities.
    Route nuanced harassment to Claude, bulk spam to a free Llama tier.
  - Cost control: meta-llama/llama-3.1-8b-instruct:free costs nothing for
    low-stakes screening; anthropic/claude-haiku-3-5 for anything flagged.

Why AnthropicBackend still exists:
  - Prompt caching (anthropic-beta: prompt-caching-2024-07-31) requires the
    native SDK. The system prompt is static — caching it cuts per-call token
    cost by ~80%. OpenRouter cannot surface this feature.
  - Extended thinking and tool use also require the native SDK.

Default model: anthropic/claude-haiku-3-5
  Same model as the direct AnthropicBackend default, same quality, same
  pricing — but with OpenRouter's failover and unified billing on top.

Configuration (.env):
  OPENROUTER_API_KEY=sk-or-...
  OPENROUTER_MODEL=anthropic/claude-haiku-3-5
  OPENROUTER_SITE_URL=https://github.com/codesultan/x-socials
  OPENROUTER_APP_TITLE=X-Socials AI Moderator

OpenRouter requires two extra headers per their docs:
  HTTP-Referer: your site URL (used for their abuse filtering dashboard)
  X-Title:      your app name (appears in your OpenRouter usage dashboard)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from app.models.schemas import ContentType, ModerationResult
from app.models.settings import settings
from app.services.backends._prompts import COMMENT_PROMPT, POST_PROMPT
from app.services.backends.base import ModerationBackend

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend(ModerationBackend):
    """
    Moderation backend backed by OpenRouter.

    Inherits the same prompt schema and JSON parsing as AnthropicBackend
    and OpenAICompatBackend — results are fully interchangeable.

    Args:
        model:      OpenRouter model string (provider/model format).
                    Defaults to settings.openrouter_model.
        site_url:   Your site URL for OpenRouter's HTTP-Referer header.
        app_title:  Your app name for OpenRouter's X-Title header.
        max_tokens: Token budget for the response (512 is sufficient for JSON).
    """

    def __init__(
        self,
        model: str | None = None,
        site_url: str | None = None,
        app_title: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        self._model = model or settings.openrouter_model
        self._max_tokens = max_tokens
        self._client: AsyncOpenAI | None = None

        # OpenRouter-required headers — stored here, injected on first client build
        self._site_url = site_url or settings.openrouter_site_url
        self._app_title = app_title or settings.openrouter_app_title

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy singleton — built once, reused across all calls."""
        if self._client is None:
            if not settings.openrouter_api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not configured. "
                    "Get a key at https://openrouter.ai/keys and set it in .env."
                )
            self._client = AsyncOpenAI(
                base_url=_OPENROUTER_BASE_URL,
                api_key=settings.openrouter_api_key,
                # OpenRouter requires these two headers on every request.
                # Passing them as default_headers means they're sent automatically
                # without any change to the call sites.
                default_headers={
                    "HTTP-Referer": self._site_url,
                    "X-Title": self._app_title,
                },
            )
            logger.info("OpenRouterBackend client initialised (model=%s)", self._model)
        return self._client

    async def analyse(
        self,
        content_id: str,
        content: str,
        content_type: ContentType = "comment",
        author_id: str = "",
    ) -> ModerationResult:
        try:
            raw = await self._call(content, content_type)
            return self._parse(content_id, raw)
        except Exception as exc:
            logger.exception("OpenRouterBackend error for %s", content_id)
            return self._error_result(content_id, exc)

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _call(self, content: str, content_type: ContentType) -> dict[str, Any]:
        system = POST_PROMPT if content_type == "post" else COMMENT_PROMPT
        label = "Post" if content_type == "post" else "Comment"
        message = f"{label} to moderate:\n\n{content}"

        response = await self.client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ],
            # Most OpenRouter models support json_object response format.
            # Models that don't will ignore it and still return JSON because
            # the system prompt explicitly instructs them to.
            response_format={"type": "json_object"},
        )

        text = (response.choices[0].message.content or "").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"OpenRouter ({self._model}) returned invalid JSON: {text[:200]}"
            ) from exc

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
