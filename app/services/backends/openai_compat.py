"""
openai_compat.py

LLM moderation backend for any OpenAI-compatible API.

One client, zero lock-in.  Point it at:
  - OpenAI:     base_url=https://api.openai.com/v1        model=gpt-4o-mini
  - Groq:       base_url=https://api.groq.com/openai/v1   model=llama-3.1-8b-instant
  - Together:   base_url=https://api.together.xyz/v1
                model=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
  - Mistral:    base_url=https://api.mistral.ai/v1         model=mistral-small-latest
  - Ollama:     base_url=http://localhost:11434/v1         model=llama3.2   api_key=ollama
  - LM Studio:  base_url=http://localhost:1234/v1          model=<loaded>   api_key=lm-studio

All providers receive the same structured JSON prompt; the schema is
identical to the Anthropic backend so results are interchangeable.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from app.models.schemas import ContentType, ModerationResult
from app.services.backends._prompts import COMMENT_PROMPT, POST_PROMPT
from app.services.backends.base import ModerationBackend

logger = logging.getLogger(__name__)


class OpenAICompatBackend(ModerationBackend):
    """
    Wraps any OpenAI-compatible chat completion endpoint.

    Args:
        base_url:  Full API base URL (including /v1 if required).
        api_key:   API key string. Use "ollama" or "lm-studio" for local servers.
        model:     Model identifier as accepted by the provider.
        max_tokens: Token budget for the response (JSON only, 512 is plenty).
    """

    def __init__(
        self,
        base_url:   str,
        api_key:    str,
        model:      str,
        max_tokens: int = 512,
    ) -> None:
        self._client    = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model     = model
        self._max_tokens = max_tokens

    async def analyse(
        self,
        content_id:   str,
        content:      str,
        content_type: ContentType = "comment",
        author_id:    str         = "",
    ) -> ModerationResult:
        try:
            raw = await self._call(content, content_type)
            return self._parse(content_id, raw)
        except Exception as exc:
            logger.exception("OpenAICompatBackend error for %s", content_id)
            return self._error_result(content_id, exc)

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _call(self, content: str, content_type: ContentType) -> dict[str, Any]:
        system  = POST_PROMPT if content_type == "post" else COMMENT_PROMPT
        label   = "Post" if content_type == "post" else "Comment"
        message = f"{label} to moderate:\n\n{content}"

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": message},
            ],
            # Request JSON mode when the provider supports it.
            # Providers that don't support response_format silently ignore it.
            response_format={"type": "json_object"},
        )

        text = (response.choices[0].message.content or "").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Provider returned invalid JSON: {text[:200]}") from exc

    def _parse(self, content_id: str, raw: dict[str, Any]) -> ModerationResult:
        is_problematic  = bool(raw.get("is_problematic", False))
        confidence      = float(raw.get("confidence", 0.0))
        categories      = raw.get("categories", [])
        explanation     = raw.get("explanation", "No explanation provided.")
        flagged_phrases = raw.get("flagged_phrases", [])
        verdict         = self._verdict(is_problematic, confidence)

        return ModerationResult(
            id=content_id,
            verdict=verdict,
            confidence=confidence,
            categories=categories,
            explanation=explanation,
            flaggedPhrases=flagged_phrases,
        )
