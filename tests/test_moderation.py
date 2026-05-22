"""
Tests for the moderation service and all backends.

Run locally:
    docker compose run --rm test
    docker compose run --rm test pytest app/tests/test_moderation.py -v -k "Hybrid"

Coverage:
    AnthropicBackend   — _parse(), _verdict(), analyse() happy path + error path
    OpenAICompatBackend — analyse(), JSON fence stripping, error handling, post content_type
    OpenRouterBackend  — analyse(), JSON fence stripping, error handling, missing API key
    RuleBasedBackend   — model absent passthrough, detoxify scores, safe content, predict error
    HybridBackend      — safe skip, toxic skip, ambiguous escalation, fast error, smart error,
                         boundary conditions
    ModerationService  — single moderate, batch, model override wiring, batch content_type
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas import ModerationResult
from app.services.backends.anthropic_backend import AnthropicBackend
from app.services.backends.hybrid import HybridBackend
from app.services.backends.openai_compat import OpenAICompatBackend
from app.services.backends.openrouter_backend import OpenRouterBackend
from app.services.backends.rule_based import RuleBasedBackend
from app.services.moderation_service import ModerationService

# ── Shared test data ───────────────────────────────────────────────────────────


def _safe_raw() -> dict:
    return {
        "is_problematic": False,
        "confidence": 0.05,
        "categories": [],
        "explanation": "Friendly comment with no violations.",
        "flagged_phrases": [],
    }


def _remove_raw() -> dict:
    return {
        "is_problematic": True,
        "confidence": 0.95,
        "categories": ["hate_speech"],
        "explanation": "Contains explicit hate speech.",
        "flagged_phrases": ["awful slur"],
    }


def _review_raw() -> dict:
    return {
        "is_problematic": True,
        "confidence": 0.65,
        "categories": ["harassment"],
        "explanation": "Mildly aggressive tone.",
        "flagged_phrases": [],
    }


def _safe_result(content_id: str = "c1") -> ModerationResult:
    return ModerationResult(id=content_id, verdict="safe", confidence=0.05, explanation="ok")


def _remove_result(content_id: str = "c2") -> ModerationResult:
    return ModerationResult(id=content_id, verdict="remove", confidence=0.95, explanation="bad")


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def anthropic_backend() -> AnthropicBackend:
    return AnthropicBackend()


@pytest.fixture
def anthropic_service() -> ModerationService:
    """Service wired to AnthropicBackend — used for service-level tests."""
    return ModerationService(backend=AnthropicBackend())


# ══════════════════════════════════════════════════════════════════════════════
# AnthropicBackend._parse()
# ══════════════════════════════════════════════════════════════════════════════


class TestAnthropicParse:
    def test_safe_comment(self, anthropic_backend: AnthropicBackend) -> None:
        result = anthropic_backend._parse("c1", _safe_raw())
        assert result.verdict == "safe"
        assert result.confidence == 0.05
        assert result.categories == []
        assert result.flaggedPhrases == []
        assert result.error is False

    def test_remove_verdict_high_confidence(self, anthropic_backend: AnthropicBackend) -> None:
        result = anthropic_backend._parse("c2", _remove_raw())
        assert result.verdict == "remove"
        assert result.flaggedPhrases == ["awful slur"]

    def test_review_verdict_medium_confidence(self, anthropic_backend: AnthropicBackend) -> None:
        result = anthropic_backend._parse("c3", _review_raw())
        assert result.verdict == "review"

    def test_safe_when_confidence_below_review_threshold(
        self, anthropic_backend: AnthropicBackend
    ) -> None:
        raw = {
            "is_problematic": True,
            "confidence": 0.30,
            "categories": ["off_topic"],
            "explanation": "Slightly off-topic.",
            "flagged_phrases": [],
        }
        result = anthropic_backend._parse("c4", raw)
        assert result.verdict == "safe"

    def test_missing_fields_use_defaults(self, anthropic_backend: AnthropicBackend) -> None:
        result = anthropic_backend._parse("c5", {})
        assert result.verdict == "safe"
        assert result.confidence == 0.0
        assert result.explanation == "No explanation provided."


# ══════════════════════════════════════════════════════════════════════════════
# Verdict thresholds (defined on ModerationBackend base, tested via Anthropic)
# ══════════════════════════════════════════════════════════════════════════════


class TestVerdict:
    @pytest.mark.parametrize(
        "confidence,expected",
        [
            (0.95, "remove"),
            (0.85, "remove"),
            (0.70, "review"),
            (0.50, "review"),
            (0.49, "safe"),
            (0.0, "safe"),
        ],
    )
    def test_thresholds(
        self,
        anthropic_backend: AnthropicBackend,
        confidence: float,
        expected: str,
    ) -> None:
        assert anthropic_backend._verdict(True, confidence) == expected

    def test_not_problematic_always_safe(self, anthropic_backend: AnthropicBackend) -> None:
        # is_problematic=False short-circuits regardless of confidence score
        assert anthropic_backend._verdict(False, 0.99) == "safe"


# ══════════════════════════════════════════════════════════════════════════════
# AnthropicBackend.analyse() — patch at the analyse boundary, not _call_claude
# ══════════════════════════════════════════════════════════════════════════════


class TestAnthropicAnalyse:
    async def test_happy_path_comment(self, anthropic_backend: AnthropicBackend) -> None:
        with patch.object(
            anthropic_backend, "_call_claude", new=AsyncMock(return_value=_safe_raw())
        ):
            result = await anthropic_backend.analyse("c10", "Hello world!")
        assert result.verdict == "safe"
        assert result.id == "c10"
        assert result.error is False

    async def test_happy_path_post(self, anthropic_backend: AnthropicBackend) -> None:
        with patch.object(
            anthropic_backend, "_call_claude", new=AsyncMock(return_value=_remove_raw())
        ):
            result = await anthropic_backend.analyse(
                "post-1", "Title: Bad\n\nBody: Hate speech here.", content_type="post"
            )
        assert result.verdict == "remove"
        assert result.flaggedPhrases == ["awful slur"]

    async def test_api_error_returns_error_result(
        self, anthropic_backend: AnthropicBackend
    ) -> None:
        with patch.object(
            anthropic_backend, "_call_claude", new=AsyncMock(side_effect=Exception("API down"))
        ):
            result = await anthropic_backend.analyse("c11", "Test comment")
        assert result.error is True
        assert result.verdict == "review"
        assert result.id == "c11"

    async def test_post_prompt_differs_from_comment_prompt(self) -> None:
        from app.services.backends._prompts import COMMENT_PROMPT, POST_PROMPT

        assert POST_PROMPT != COMMENT_PROMPT
        assert "post" in POST_PROMPT.lower()


# ══════════════════════════════════════════════════════════════════════════════
# OpenAICompatBackend
# ══════════════════════════════════════════════════════════════════════════════


class TestOpenAICompatBackend:
    def _make_backend(self) -> OpenAICompatBackend:
        return OpenAICompatBackend(
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4o-mini",
        )

    def _mock_response(self, json_str: str) -> MagicMock:
        choice = MagicMock()
        choice.message.content = json_str
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    async def test_safe_result(self) -> None:
        backend = self._make_backend()
        raw_json = (
            '{"is_problematic":false,"confidence":0.03,'
            '"categories":[],"explanation":"Fine.","flagged_phrases":[]}'
        )
        backend._client = MagicMock()
        backend._client.chat.completions.create = AsyncMock(
            return_value=self._mock_response(raw_json)
        )
        result = await backend.analyse("o1", "Hello!")
        assert result.verdict == "safe"
        assert result.confidence == 0.03

    async def test_strips_markdown_fences(self) -> None:
        backend = self._make_backend()
        raw_json = (
            '```json\n{"is_problematic":true,"confidence":0.88,'
            '"categories":["hate_speech"],"explanation":"Bad.","flagged_phrases":["slur"]}\n```'
        )
        backend._client = MagicMock()
        backend._client.chat.completions.create = AsyncMock(
            return_value=self._mock_response(raw_json)
        )
        result = await backend.analyse("o2", "some text")
        assert result.verdict == "remove"
        assert result.flaggedPhrases == ["slur"]

    async def test_api_error_returns_error_result(self) -> None:
        backend = self._make_backend()
        backend._client = MagicMock()
        backend._client.chat.completions.create = AsyncMock(side_effect=Exception("network error"))
        result = await backend.analyse("o3", "text")
        assert result.error is True
        assert result.verdict == "review"

    async def test_invalid_json_returns_error(self) -> None:
        backend = self._make_backend()
        backend._client = MagicMock()
        backend._client.chat.completions.create = AsyncMock(
            return_value=self._mock_response("not json at all")
        )
        result = await backend.analyse("o4", "text")
        assert result.error is True

    async def test_post_content_type_uses_post_framing(self) -> None:
        backend = self._make_backend()
        captured: list[dict] = []

        async def fake_create(**kwargs: object) -> MagicMock:
            captured.append(kwargs)
            return self._mock_response(
                '{"is_problematic":false,"confidence":0.01,"categories":[],'
                '"explanation":"ok","flagged_phrases":[]}'
            )

        backend._client = MagicMock()
        backend._client.chat.completions.create = AsyncMock(side_effect=fake_create)
        await backend.analyse("o5", "Title: Hi\n\nBody: World", content_type="post")

        user_msg = captured[0]["messages"][1]["content"]
        assert user_msg.startswith("Post to moderate:")


# ══════════════════════════════════════════════════════════════════════════════
# OpenRouterBackend
# ══════════════════════════════════════════════════════════════════════════════


class TestOpenRouterBackend:
    def _make_backend(self) -> OpenRouterBackend:
        backend = OpenRouterBackend(
            model="anthropic/claude-haiku-3-5",
            site_url="https://example.com",
            app_title="Test",
        )
        backend._client = MagicMock()  # type: ignore[assignment]
        return backend

    def _mock_response(self, json_str: str) -> MagicMock:
        choice = MagicMock()
        choice.message.content = json_str
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    async def test_safe_result(self) -> None:
        backend = self._make_backend()
        raw_json = (
            '{"is_problematic":false,"confidence":0.02,'
            '"categories":[],"explanation":"Clean.","flagged_phrases":[]}'
        )
        client: Any = backend._client
        client.chat.completions.create = AsyncMock(return_value=self._mock_response(raw_json))
        result = await backend.analyse("or1", "Nice post!")
        assert result.verdict == "safe"
        assert result.confidence == 0.02
        assert result.error is False

    async def test_remove_verdict(self) -> None:
        backend = self._make_backend()
        raw_json = (
            '{"is_problematic":true,"confidence":0.91,'
            '"categories":["hate_speech"],"explanation":"Bad.","flagged_phrases":["slur"]}'
        )
        client: Any = backend._client
        client.chat.completions.create = AsyncMock(return_value=self._mock_response(raw_json))
        result = await backend.analyse("or2", "Hateful content")
        assert result.verdict == "remove"
        assert result.flaggedPhrases == ["slur"]

    async def test_strips_markdown_fences(self) -> None:
        backend = self._make_backend()
        raw_json = (
            '```json\n{"is_problematic":false,"confidence":0.01,'
            '"categories":[],"explanation":"Fine.","flagged_phrases":[]}\n```'
        )
        client: Any = backend._client
        client.chat.completions.create = AsyncMock(return_value=self._mock_response(raw_json))
        result = await backend.analyse("or3", "text")
        assert result.verdict == "safe"
        assert result.error is False

    async def test_api_error_returns_error_result(self) -> None:
        backend = self._make_backend()
        client: Any = backend._client
        client.chat.completions.create = AsyncMock(side_effect=Exception("OpenRouter down"))
        result = await backend.analyse("or4", "text")
        assert result.error is True
        assert result.verdict == "review"

    async def test_invalid_json_returns_error(self) -> None:
        backend = self._make_backend()
        client: Any = backend._client
        client.chat.completions.create = AsyncMock(
            return_value=self._mock_response("not valid json")
        )
        result = await backend.analyse("or5", "text")
        assert result.error is True

    async def test_missing_api_key_raises_on_client_access(self) -> None:
        """Client property should raise RuntimeError when API key is not configured."""
        backend = OpenRouterBackend(model="anthropic/claude-haiku-3-5")
        # Don't pre-build _client — let the property try to build it
        with patch("app.services.backends.openrouter_backend.settings") as mock_settings:
            mock_settings.openrouter_api_key = ""
            mock_settings.openrouter_site_url = "https://example.com"
            mock_settings.openrouter_app_title = "Test"
            backend._client = None  # force property rebuild
            with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
                _ = backend.client

    async def test_post_content_type_uses_post_framing(self) -> None:
        backend = self._make_backend()
        captured: list[dict] = []

        async def fake_create(**kwargs: object) -> MagicMock:
            captured.append(kwargs)
            return self._mock_response(
                '{"is_problematic":false,"confidence":0.01,"categories":[],'
                '"explanation":"ok","flagged_phrases":[]}'
            )

        client: Any = backend._client
        client.chat.completions.create = AsyncMock(side_effect=fake_create)
        await backend.analyse("or6", "Title: Hi\n\nBody: World", content_type="post")

        user_msg = captured[0]["messages"][1]["content"]
        assert user_msg.startswith("Post to moderate:")


# ══════════════════════════════════════════════════════════════════════════════
# RuleBasedBackend
# ══════════════════════════════════════════════════════════════════════════════


class TestRuleBasedBackend:
    def _make_backend(self, model: object = None) -> RuleBasedBackend:
        """Bypass __init__ to avoid triggering detoxify model download in CI."""
        backend = RuleBasedBackend.__new__(RuleBasedBackend)
        backend._model = model
        return backend

    async def test_returns_safe_when_model_absent(self) -> None:
        """No detoxify installed → safe passthrough, no error."""
        result = await self._make_backend(model=None).analyse("r1", "Hello world")
        assert result.verdict == "safe"
        assert result.confidence == 0.0
        assert result.error is False

    async def test_remove_verdict_from_high_toxicity_scores(self) -> None:
        fake_model = MagicMock()
        fake_model.predict.return_value = {
            "toxicity": 0.92,
            "severe_toxicity": 0.70,
            "obscene": 0.10,
            "threat": 0.05,
            "insult": 0.60,
            "identity_attack": 0.50,
        }
        result = await self._make_backend(model=fake_model).analyse("r2", "very toxic text")

        fake_model.predict.assert_called_once_with("very toxic text")
        assert result.confidence >= 0.85
        assert result.verdict == "remove"
        assert "toxicity" in result.categories

    async def test_safe_content_returns_safe(self) -> None:
        fake_model = MagicMock()
        fake_model.predict.return_value = {
            "toxicity": 0.01,
            "severe_toxicity": 0.00,
            "obscene": 0.00,
            "threat": 0.00,
            "insult": 0.02,
            "identity_attack": 0.00,
        }
        result = await self._make_backend(model=fake_model).analyse("r3", "Great post!")
        assert result.verdict == "safe"
        assert result.categories == []

    async def test_predict_exception_returns_error_result(self) -> None:
        fake_model = MagicMock()
        fake_model.predict.side_effect = RuntimeError("model broken")
        result = await self._make_backend(model=fake_model).analyse("r4", "anything")
        assert result.error is True
        assert result.verdict == "review"

    async def test_categories_deduplicated(self) -> None:
        """severe_toxicity and identity_attack both map to hate_speech — should appear once."""
        fake_model = MagicMock()
        fake_model.predict.return_value = {
            "toxicity": 0.30,
            "severe_toxicity": 0.60,
            "obscene": 0.10,
            "threat": 0.05,
            "insult": 0.10,
            "identity_attack": 0.50,
        }
        result = await self._make_backend(model=fake_model).analyse("r5", "hateful text")
        assert result.categories.count("hate_speech") == 1


# ══════════════════════════════════════════════════════════════════════════════
# HybridBackend
# ══════════════════════════════════════════════════════════════════════════════


class TestHybridBackend:
    def _make_hybrid(
        self,
        fast_result: ModerationResult,
        smart_result: ModerationResult | None = None,
        safe_ceiling: float = 0.15,
        flag_floor: float = 0.80,
    ) -> HybridBackend:
        fast = MagicMock()
        smart = MagicMock()
        fast.analyse = AsyncMock(return_value=fast_result)
        smart.analyse = AsyncMock(return_value=smart_result or fast_result)
        return HybridBackend(
            fast=fast, smart=smart, safe_ceiling=safe_ceiling, flag_floor=flag_floor
        )

    async def test_clearly_safe_skips_smart_backend(self) -> None:
        fast_r = _safe_result("h1")
        hybrid = self._make_hybrid(fast_result=fast_r)

        result = await hybrid.analyse("h1", "nice content")

        hybrid._fast.analyse.assert_called_once()  # type: ignore[attr-defined]
        hybrid._smart.analyse.assert_not_called()  # type: ignore[attr-defined]
        assert result.confidence == 0.05

    async def test_clearly_toxic_skips_smart_backend(self) -> None:
        fast_r = _remove_result("h2")
        hybrid = self._make_hybrid(fast_result=fast_r)

        result = await hybrid.analyse("h2", "toxic content")

        hybrid._fast.analyse.assert_called_once()  # type: ignore[attr-defined]
        hybrid._smart.analyse.assert_not_called()  # type: ignore[attr-defined]
        assert result.verdict == "remove"

    async def test_ambiguous_score_escalates_to_smart(self) -> None:
        fast_r = ModerationResult(id="h3", verdict="review", confidence=0.45, explanation="maybe")
        smart_r = ModerationResult(
            id="h3",
            verdict="remove",
            confidence=0.91,
            explanation="yes bad",
            flaggedPhrases=["bad word"],
        )
        hybrid = self._make_hybrid(fast_result=fast_r, smart_result=smart_r)

        result = await hybrid.analyse("h3", "ambiguous text")

        hybrid._fast.analyse.assert_called_once()  # type: ignore[attr-defined]
        hybrid._smart.analyse.assert_called_once()  # type: ignore[attr-defined]
        assert result.verdict == "remove"
        assert result.flaggedPhrases == ["bad word"]

    async def test_fast_error_escalates_to_smart(self) -> None:
        fast_r = ModerationResult(
            id="h4", verdict="review", confidence=0.0, explanation="error", error=True
        )
        smart_r = _safe_result("h4")
        smart_r = ModerationResult(id="h4", verdict="safe", confidence=0.02, explanation="fine")
        hybrid = self._make_hybrid(fast_result=fast_r, smart_result=smart_r)

        result = await hybrid.analyse("h4", "content")

        hybrid._smart.analyse.assert_called_once()  # type: ignore[attr-defined]
        assert result.verdict == "safe"

    async def test_smart_error_falls_back_to_fast_result(self) -> None:
        fast_r = ModerationResult(id="h5", verdict="review", confidence=0.45, explanation="maybe")
        smart_r = ModerationResult(
            id="h5", verdict="review", confidence=0.0, explanation="error", error=True
        )
        hybrid = self._make_hybrid(fast_result=fast_r, smart_result=smart_r)

        result = await hybrid.analyse("h5", "ambiguous")

        # Smart failed — we fall back to the fast result, no error surfaced
        assert result.confidence == 0.45
        assert result.error is False

    async def test_boundary_at_safe_ceiling_skips_smart(self) -> None:
        """Score exactly at safe_ceiling is still considered safe — no escalation."""
        fast_r = _safe_result("h6")
        hybrid = self._make_hybrid(fast_result=fast_r, safe_ceiling=0.05)
        await hybrid.analyse("h6", "content")
        hybrid._smart.analyse.assert_not_called()  # type: ignore[attr-defined]

    async def test_boundary_at_flag_floor_skips_smart(self) -> None:
        """Score exactly at flag_floor is still flagged — no escalation."""
        fast_r = _remove_result("h7")
        hybrid = self._make_hybrid(fast_result=fast_r, flag_floor=0.95)
        await hybrid.analyse("h7", "content")
        hybrid._smart.analyse.assert_not_called()  # type: ignore[attr-defined]

    async def test_both_backends_called_with_same_args(self) -> None:
        """Escalation must forward the original content_id, content, and content_type."""
        fast_r = ModerationResult(id="h8", verdict="review", confidence=0.45, explanation="maybe")
        smart_r = ModerationResult(id="h8", verdict="safe", confidence=0.10, explanation="fine")
        hybrid = self._make_hybrid(fast_result=fast_r, smart_result=smart_r)

        await hybrid.analyse("h8", "test content", content_type="post", author_id="u1")

        hybrid._fast.analyse.assert_called_once_with("h8", "test content", "post", "u1")  # type: ignore[attr-defined]
        hybrid._smart.analyse.assert_called_once_with("h8", "test content", "post", "u1")  # type: ignore[attr-defined]


# ══════════════════════════════════════════════════════════════════════════════
# ModerationService
# ══════════════════════════════════════════════════════════════════════════════


class TestModerationService:
    async def test_moderate_returns_result(self, anthropic_service: ModerationService) -> None:
        with patch.object(
            anthropic_service.backend, "analyse", new=AsyncMock(return_value=_safe_result("c10"))
        ):
            result = await anthropic_service.moderate("c10", "Hello world!")
        assert isinstance(result, ModerationResult)
        assert result.id == "c10"
        assert result.verdict == "safe"
        assert result.error is False

    async def test_moderate_error_path(self, anthropic_service: ModerationService) -> None:
        """analyse() raising should surface as an error result, not an unhandled exception."""
        error_result = ModerationResult(
            id="c11", verdict="review", confidence=0.0, explanation="error", error=True
        )
        with patch.object(
            anthropic_service.backend, "analyse", new=AsyncMock(return_value=error_result)
        ):
            result = await anthropic_service.moderate("c11", "Test comment")
        assert result.error is True
        assert result.verdict == "review"

    async def test_batch_calls_analyse_for_every_item(
        self, anthropic_service: ModerationService
    ) -> None:
        call_count = 0

        async def fake_analyse(
            content_id: str, content: str, content_type: str = "comment", author_id: str = ""
        ) -> ModerationResult:
            nonlocal call_count
            call_count += 1
            return _safe_result(content_id)

        with patch.object(anthropic_service.backend, "analyse", side_effect=fake_analyse):
            items = [{"id": f"c{i}", "content": f"comment {i}"} for i in range(5)]
            results = await anthropic_service.moderate_batch(items)

        assert len(results) == 5
        assert call_count == 5

    async def test_batch_propagates_content_type(
        self, anthropic_service: ModerationService
    ) -> None:
        received: list[str] = []

        async def fake_analyse(
            content_id: str, content: str, content_type: str = "comment", author_id: str = ""
        ) -> ModerationResult:
            received.append(content_type)
            return _safe_result(content_id)

        with patch.object(anthropic_service.backend, "analyse", side_effect=fake_analyse):
            await anthropic_service.moderate_batch(
                [{"id": f"p{i}", "content": "text"} for i in range(3)],
                content_type="post",
            )

        assert all(t == "post" for t in received)

    async def test_model_override_with_slash_uses_openrouter(self) -> None:
        """Model strings containing '/' should route to OpenRouterBackend."""
        svc = ModerationService(backend=AnthropicBackend())

        with patch("app.services.moderation_service.OpenRouterBackend") as MockOpenRouter:
            mock_backend = MagicMock()
            mock_backend.analyse = AsyncMock(return_value=_safe_result("x1"))
            MockOpenRouter.return_value = mock_backend

            result = await svc.moderate("x1", "text", model="anthropic/claude-haiku-3-5")

        MockOpenRouter.assert_called_once_with(model="anthropic/claude-haiku-3-5")
        assert result.verdict == "safe"

    async def test_model_override_without_slash_uses_anthropic(self) -> None:
        """Model strings without '/' should route to AnthropicBackend."""
        svc = ModerationService(backend=AnthropicBackend())

        with patch("app.services.moderation_service.AnthropicBackend") as MockAnthropic:
            mock_backend = MagicMock()
            mock_backend.analyse = AsyncMock(return_value=_safe_result("x2"))
            MockAnthropic.return_value = mock_backend

            result = await svc.moderate("x2", "text", model="claude-sonnet-4-20250514")

        MockAnthropic.assert_called_once_with(model="claude-sonnet-4-20250514")
        assert result.verdict == "safe"
