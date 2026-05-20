"""
Tests for the moderation service and all backends.

Run: pytest app/tests/ -v
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.schemas import ModerationResult
from app.services.backends.anthropic_backend import AnthropicBackend
from app.services.backends.rule_based import RuleBasedBackend
from app.services.backends.openai_compat import OpenAICompatBackend
from app.services.backends.hybrid import HybridBackend
from app.services.moderation_service import ModerationService


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def anthropic_backend() -> AnthropicBackend:
    return AnthropicBackend()


@pytest.fixture
def service() -> ModerationService:
    """Service wired to an AnthropicBackend (legacy fixture for old tests)."""
    return ModerationService(backend=AnthropicBackend())


def _safe_raw() -> dict:
    return {
        "is_problematic": False,
        "confidence":     0.05,
        "categories":     [],
        "explanation":    "Friendly comment with no violations.",
        "flagged_phrases": [],
    }

def _remove_raw() -> dict:
    return {
        "is_problematic": True,
        "confidence":     0.95,
        "categories":     ["hate_speech"],
        "explanation":    "Contains explicit hate speech.",
        "flagged_phrases": ["awful slur"],
    }

def _review_raw() -> dict:
    return {
        "is_problematic": True,
        "confidence":     0.65,
        "categories":     ["harassment"],
        "explanation":    "Mildly aggressive tone.",
        "flagged_phrases": [],
    }

def _safe_result(content_id: str = "c1") -> ModerationResult:
    return ModerationResult(id=content_id, verdict="safe", confidence=0.05, explanation="ok")

def _remove_result(content_id: str = "c2") -> ModerationResult:
    return ModerationResult(id=content_id, verdict="remove", confidence=0.95, explanation="bad")


# ══════════════════════════════════════════════════════════════════════════════
# AnthropicBackend._parse() — unchanged from original tests
# ══════════════════════════════════════════════════════════════════════════════

class TestParse:
    def test_safe_comment(self, anthropic_backend: AnthropicBackend) -> None:
        result = anthropic_backend._parse("c1", _safe_raw())
        assert result.verdict        == "safe"
        assert result.confidence     == 0.05
        assert result.categories     == []
        assert result.flaggedPhrases == []
        assert result.error is False

    def test_remove_verdict_high_confidence(self, anthropic_backend: AnthropicBackend) -> None:
        result = anthropic_backend._parse("c2", _remove_raw())
        assert result.verdict        == "remove"
        assert result.flaggedPhrases == ["awful slur"]

    def test_review_verdict_medium_confidence(self, anthropic_backend: AnthropicBackend) -> None:
        result = anthropic_backend._parse("c3", _review_raw())
        assert result.verdict == "review"

    def test_safe_when_confidence_below_review_threshold(self, anthropic_backend: AnthropicBackend) -> None:
        raw = {"is_problematic": True, "confidence": 0.30, "categories": ["off_topic"],
               "explanation": "Slightly off-topic.", "flagged_phrases": []}
        result = anthropic_backend._parse("c4", raw)
        assert result.verdict == "safe"

    def test_missing_fields_use_defaults(self, anthropic_backend: AnthropicBackend) -> None:
        result = anthropic_backend._parse("c5", {})
        assert result.verdict      == "safe"
        assert result.confidence   == 0.0
        assert result.explanation  == "No explanation provided."


# ══════════════════════════════════════════════════════════════════════════════
# Verdict thresholds (backend base)
# ══════════════════════════════════════════════════════════════════════════════

class TestVerdict:
    @pytest.mark.parametrize("confidence,expected", [
        (0.95, "remove"),
        (0.85, "remove"),
        (0.70, "review"),
        (0.50, "review"),
        (0.49, "safe"),
        (0.0,  "safe"),
    ])
    def test_thresholds(self, anthropic_backend: AnthropicBackend,
                        confidence: float, expected: str) -> None:
        assert anthropic_backend._verdict(True, confidence) == expected

    def test_not_problematic_always_safe(self, anthropic_backend: AnthropicBackend) -> None:
        assert anthropic_backend._verdict(False, 0.99) == "safe"


# ══════════════════════════════════════════════════════════════════════════════
# ModerationService — integration (legacy + new)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_moderate_returns_result(service: ModerationService) -> None:
    with patch.object(
        service.backend, "_call_claude", new=AsyncMock(return_value=_safe_raw())
    ):
        result = await service.moderate("c10", "Hello world!")
    assert isinstance(result, ModerationResult)
    assert result.id      == "c10"
    assert result.verdict == "safe"
    assert result.error   is False


@pytest.mark.asyncio
async def test_moderate_handles_api_error(service: ModerationService) -> None:
    with patch.object(
        service.backend, "_call_claude", new=AsyncMock(side_effect=Exception("API down"))
    ):
        result = await service.moderate("c11", "Test comment")
    assert result.error   is True
    assert result.verdict == "review"


@pytest.mark.asyncio
async def test_moderate_batch_runs_concurrently(service: ModerationService) -> None:
    call_count = 0

    async def fake_analyse(content_id, content, content_type="comment", author_id=""):
        nonlocal call_count
        call_count += 1
        return _safe_result(content_id)

    with patch.object(service.backend, "analyse", side_effect=fake_analyse):
        items   = [{"id": f"c{i}", "content": f"comment {i}"} for i in range(5)]
        results = await service.moderate_batch(items)

    assert len(results) == 5
    assert call_count   == 5


# ── Post-specific tests (kept from original) ──────────────────────────────────

@pytest.mark.asyncio
async def test_moderate_post_uses_post_prompt() -> None:
    from app.services.backends._prompts import POST_PROMPT, COMMENT_PROMPT
    assert POST_PROMPT != COMMENT_PROMPT
    assert "post" in POST_PROMPT.lower()


@pytest.mark.asyncio
async def test_moderate_post_returns_result(service: ModerationService) -> None:
    with patch.object(
        service.backend, "_call_claude", new=AsyncMock(return_value=_remove_raw())
    ):
        result = await service.moderate(
            content_id="post-abc",
            content="Title: Bad Title\n\nBody:\nSome content.",
            content_type="post",
        )
    assert result.id             == "post-abc"
    assert result.verdict        == "remove"
    assert result.flaggedPhrases == ["awful slur"]
    assert result.error          is False


@pytest.mark.asyncio
async def test_moderate_post_error_returns_review_verdict(service: ModerationService) -> None:
    with patch.object(
        service.backend, "_call_claude", new=AsyncMock(side_effect=Exception("timeout"))
    ):
        result = await service.moderate("p-err", "some post content", content_type="post")
    assert result.error   is True
    assert result.verdict == "review"
    assert result.id      == "p-err"


# ══════════════════════════════════════════════════════════════════════════════
# RuleBasedBackend
# ══════════════════════════════════════════════════════════════════════════════

class TestRuleBasedBackend:
    @pytest.mark.asyncio
    async def test_returns_valid_result_when_model_absent(self) -> None:
        """When detoxify is not installed _model is None → safe passthrough."""
        backend = RuleBasedBackend.__new__(RuleBasedBackend)
        backend._model = None
        result = await backend.analyse("r1", "Hello world")
        assert isinstance(result, ModerationResult)
        assert result.id      == "r1"
        assert result.verdict == "safe"
        assert result.error   is False

    @pytest.mark.asyncio
    async def test_uses_detoxify_scores(self) -> None:
        """When a model is present its predict() result drives the verdict."""
        fake_model        = MagicMock()
        fake_model.predict.return_value = {
            "toxicity":        0.92,
            "severe_toxicity": 0.70,
            "obscene":         0.10,
            "threat":          0.05,
            "insult":          0.60,
            "identity_attack": 0.50,
        }
        backend        = RuleBasedBackend.__new__(RuleBasedBackend)
        backend._model = fake_model

        result = await backend.analyse("r2", "very toxic text")

        fake_model.predict.assert_called_once_with("very toxic text")
        assert result.confidence >= 0.85            # 0.92 → remove
        assert result.verdict    == "remove"
        assert "toxicity"        in result.categories

    @pytest.mark.asyncio
    async def test_safe_content_low_scores(self) -> None:
        fake_model        = MagicMock()
        fake_model.predict.return_value = {
            "toxicity":        0.01,
            "severe_toxicity": 0.00,
            "obscene":         0.00,
            "threat":          0.00,
            "insult":          0.02,
            "identity_attack": 0.00,
        }
        backend        = RuleBasedBackend.__new__(RuleBasedBackend)
        backend._model = fake_model

        result = await backend.analyse("r3", "Great post, thanks!")
        assert result.verdict    == "safe"
        assert result.categories == []

    @pytest.mark.asyncio
    async def test_handles_predict_exception(self) -> None:
        fake_model        = MagicMock()
        fake_model.predict.side_effect = RuntimeError("model broken")
        backend        = RuleBasedBackend.__new__(RuleBasedBackend)
        backend._model = fake_model

        result = await backend.analyse("r4", "anything")
        assert result.error   is True
        assert result.verdict == "review"


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
        choice  = MagicMock()
        choice.message.content = json_str
        resp    = MagicMock()
        resp.choices = [choice]
        return resp

    @pytest.mark.asyncio
    async def test_returns_valid_result(self) -> None:
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
        assert result.verdict    == "safe"
        assert result.confidence == 0.03

    @pytest.mark.asyncio
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
        assert result.verdict        == "remove"
        assert result.flaggedPhrases == ["slur"]

    @pytest.mark.asyncio
    async def test_api_error_returns_error_result(self) -> None:
        backend = self._make_backend()
        backend._client = MagicMock()
        backend._client.chat.completions.create = AsyncMock(
            side_effect=Exception("network error")
        )
        result = await backend.analyse("o3", "text")
        assert result.error   is True
        assert result.verdict == "review"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self) -> None:
        backend = self._make_backend()
        backend._client = MagicMock()
        backend._client.chat.completions.create = AsyncMock(
            return_value=self._mock_response("not json at all")
        )
        result = await backend.analyse("o4", "text")
        assert result.error is True

    @pytest.mark.asyncio
    async def test_post_content_type_uses_post_framing(self) -> None:
        backend  = self._make_backend()
        captured: list[dict] = []

        async def fake_create(**kwargs):
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
# HybridBackend
# ══════════════════════════════════════════════════════════════════════════════

class TestHybridBackend:
    def _make_hybrid(
        self,
        fast_result:  ModerationResult,
        smart_result: ModerationResult | None = None,
        safe_ceiling: float = 0.15,
        flag_floor:   float = 0.80,
    ) -> HybridBackend:
        fast  = MagicMock()
        smart = MagicMock()
        fast.analyse  = AsyncMock(return_value=fast_result)
        smart.analyse = AsyncMock(return_value=smart_result or fast_result)
        fast.error  = False
        smart.error = False
        return HybridBackend(fast=fast, smart=smart,
                             safe_ceiling=safe_ceiling, flag_floor=flag_floor)

    @pytest.mark.asyncio
    async def test_clearly_safe_skips_smart(self) -> None:
        fast_r = ModerationResult(id="h1", verdict="safe", confidence=0.05, explanation="ok")
        hybrid = self._make_hybrid(fast_result=fast_r)

        result = await hybrid.analyse("h1", "nice content")

        hybrid._fast.analyse.assert_called_once()
        hybrid._smart.analyse.assert_not_called()
        assert result.confidence == 0.05

    @pytest.mark.asyncio
    async def test_clearly_toxic_skips_smart(self) -> None:
        fast_r = ModerationResult(id="h2", verdict="remove", confidence=0.92, explanation="bad")
        hybrid = self._make_hybrid(fast_result=fast_r)

        result = await hybrid.analyse("h2", "toxic content")

        hybrid._fast.analyse.assert_called_once()
        hybrid._smart.analyse.assert_not_called()
        assert result.verdict == "remove"

    @pytest.mark.asyncio
    async def test_ambiguous_escalates_to_smart(self) -> None:
        fast_r  = ModerationResult(id="h3", verdict="review", confidence=0.45, explanation="maybe")
        smart_r = ModerationResult(id="h3", verdict="remove", confidence=0.91, explanation="yes bad",
                                   flaggedPhrases=["bad word"])
        hybrid  = self._make_hybrid(fast_result=fast_r, smart_result=smart_r)

        result = await hybrid.analyse("h3", "ambiguous text")

        hybrid._fast.analyse.assert_called_once()
        hybrid._smart.analyse.assert_called_once()
        assert result.verdict        == "remove"
        assert result.flaggedPhrases == ["bad word"]

    @pytest.mark.asyncio
    async def test_fast_error_escalates_to_smart(self) -> None:
        fast_r  = ModerationResult(id="h4", verdict="review", confidence=0.0,
                                   explanation="error", error=True)
        smart_r = ModerationResult(id="h4", verdict="safe", confidence=0.02, explanation="fine")
        hybrid  = self._make_hybrid(fast_result=fast_r, smart_result=smart_r)

        result = await hybrid.analyse("h4", "content")

        hybrid._smart.analyse.assert_called_once()
        assert result.verdict == "safe"

    @pytest.mark.asyncio
    async def test_smart_error_falls_back_to_fast(self) -> None:
        fast_r  = ModerationResult(id="h5", verdict="review", confidence=0.45, explanation="maybe")
        smart_r = ModerationResult(id="h5", verdict="review", confidence=0.0,
                                   explanation="error", error=True)
        hybrid  = self._make_hybrid(fast_result=fast_r, smart_result=smart_r)

        result = await hybrid.analyse("h5", "ambiguous")

        # When smart fails we fall back to the fast result
        assert result.confidence == 0.45
        assert result.error      is False

    @pytest.mark.asyncio
    async def test_boundary_at_safe_ceiling(self) -> None:
        """Score exactly at safe_ceiling → still skips smart."""
        fast_r = ModerationResult(id="h6", verdict="safe", confidence=0.15, explanation="ok")
        hybrid = self._make_hybrid(fast_result=fast_r, safe_ceiling=0.15)
        await hybrid.analyse("h6", "content")
        hybrid._smart.analyse.assert_not_called()

    @pytest.mark.asyncio
    async def test_boundary_at_flag_floor(self) -> None:
        """Score exactly at flag_floor → still skips smart."""
        fast_r = ModerationResult(id="h7", verdict="remove", confidence=0.80, explanation="bad")
        hybrid = self._make_hybrid(fast_result=fast_r, flag_floor=0.80)
        await hybrid.analyse("h7", "content")
        hybrid._smart.analyse.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# ModerationService backend wiring
# ══════════════════════════════════════════════════════════════════════════════

class TestModerationServiceWiring:
    @pytest.mark.asyncio
    async def test_model_override_uses_anthropic_backend(self) -> None:
        """force_model triggers a one-shot AnthropicBackend regardless of mode."""
        from app.services.backends.anthropic_backend import AnthropicBackend as AB

        svc = ModerationService(backend=RuleBasedBackend.__new__(RuleBasedBackend))
        svc.backend._model = None  # no detoxify needed

        fake_raw = _safe_raw()
        with patch.object(AB, "_call_claude", new=AsyncMock(return_value=fake_raw)):
            result = await svc.moderate("x1", "text", model="claude-sonnet-4-20250514")

        assert result.verdict == "safe"

    @pytest.mark.asyncio
    async def test_batch_propagates_content_type(self) -> None:
        received: list[str] = []

        async def fake_analyse(content_id, content, content_type="comment", author_id=""):
            received.append(content_type)
            return _safe_result(content_id)

        svc = ModerationService(backend=AnthropicBackend())
        with patch.object(svc.backend, "analyse", side_effect=fake_analyse):
            await svc.moderate_batch(
                [{"id": f"p{i}", "content": "text"} for i in range(3)],
                content_type="post",
            )

        assert all(t == "post" for t in received)
