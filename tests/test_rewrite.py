"""Tests for src/rewrite.py and the rewrite section of src/slack.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.models import (
    FeatureResult,
    PostText,
    RewriteResult,
    RewriteSuggestion,
    ScoreReport,
)
from src.rewrite import (
    DEFAULT_REWRITE_THRESHOLD,
    MAX_SUGGESTIONS,
    _parse_suggestions,
    build_prompt,
    select_sentences,
    suggest_rewrites,
)
from src.slack import _REWRITE_NOTE, build_slack_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report(score: float = 42.0) -> ScoreReport:
    return ScoreReport(
        score=score,
        korean_ratio=0.80,
        korean_chars=800,
        sentence_count=15,
        features=[
            FeatureResult(
                name="pattern_tells",
                contribution=score * 0.4,
                max_contribution=25,
                severity="S1",
                evidence="[S1] translationese_에_있어서 ×3",
            ),
            FeatureResult(name="comma_usage", contribution=score * 0.3, max_contribution=40, evidence="e"),
            FeatureResult(name="pos_ngram_diversity", contribution=0, max_contribution=20, evidence="e"),
            FeatureResult(name="word_spacing", contribution=0, max_contribution=15, evidence="e"),
        ],
    )


def _suggestion(n: int = 1) -> RewriteSuggestion:
    return RewriteSuggestion(
        original=f"원본 문장 {n}.",
        revised=f"수정된 문장 {n}.",
        reason=f"이유 {n}.",
        pattern_id="translationese_에_있어서",
    )


def _gemini_response(suggestions: list[dict]) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps(suggestions, ensure_ascii=False)}]}}],
        "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 60},
    }


# ---------------------------------------------------------------------------
# Trigger logic
# ---------------------------------------------------------------------------

def test_below_threshold_skips(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    result = suggest_rewrites(["문장입니다."], _report(score=39.0))
    assert result.skipped
    assert "below threshold" in result.skip_reason


def test_at_threshold_calls_api(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    payload = [{"original": "분석에 있어서 중요합니다.", "revised": "분석에서 중요합니다.", "reason": "이유.", "pattern_id": "translationese_에_있어서"}]
    with patch("src.rewrite._call_gemini_raw", return_value=_gemini_response(payload)):
        result = suggest_rewrites(
            ["분석에 있어서 중요합니다.", "그것을 통해 알 수 있습니다."],
            _report(score=40.0),
        )
    assert not result.skipped
    assert len(result.suggestions) >= 1


def test_missing_api_key_skips(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = suggest_rewrites(["문장입니다."], _report(score=50.0))
    assert result.skipped
    assert "GEMINI_API_KEY" in result.skip_reason


def test_force_bypasses_threshold(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    payload = [{"original": "문장입니다.", "revised": "문장이에요.", "reason": "이유.", "pattern_id": "high_comma_density"}]
    with patch("src.rewrite._call_gemini_raw", return_value=_gemini_response(payload)):
        result = suggest_rewrites(["문장입니다."], _report(score=5.0), force=True)
    assert not result.skipped


# ---------------------------------------------------------------------------
# Sentence selection
# ---------------------------------------------------------------------------

S1_SENTENCE = "이번 분석에 있어서, 가장 중요한 것은 데이터입니다."
S2_SENTENCE = "이 방법을 통해, 결과를 얻을 수 있었습니다."
COMMA_DENSE = "가나, 나다, 다라, 라마, 마바입니다."  # 4 commas, short → high density
PLAIN_SENTENCE = "결과가 좋았습니다."


def test_s1_sentence_is_tier_1():
    report = _report(score=50.0)
    selections = select_sentences([S1_SENTENCE, PLAIN_SENTENCE], report)
    assert selections[0].tier == 1
    assert selections[0].pattern_id == "translationese_에_있어서"


def test_s1_before_comma_dense():
    report = _report(score=50.0)
    # S1 sentence and a very comma-dense plain sentence
    selections = select_sentences([COMMA_DENSE, S1_SENTENCE], report)
    tiers = [s.tier for s in selections]
    assert tiers[0] <= tiers[-1]  # lower tier number (higher priority) comes first


def test_comma_density_is_proportional():
    report = _report(score=50.0)
    # Long sentence with 1 comma → low rate; short sentence with 2 commas → high rate
    long_one_comma = "가" * 100 + ", " + "나" * 100 + "입니다."
    short_two_commas = "가나, 나다, 다라입니다."
    selections = select_sentences([long_one_comma, short_two_commas], report)
    # short_two_commas should be tier 3 or lower; long_one_comma should not beat it
    ids = [s.sentence for s in selections]
    assert short_two_commas in ids


def test_fallback_ensures_min_three():
    # Very plain sentences — no patterns, low comma density → fallback kicks in
    plain = [f"문장 {i}입니다." for i in range(5)]
    report = _report(score=50.0)
    selections = select_sentences(plain, report)
    assert len(selections) >= 3


def test_max_count_respected():
    sentences = [S1_SENTENCE] * 10
    report = _report(score=50.0)
    selections = select_sentences(sentences, report, max_count=3)
    assert len(selections) <= 3


# ---------------------------------------------------------------------------
# Pydantic / JSON parsing
# ---------------------------------------------------------------------------

def test_parse_good_json():
    raw = json.dumps([
        {"original": "원본.", "revised": "수정본.", "reason": "이유.", "pattern_id": "translationese_에_있어서"}
    ])
    suggestions = _parse_suggestions(raw)
    assert len(suggestions) == 1
    assert suggestions[0].revised == "수정본."


def test_parse_malformed_json_raises():
    with pytest.raises(Exception):
        _parse_suggestions("not json at all")


def test_parse_wrong_schema_raises():
    raw = json.dumps([{"wrong_key": "value"}])
    with pytest.raises(Exception):
        _parse_suggestions(raw)


def test_deduplication_drops_unchanged(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    payload = [
        {"original": "원본.", "revised": "원본.", "reason": "변경 없음.", "pattern_id": "translationese_에_있어서"},
    ]
    with patch("src.rewrite._call_gemini_raw", return_value=_gemini_response(payload)):
        result = suggest_rewrites(
            ["원본.", "분석에 있어서 중요합니다."],
            _report(score=50.0),
            force=True,
        )
    assert all(s.revised != s.original for s in result.suggestions)


def test_retry_on_bad_json_then_succeeds(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    good_payload = [{"original": "분석에 있어서.", "revised": "분석에서.", "reason": "이유.", "pattern_id": "translationese_에_있어서"}]
    bad_response = {"candidates": [{"content": {"parts": [{"text": "not json"}]}}], "usageMetadata": {}}
    good_response = _gemini_response(good_payload)
    call_count = 0

    def mock_call(prompt, api_key, **kw):
        nonlocal call_count
        call_count += 1
        return bad_response if call_count == 1 else good_response

    with patch("src.rewrite._call_gemini_raw", side_effect=mock_call):
        result = suggest_rewrites(["분석에 있어서."], _report(score=50.0), force=True)
    assert call_count == 2
    assert not result.skipped


def test_two_bad_json_degrades_gracefully(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    bad = {"candidates": [{"content": {"parts": [{"text": "bad"}]}}], "usageMetadata": {}}
    with patch("src.rewrite._call_gemini_raw", return_value=bad):
        result = suggest_rewrites(["분석에 있어서."], _report(score=50.0), force=True)
    assert result.skipped
    assert "unparseable" in result.skip_reason.lower()


def test_api_error_degrades_gracefully(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    with patch("src.rewrite._call_gemini_raw", side_effect=ConnectionError("timeout")):
        result = suggest_rewrites(["분석에 있어서."], _report(score=50.0), force=True)
    assert result.skipped
    assert "API error" in result.skip_reason


# ---------------------------------------------------------------------------
# Slack block rendering
# ---------------------------------------------------------------------------

def _post() -> PostText:
    return PostText(url="https://example.com/p", title="테스트", text="본문")


def test_slack_no_rewrite_no_suggestion_blocks():
    msg = build_slack_message(_post(), _report())
    all_text = " ".join(
        b.get("text", {}).get("text", "") for b in msg.blocks
    )
    assert "Suggested revisions" not in all_text
    assert _REWRITE_NOTE not in all_text


def test_slack_skipped_adds_footer_context():
    rewrite = RewriteResult(skipped=True, skip_reason="score 9 below threshold 40")
    msg = build_slack_message(_post(), _report(), rewrite)
    contexts = [b for b in msg.blocks if b["type"] == "context"]
    footer_texts = [b["elements"][0]["text"] for b in contexts]
    assert any("below threshold" in t for t in footer_texts)


def test_slack_three_suggestions_renders_correctly():
    rewrite = RewriteResult(suggestions=[_suggestion(i) for i in range(3)])
    msg = build_slack_message(_post(), _report(), rewrite)
    assert any(b["type"] == "divider" for b in msg.blocks)
    suggestion_blocks = [
        b for b in msg.blocks
        if b["type"] == "section" and "Before:" in b.get("text", {}).get("text", "")
    ]
    assert len(suggestion_blocks) == 3
    assert str(len(rewrite.suggestions)) in msg.text


def test_slack_six_suggestions_capped_at_five():
    rewrite = RewriteResult(suggestions=[_suggestion(i) for i in range(6)])
    msg = build_slack_message(_post(), _report(), rewrite)
    suggestion_blocks = [
        b for b in msg.blocks
        if b["type"] == "section" and "Before:" in b.get("text", {}).get("text", "")
    ]
    assert len(suggestion_blocks) == MAX_SUGGESTIONS


def test_slack_long_sentence_truncated():
    long = "가" * 300
    rewrite = RewriteResult(suggestions=[
        RewriteSuggestion(original=long, revised=long[:10] + "수정", reason="이유.", pattern_id="pid")
    ])
    msg = build_slack_message(_post(), _report(), rewrite)
    suggestion_text = next(
        b["text"]["text"] for b in msg.blocks
        if b["type"] == "section" and "Before:" in b.get("text", {}).get("text", "")
    )
    assert "…" in suggestion_text
    assert len(suggestion_text) < 3000


def test_slack_block_count_within_limit():
    rewrite = RewriteResult(suggestions=[_suggestion(i) for i in range(MAX_SUGGESTIONS)])
    msg = build_slack_message(_post(), _report(), rewrite)
    assert len(msg.blocks) <= 50
