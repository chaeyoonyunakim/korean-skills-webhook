from __future__ import annotations

from src.features import spacing


def test_direction_ai_scores_at_least_as_high(human_sentences, ai_sentences):
    human = spacing.extract(human_sentences)
    ai = spacing.extract(ai_sentences)
    assert ai.contribution >= human.contribution


def test_short_text_is_neutral():
    result = spacing.extract(["짧은 문장입니다."])
    assert result.contribution == 0.0
    assert "too short" in result.evidence


def test_bounds(ai_sentences):
    result = spacing.extract(ai_sentences)
    assert 0 <= result.contribution <= spacing.WEIGHT
