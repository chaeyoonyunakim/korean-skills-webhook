from __future__ import annotations

from src.features import pos_ngrams


def test_direction_ai_scores_at_least_as_high(human_sentences, ai_sentences):
    human = pos_ngrams.extract(human_sentences)
    ai = pos_ngrams.extract(ai_sentences)
    assert ai.contribution >= human.contribution


def test_short_text_is_neutral():
    result = pos_ngrams.extract(["짧다."])
    assert result.contribution == 0.0
    assert "too short" in result.evidence


def test_bounds(ai_sentences):
    result = pos_ngrams.extract(ai_sentences)
    assert 0 <= result.contribution <= pos_ngrams.WEIGHT


def test_mattr_repetitive_vs_varied():
    repetitive = ["a", "b"] * 100
    varied = [str(i) for i in range(200)]
    assert pos_ngrams._mattr(repetitive, 50) < pos_ngrams._mattr(varied, 50)
