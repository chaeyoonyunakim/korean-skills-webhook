from __future__ import annotations

from src.features import comma


def test_direction_ai_scores_higher(human_sentences, ai_sentences):
    human = comma.extract(human_sentences)
    ai = comma.extract(ai_sentences)
    assert ai.contribution > human.contribution


def test_bounds(ai_sentences):
    result = comma.extract(ai_sentences)
    assert 0 <= result.contribution <= comma.WEIGHT


def test_empty_input():
    assert comma.extract([]).contribution == 0.0


def test_evidence_mentions_inclusion_rate(ai_sentences):
    assert "commas in" in comma.extract(ai_sentences).evidence
