from __future__ import annotations

from src.features import patterns


def test_direction_ai_scores_higher(human_sentences, ai_sentences):
    human = patterns.extract(human_sentences)
    ai = patterns.extract(ai_sentences)
    assert ai.contribution > human.contribution


def test_ai_fixture_fires_s1(ai_sentences):
    hits = patterns.find_pattern_hits(ai_sentences)
    severities = {h.severity for h in hits}
    assert "S1" in severities  # 되어진다 / 에 있어서 / 결론적으로


def test_double_passive_detected():
    hits = patterns.find_pattern_hits(["이 결과는 자동으로 생성되어집니다."])
    assert any("double_passive" in h.pattern_name for h in hits)


def test_uniform_politeness_detected():
    sents = [f"이것은 {i}번째 문장입니다." for i in range(6)]
    hits = patterns.find_pattern_hits(sents)
    assert any("uniform_politeness" in h.pattern_name for h in hits)


def test_clean_text_fires_nothing():
    hits = patterns.find_pattern_hits(["오늘은 날씨가 좋았다.", "산책을 다녀왔다."])
    assert hits == []


def test_contribution_capped(ai_sentences):
    assert patterns.extract(ai_sentences).contribution <= patterns.WEIGHT
