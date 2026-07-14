from __future__ import annotations

from pathlib import Path

from src.models import PostText
from src.scorer import korean_ai_score
from src.segment import segment_korean

FIXTURES = Path(__file__).parent / "fixtures"


def _score(name: str):
    text = (FIXTURES / name).read_text(encoding="utf-8")
    return korean_ai_score(segment_korean(PostText(url="x", title=name, text=text)))


def test_direction_ai_scores_clearly_higher():
    human = _score("human_like.txt")
    ai = _score("ai_like.txt")
    assert ai.score > human.score
    assert ai.score - human.score >= 20  # clear separation, not a coin flip


def test_score_bounds():
    for name in ("human_like.txt", "ai_like.txt"):
        report = _score(name)
        assert 0 <= report.score <= 100


def test_features_reported_with_evidence():
    report = _score("ai_like.txt")
    assert len(report.features) == 4
    assert all(f.evidence for f in report.features)


def test_short_text_warning():
    seg = segment_korean(PostText(url="x", title="t", text="짧은 글이다. 정말 짧다."))
    report = korean_ai_score(seg)
    assert report.short_text_warning
    assert any("short" in n or "noisy" in n for n in report.notes)
