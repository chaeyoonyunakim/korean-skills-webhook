from __future__ import annotations

from src.models import PostText
from src.segment import hangul_ratio, segment_korean


def test_hangul_ratio_pure_korean():
    assert hangul_ratio("안녕하세요") == 1.0


def test_hangul_ratio_mixed():
    ratio = hangul_ratio("NLTK로 tokenize를 했다")
    assert 0.2 < ratio < 0.8


def test_segment_filters_english_only_sentences():
    text = "이 프로젝트는 재미있었다.\nThis sentence is entirely English, nothing else.\n다음에 또 해보고 싶다."
    seg = segment_korean(PostText(url="x", title="t", text=text))
    assert len(seg.sentences) == 2
    assert all("English" not in s for s in seg.sentences)
    assert 0 < seg.korean_ratio < 1
    assert seg.korean_chars > 0


def test_segment_empty_text():
    seg = segment_korean(PostText(url="x", title="t", text=""))
    assert seg.sentences == []
    assert seg.korean_ratio == 0.0
