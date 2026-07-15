"""Stage 3: language detection / segmentation for mixed-language posts.

Korean prose with embedded English technical terms is the expected input.
We compute a whole-text Korean ratio, then keep only sentences whose hangul
share is high enough to be meaningfully Korean; short English-only lines
(link lists, term definitions) are excluded from tone scoring.
"""

from __future__ import annotations

import re
from functools import lru_cache

from kiwipiepy import Kiwi

from .models import PostText, SegmentedText

_HANGUL_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
_NONSPACE_RE = re.compile(r"\S")

# Minimum hangul share (of non-space chars) for a sentence to be scored.
KOREAN_SENTENCE_THRESHOLD = 0.30


@lru_cache(maxsize=1)
def get_kiwi() -> Kiwi:
    """Shared Kiwi instance — model load takes ~1s, so build it once."""
    return Kiwi()


def hangul_ratio(text: str) -> float:
    """Share of hangul among non-space characters (0.0 for empty text)."""
    nonspace = len(_NONSPACE_RE.findall(text))
    if nonspace == 0:
        return 0.0
    return len(_HANGUL_RE.findall(text)) / nonspace


def segment_korean(post: PostText) -> SegmentedText:
    """Split into sentences and keep the Korean-bearing ones."""
    text = post.text
    kiwi = get_kiwi()
    # The extractor emits one block per line; treat newlines as hard sentence
    # boundaries (Kiwi does not always split after non-Korean sentences).
    sentences = [
        s.text.strip()
        for line in text.split("\n")
        if line.strip()
        for s in kiwi.split_into_sents(line)
    ]
    korean_sentences = [
        s for s in sentences if s and hangul_ratio(s) >= KOREAN_SENTENCE_THRESHOLD
    ]
    return SegmentedText(
        full_text=text,
        sentences=korean_sentences,
        korean_ratio=round(hangul_ratio(text), 4),
        total_chars=len(text),
        korean_chars=len(_HANGUL_RE.findall(text)),
    )
