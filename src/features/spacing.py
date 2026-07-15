"""Word-spacing consistency features.

Korean spacing (띄어쓰기) rules are complex and humans apply them
inconsistently; LLMs emit rigidly standard-compliant spacing. We run
Kiwi's spacing normaliser over the text and measure how much it changes:
near-zero corrections means suspiciously standard spacing (AI-like),
frequent corrections means human inconsistency.
"""

from __future__ import annotations

from ..models import FeatureResult
from ..segment import get_kiwi

WEIGHT = 15.0

# Anchor points for the space-correction rate (corrections per word
# boundary). At or below AI_ANCHOR -> full contribution; at or above
# HUMAN_ANCHOR -> zero.
AI_ANCHOR = 0.004
HUMAN_ANCHOR = 0.03

# Below this many hangul characters the signal is pure noise.
MIN_CHARS = 300


def _space_boundary_set(text: str) -> tuple[str, set[int]]:
    """Return the text without spaces plus the despaced indices where a
    space followed, so two spacings of the same text can be diffed."""
    despaced = []
    boundaries: set[int] = set()
    for ch in text:
        if ch.isspace():
            if despaced:
                boundaries.add(len(despaced))
        else:
            despaced.append(ch)
    return "".join(despaced), boundaries


def extract(sentences: list[str]) -> FeatureResult:
    text = " ".join(sentences)
    hangul_count = sum(1 for ch in text if "가" <= ch <= "힣")
    if hangul_count < MIN_CHARS:
        return FeatureResult(
            name="word_spacing",
            contribution=0.0,
            max_contribution=WEIGHT,
            evidence=f"only {hangul_count} hangul chars — too short for a spacing signal",
        )

    kiwi = get_kiwi()
    normalized = kiwi.space(text)

    orig_chars, orig_bounds = _space_boundary_set(text)
    norm_chars, norm_bounds = _space_boundary_set(normalized)
    if orig_chars != norm_chars:
        # The normaliser altered more than spacing; diff would be unreliable.
        return FeatureResult(
            name="word_spacing",
            contribution=0.0,
            max_contribution=WEIGHT,
            evidence="spacing diff unavailable (normaliser changed non-space chars)",
        )

    possible = max(len(orig_bounds | norm_bounds), 1)
    corrections = len(orig_bounds ^ norm_bounds)
    rate = corrections / possible

    # Low correction rate -> rigid standard spacing -> AI-like.
    norm_score = max(0.0, min(1.0, (HUMAN_ANCHOR - rate) / (HUMAN_ANCHOR - AI_ANCHOR)))

    return FeatureResult(
        name="word_spacing",
        contribution=round(norm_score * WEIGHT, 2),
        max_contribution=WEIGHT,
        evidence=(
            f"{corrections} spacing corrections over {possible} word boundaries "
            f"(rate {rate:.3f}; rigid standard spacing is AI-like)"
        ),
    )
