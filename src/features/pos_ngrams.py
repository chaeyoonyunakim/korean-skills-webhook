"""POS n-gram diversity features.

LLM-generated Korean tends to reuse the same grammatical constructions,
which shows up as low diversity in part-of-speech n-gram sequences. We
tag with Kiwi, drop foreign-language tokens (tag SL — embedded English
terms in mixed-language posts would distort the signal), and compute a
moving-average type-token ratio (MATTR) over POS 1/2/3-grams. MATTR is
length-robust, unlike a raw distinct/total ratio.
"""

from __future__ import annotations

from statistics import mean

from ..models import FeatureResult
from ..segment import get_kiwi

WEIGHT = 20.0

# MATTR anchors (averaged over n=1..3): at or above HUMAN_ANCHOR -> 0
# contribution, at or below AI_ANCHOR -> full contribution.
HUMAN_ANCHOR = 0.62
AI_ANCHOR = 0.42

MATTR_WINDOW = 50
MIN_TOKENS = 60
DAMPEN_BELOW_TOKENS = 200

# Tags excluded from the POS sequence: foreign script and pure symbols.
_EXCLUDED_TAGS = {"SL", "SH", "SW", "SE", "SO", "SB"}


def _mattr(items: list[str], window: int) -> float:
    """Moving-average type-token ratio; falls back to plain TTR when the
    sequence is shorter than the window."""
    if not items:
        return 1.0
    if len(items) <= window:
        return len(set(items)) / len(items)
    ratios = []
    for start in range(len(items) - window + 1):
        chunk = items[start : start + window]
        ratios.append(len(set(chunk)) / window)
    return mean(ratios)


def extract(sentences: list[str]) -> FeatureResult:
    kiwi = get_kiwi()
    tags: list[str] = []
    for sent in sentences:
        for tok in kiwi.tokenize(sent):
            if tok.tag not in _EXCLUDED_TAGS:
                tags.append(tok.tag)

    if len(tags) < MIN_TOKENS:
        return FeatureResult(
            name="pos_ngram_diversity",
            contribution=0.0,
            max_contribution=WEIGHT,
            evidence=f"only {len(tags)} POS tokens — too short for a diversity signal",
        )

    mattrs = []
    for n in (1, 2, 3):
        ngrams = ["|".join(tags[i : i + n]) for i in range(len(tags) - n + 1)]
        mattrs.append(_mattr(ngrams, MATTR_WINDOW))
    avg_mattr = mean(mattrs)

    norm = max(0.0, min(1.0, (HUMAN_ANCHOR - avg_mattr) / (HUMAN_ANCHOR - AI_ANCHOR)))
    if len(tags) < DAMPEN_BELOW_TOKENS:
        norm *= len(tags) / DAMPEN_BELOW_TOKENS

    return FeatureResult(
        name="pos_ngram_diversity",
        contribution=round(norm * WEIGHT, 2),
        max_contribution=WEIGHT,
        evidence=(
            f"POS 1/2/3-gram MATTR = {mattrs[0]:.2f}/{mattrs[1]:.2f}/{mattrs[2]:.2f} "
            f"over {len(tags)} tokens (low diversity is AI-like)"
        ),
    )
