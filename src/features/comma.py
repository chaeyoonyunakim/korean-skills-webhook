"""Comma-usage features.

The KatFishNet paper reports comma usage as the strongest single signal for
LLM-generated Korean (~94.9 AUC on essays): LLM output puts a comma in ~61%
of sentences versus ~26% for human text. We reimplement the *idea* — comma
inclusion rate, comma density, positional regularity, and POS diversity of
the tokens around commas — from scratch.

Sub-signals and their share of the 40-point weight:
- inclusion rate       20  (human anchor 0.26 -> 0, LLM anchor 0.61 -> full)
- commas per sentence  10  (anchors 0.35 -> 0, 1.2 -> full)
- positional regularity 5  (low spread of relative comma positions)
- context POS diversity 5  (same POS pairs around every comma)
"""

from __future__ import annotations

from statistics import mean

from ..models import FeatureResult
from ..segment import get_kiwi

WEIGHT = 40.0

# Calibration anchors derived from figures reported in the paper.
HUMAN_INCLUSION_RATE = 0.26
LLM_INCLUSION_RATE = 0.61
HUMAN_COMMAS_PER_SENT = 0.35
LLM_COMMAS_PER_SENT = 1.20

_MIN_COMMAS_FOR_SHAPE_SIGNALS = 3


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _scale(value: float, low: float, high: float) -> float:
    """Map value linearly from [low, high] onto [0, 1], clamped."""
    return _clamp01((value - low) / (high - low))


def extract(sentences: list[str]) -> FeatureResult:
    if not sentences:
        return FeatureResult(
            name="comma_usage",
            contribution=0.0,
            max_contribution=WEIGHT,
            evidence="no Korean sentences to score",
        )

    comma_counts = [s.count(",") + s.count("、") for s in sentences]
    inclusion_rate = sum(1 for c in comma_counts if c > 0) / len(sentences)
    per_sentence = sum(comma_counts) / len(sentences)

    # Relative positions of commas within their sentences (0=start, 1=end).
    rel_positions: list[float] = []
    context_pairs: set[tuple[str, str]] = set()
    total_commas = 0
    kiwi = get_kiwi()
    for sent in sentences:
        length = max(len(sent), 1)
        for i, ch in enumerate(sent):
            if ch in ",、":
                rel_positions.append(i / length)
        if "," in sent or "、" in sent:
            tokens = kiwi.tokenize(sent)
            for idx, tok in enumerate(tokens):
                if tok.tag == "SP":  # comma/slash separator tag
                    before = tokens[idx - 1].tag if idx > 0 else "BOS"
                    after = tokens[idx + 1].tag if idx + 1 < len(tokens) else "EOS"
                    context_pairs.add((before, after))
                    total_commas += 1

    inclusion_score = _scale(inclusion_rate, HUMAN_INCLUSION_RATE, LLM_INCLUSION_RATE)
    density_score = _scale(per_sentence, HUMAN_COMMAS_PER_SENT, LLM_COMMAS_PER_SENT)

    # Positional regularity: commas always at the same relative spot in the
    # sentence is machine-like; humans scatter them. Needs enough commas.
    position_score = 0.0
    if len(rel_positions) >= _MIN_COMMAS_FOR_SHAPE_SIGNALS:
        avg = mean(rel_positions)
        spread = mean(abs(p - avg) for p in rel_positions)
        position_score = _scale(spread, 0.20, 0.05)  # low spread -> high score

    # POS-context diversity: few distinct (before, after) POS pairs relative
    # to the number of commas means formulaic comma placement.
    context_score = 0.0
    if total_commas >= _MIN_COMMAS_FOR_SHAPE_SIGNALS:
        diversity = len(context_pairs) / total_commas
        context_score = 1.0 - _clamp01(diversity)

    contribution = (
        inclusion_score * 20.0
        + density_score * 10.0
        + position_score * 5.0
        + context_score * 5.0
    )

    items = [
        f"commas in {inclusion_rate:.0%} of sentences "
        f"(human ≈{HUMAN_INCLUSION_RATE:.0%}, LLM ≈{LLM_INCLUSION_RATE:.0%})",
        f"{per_sentence:.2f} commas/sentence",
    ]
    if total_commas >= _MIN_COMMAS_FOR_SHAPE_SIGNALS:
        items.append(f"{len(context_pairs)}/{total_commas} distinct POS contexts")

    return FeatureResult(
        name="comma_usage",
        contribution=round(min(contribution, WEIGHT), 2),
        max_contribution=WEIGHT,
        evidence="; ".join(items),
        evidence_items=items,
    )
