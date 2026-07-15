"""Regex patterns for Korean AI-writing tells, with S1/S2/S3 severities.

Severity taxonomy (style inspired by the MIT-licensed DaleSeo/korean-skills
humanizer; the patterns themselves are written fresh):

- S1 (critical): constructions that read as machine translation-ese, e.g.
  the double passive 되어지다 or formulaic essay closers.
- S2 (moderate): overuse signals that are rate-gated — normal prose uses
  these forms occasionally, AI text leans on them constantly.
- S3 (weak): structural monotony hints.

Every fired pattern is reported with counts and examples so the advisory
flag stays contestable by a human reviewer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..models import FeatureResult, PatternHit, Severity

WEIGHT = 25.0

# Points per fired pattern by severity, capped at WEIGHT overall.
POINTS = {"S1": 8.0, "S2": 5.0, "S3": 2.0}

_MIN_SENTENCES_FOR_RATIOS = 5


@dataclass(frozen=True)
class RatePattern:
    """A regex whose hit *rate* (per 1000 hangul chars) must exceed a
    threshold to fire — occasional use is normal Korean."""

    name: str
    severity: Severity
    regex: re.Pattern[str]
    per_1000_threshold: float


# --- S1: fire on any occurrence -------------------------------------------
S1_PATTERNS = [
    ("double_passive_되어지다", re.compile(r"되어\s?(지[고는며]|진다|집니다|졌|져)")),
    ("translationese_에_있어서", re.compile(r"에\s?있어서")),
    (
        "formulaic_closer",
        re.compile(r"(결론적으로|요약하자면|종합해\s?보면|마지막으로\s?중요한\s?것은)"),
    ),
]

# --- S2: rate-gated overuse -------------------------------------------------
S2_RATE_PATTERNS = [
    RatePattern("overuse_에_대해", "S2", re.compile(r"에\s?대(해|한)"), 3.0),
    RatePattern("overuse_을_통해", "S2", re.compile(r"[을를]\s?통(해|한)"), 2.0),
    RatePattern("pronoun_overuse_그것_이것", "S2", re.compile(r"(그것|이것)"), 3.0),
]

# Uniform politeness: nearly every sentence ending in 합쇼체 (~니다) style,
# e.g. 합니다/입니다/됩니다/되어집니다.
_POLITE_ENDING = re.compile(r"[가-힣]니다\s*[.!?…]*\s*$")
UNIFORM_POLITENESS_THRESHOLD = 0.9

# --- S3: structural monotony ------------------------------------------------
_CONNECTIVE_START = re.compile(r"^\s*(또한|그리고|하지만|먼저|더불어|아울러)[,\s]")
CONNECTIVE_START_THRESHOLD = 0.30
_ENUMERATION = [re.compile(r"첫\s?째|첫번째|첫\s?번째"), re.compile(r"둘\s?째|두\s?번째")]


def _examples(regex: re.Pattern[str], text: str, limit: int = 2) -> list[str]:
    out = []
    for m in regex.finditer(text):
        start = max(0, m.start() - 12)
        out.append("…" + text[start : m.end() + 12].replace("\n", " ") + "…")
        if len(out) >= limit:
            break
    return out


def find_pattern_hits(sentences: list[str]) -> list[PatternHit]:
    """Evaluate all patterns; returns every fired pattern with evidence."""
    text = "\n".join(sentences)
    hangul_chars = max(sum(1 for ch in text if "가" <= ch <= "힣"), 1)
    hits: list[PatternHit] = []

    for name, regex in S1_PATTERNS:
        matches = regex.findall(text)
        if matches:
            hits.append(
                PatternHit(
                    pattern_name=name,
                    severity="S1",
                    count=len(matches),
                    examples=_examples(regex, text),
                )
            )

    for pat in S2_RATE_PATTERNS:
        count = len(pat.regex.findall(text))
        rate = count * 1000.0 / hangul_chars
        if rate > pat.per_1000_threshold:
            hits.append(
                PatternHit(
                    pattern_name=f"{pat.name} ({rate:.1f}/1000 chars)",
                    severity=pat.severity,
                    count=count,
                    examples=_examples(pat.regex, text),
                )
            )

    if len(sentences) >= _MIN_SENTENCES_FOR_RATIOS:
        polite = sum(1 for s in sentences if _POLITE_ENDING.search(s))
        polite_ratio = polite / len(sentences)
        if polite_ratio >= UNIFORM_POLITENESS_THRESHOLD:
            hits.append(
                PatternHit(
                    pattern_name=f"uniform_politeness_endings ({polite_ratio:.0%})",
                    severity="S2",
                    count=polite,
                )
            )

        connective = sum(1 for s in sentences if _CONNECTIVE_START.search(s))
        if connective / len(sentences) >= CONNECTIVE_START_THRESHOLD:
            hits.append(
                PatternHit(
                    pattern_name="repeated_sentence_initial_connectives",
                    severity="S3",
                    count=connective,
                )
            )

    if all(rx.search(text) for rx in _ENUMERATION):
        hits.append(
            PatternHit(pattern_name="enumeration_scaffolding_첫째_둘째", severity="S3", count=1)
        )

    return hits


def extract(sentences: list[str]) -> FeatureResult:
    hits = find_pattern_hits(sentences)
    contribution = min(sum(POINTS[h.severity] for h in hits), WEIGHT)
    worst: Optional[Severity] = None
    if hits:
        worst = min(hits, key=lambda h: h.severity).severity  # S1 < S2 < S3

    items = [
        f"[{h.severity}] {h.pattern_name} ×{h.count}"
        + (f" (e.g. {h.examples[0]})" if h.examples else "")
        for h in hits
    ]
    evidence = "; ".join(items) if items else "no tell-tale patterns fired"

    return FeatureResult(
        name="pattern_tells",
        contribution=round(contribution, 2),
        max_contribution=WEIGHT,
        severity=worst,
        evidence=evidence,
        evidence_items=items,
    )
