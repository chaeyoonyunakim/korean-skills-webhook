"""Stage 4: combine feature extractors into a 0-100 advisory score.

The score is advisory only. Feature weights sum to 100 (comma 40,
patterns 25, POS diversity 20, spacing 15) and every contribution is
reported so the flag can be contested by a human reviewer.
"""

from __future__ import annotations

from .features import comma, patterns, pos_ngrams, spacing
from .models import ScoreReport, SegmentedText

SHORT_TEXT_HANGUL_CHARS = 300

_EXTRACTORS = (comma.extract, patterns.extract, pos_ngrams.extract, spacing.extract)


def korean_ai_score(seg: SegmentedText) -> ScoreReport:
    """Run all feature extractors over the Korean sentences of a post."""
    features = [extract(seg.sentences) for extract in _EXTRACTORS]
    score = min(sum(f.contribution for f in features), 100.0)

    notes: list[str] = []
    short_text = seg.korean_chars < SHORT_TEXT_HANGUL_CHARS
    if short_text:
        notes.append(
            f"Only {seg.korean_chars} hangul chars — scores on short text are noisy."
        )
    if seg.korean_ratio < 0.8:
        notes.append(
            f"Mixed-language text (Korean ratio {seg.korean_ratio:.0%}); "
            "only Korean-bearing sentences were scored."
        )

    return ScoreReport(
        score=round(score, 1),
        korean_ratio=seg.korean_ratio,
        korean_chars=seg.korean_chars,
        sentence_count=len(seg.sentences),
        features=sorted(features, key=lambda f: f.contribution, reverse=True),
        short_text_warning=short_text,
        notes=notes,
    )
