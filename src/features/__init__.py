"""Feature extractors for the Korean AI-tone advisory score.

Each module exposes ``extract(sentences: list[str]) -> FeatureResult``.
Weights (max contributions) sum to 100:

- comma.py       40  (strongest signal in the KatFishNet paper)
- pos_ngrams.py  20
- patterns.py    25
- spacing.py     15
"""

from . import comma, patterns, pos_ngrams, spacing  # noqa: F401
