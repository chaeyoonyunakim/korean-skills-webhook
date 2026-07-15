"""Pydantic v2 state models passed between pipeline stages."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

Severity = Literal["S1", "S2", "S3"]


class FeedEntry(BaseModel):
    """A single entry discovered in the Atom feed."""

    title: str
    url: str
    published: Optional[datetime] = None


class PostText(BaseModel):
    """Extracted plain text of one post."""

    url: str
    title: str
    text: str


class SegmentedText(BaseModel):
    """Language-segmented view of a post.

    ``sentences`` holds only Korean-bearing sentences (hangul share >= 30%
    of non-space chars); embedded English terms inside them are kept and
    handled downstream by the POS tagger.
    """

    full_text: str
    sentences: list[str]
    korean_ratio: float = Field(ge=0.0, le=1.0)
    total_chars: int
    korean_chars: int


class PatternHit(BaseModel):
    """One fired tell-tale pattern (contestability: shown to reviewers)."""

    pattern_name: str
    severity: Severity
    count: int
    examples: list[str] = Field(default_factory=list)


class FeatureResult(BaseModel):
    """Contribution of one feature family to the advisory score."""

    name: str
    contribution: float = Field(ge=0.0)
    max_contribution: float
    severity: Optional[Severity] = None
    evidence: str


class ScoreReport(BaseModel):
    """Advisory score with per-feature breakdown. Never a verdict."""

    score: float = Field(ge=0.0, le=100.0)
    korean_ratio: float
    korean_chars: int
    sentence_count: int
    features: list[FeatureResult]
    short_text_warning: bool = False
    notes: list[str] = Field(default_factory=list)


class SlackMessage(BaseModel):
    """Block Kit payload plus mandatory plain-text fallback."""

    text: str
    blocks: list[dict]
