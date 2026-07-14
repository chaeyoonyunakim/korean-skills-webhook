from __future__ import annotations

import pytest

from src.models import FeatureResult, PostText, ScoreReport, SlackMessage
from src.slack import DISCLAIMER, build_slack_message, post_to_slack


def _report() -> ScoreReport:
    return ScoreReport(
        score=42.0,
        korean_ratio=0.62,
        korean_chars=1500,
        sentence_count=50,
        features=[
            FeatureResult(name="comma_usage", contribution=20, max_contribution=40, evidence="e1"),
            FeatureResult(
                name="pattern_tells", contribution=15, max_contribution=25, severity="S1", evidence="e2"
            ),
            FeatureResult(name="pos_ngram_diversity", contribution=5, max_contribution=20, evidence="e3"),
            FeatureResult(name="word_spacing", contribution=2, max_contribution=15, evidence="e4"),
        ],
        short_text_warning=False,
    )


def test_message_has_text_fallback_and_blocks():
    msg = build_slack_message(PostText(url="https://example.com/p", title="제목", text="본문"), _report())
    assert isinstance(msg, SlackMessage)
    assert "42" in msg.text
    assert DISCLAIMER in msg.text
    assert any(b["type"] == "header" for b in msg.blocks)


def test_disclaimer_block_present():
    msg = build_slack_message(PostText(url="u", title="t", text=""), _report())
    contexts = [b for b in msg.blocks if b["type"] == "context"]
    assert contexts and DISCLAIMER in contexts[-1]["elements"][0]["text"]


def test_top_three_features_only():
    msg = build_slack_message(PostText(url="u", title="t", text=""), _report())
    signals = next(
        b["text"]["text"] for b in msg.blocks if b.get("text", {}).get("text", "").startswith("*Top signals:*")
    )
    assert "comma_usage" in signals and "[S1]" in signals
    assert "word_spacing" not in signals  # 4th feature is cut


def test_post_without_webhook_env_raises(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    msg = build_slack_message(PostText(url="u", title="t", text=""), _report())
    with pytest.raises(RuntimeError, match="SLACK_WEBHOOK_URL"):
        post_to_slack(msg)
