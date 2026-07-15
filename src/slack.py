"""Stage 5/6: build the Slack Block Kit message and post it.

The webhook URL is read from the SLACK_WEBHOOK_URL environment variable —
never hardcoded. Every message carries a plain-text fallback and an
explicit advisory disclaimer.
"""

from __future__ import annotations

import os

import requests

from .models import PostText, ScoreReport, SlackMessage

DISCLAIMER = (
    "⚠️ Advisory flag only — human review required. AI detectors have known "
    "false-positive risks, especially on mixed-language and short text."
)

TOP_FEATURES = 3


def build_slack_message(post: PostText, report: ScoreReport) -> SlackMessage:
    """Assemble Block Kit blocks plus the mandatory plain-text fallback."""
    title = post.title or post.url
    text_fallback = (
        f"Korean AI-tone advisory for '{title}': score {report.score:.0f}/100 "
        f"(Korean ratio {report.korean_ratio:.0%}). {DISCLAIMER}"
    )

    fields = [
        {"type": "mrkdwn", "text": f"*Advisory score:*\n{report.score:.0f} / 100"},
        {"type": "mrkdwn", "text": f"*Korean-text ratio:*\n{report.korean_ratio:.0%}"},
    ]
    if report.short_text_warning:
        fields.append(
            {"type": "mrkdwn", "text": "*Caveat:*\nshort Korean text — noisy signal"}
        )

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Korean AI-tone advisory", "emoji": True},
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": f"<{post.url}|{title}>"}},
        {"type": "section", "fields": fields},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": "*Top signals*"}},
    ]

    # One block per signal (blocks give vertical spacing between them);
    # evidence findings render as quoted sub-lines under the bold title.
    firing = [f for f in report.features if f.contribution > 0][:TOP_FEATURES]
    for f in firing:
        severity = f" `{f.severity}`" if f.severity else ""
        title_line = (
            f"• *{f.name}*{severity} — {f.contribution:.1f}/{f.max_contribution:.0f} pts"
        )
        items = f.evidence_items or [f.evidence]
        quoted = "\n".join(f"> {item}" for item in items)
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": f"{title_line}\n{quoted}"}}
        )
    if not firing:
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": "_no signals fired_"}}
        )

    blocks += [
        {"type": "divider"},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": DISCLAIMER}]},
    ]
    return SlackMessage(text=text_fallback, blocks=blocks)


def post_to_slack(message: SlackMessage, webhook_url: str | None = None, timeout: int = 30) -> None:
    """POST the message to the incoming webhook from SLACK_WEBHOOK_URL."""
    url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        raise RuntimeError(
            "SLACK_WEBHOOK_URL is not set. Export it (see .env.example) or use --dry-run."
        )
    resp = requests.post(
        url, json={"text": message.text, "blocks": message.blocks}, timeout=timeout
    )
    resp.raise_for_status()
