"""Stage 5/6: build the Slack Block Kit message and post it.

The webhook URL is read from the SLACK_WEBHOOK_URL environment variable —
never hardcoded. Every message carries a plain-text fallback and an
explicit advisory disclaimer.
"""

from __future__ import annotations

import os

import requests

from .models import PostText, RewriteResult, ScoreReport, SlackMessage

DISCLAIMER = (
    "⚠️ Advisory flag only — human review required. AI detectors have known "
    "false-positive risks, especially on mixed-language and short text."
)

TOP_FEATURES = 3
MAX_REWRITE_BLOCKS = 5
_REWRITE_NOTE = (
    "Suggestions are optional — accept, edit, or ignore each one. "
    "The author's voice takes priority."
)


def _trunc(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def build_slack_message(
    post: PostText,
    report: ScoreReport,
    rewrite: RewriteResult | None = None,
) -> SlackMessage:
    """Assemble Block Kit blocks plus the mandatory plain-text fallback."""
    title = post.title or post.url
    rewrite_summary = ""
    if rewrite and rewrite.suggestions:
        rewrite_summary = f" {len(rewrite.suggestions)} rewrite suggestion(s) included."
    text_fallback = (
        f"Korean AI-tone advisory for '{title}': score {report.score:.0f}/100 "
        f"(Korean ratio {report.korean_ratio:.0%}).{rewrite_summary} {DISCLAIMER}"
    )

    firing = [f for f in report.features if f.contribution > 0][:TOP_FEATURES]
    if firing:
        feature_lines = "\n".join(
            f"• *{f.name}*{f' [{f.severity}]' if f.severity else ''} — "
            f"{f.contribution:.1f}/{f.max_contribution:.0f} pts: {f.evidence}"
            for f in firing
        )
    else:
        feature_lines = "• no features fired"

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
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Top signals:*\n{feature_lines}"},
        },
        {"type": "context", "elements": [{"type": "mrkdwn", "text": DISCLAIMER}]},
    ]

    if rewrite and rewrite.suggestions:
        shown = rewrite.suggestions[:MAX_REWRITE_BLOCKS]
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Suggested revisions (optional)*"},
        })
        for s in shown:
            body = (
                f"*Before:* {_trunc(s.original)}\n"
                f"*After:* {_trunc(s.revised)}\n"
                f"*Why:* {s.reason} (`{s.pattern_id}`)"
            )
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": body}})
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": _REWRITE_NOTE}],
        })
    elif rewrite and rewrite.skipped:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"Rewrite suggestions: {rewrite.skip_reason}."}],
        })

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
