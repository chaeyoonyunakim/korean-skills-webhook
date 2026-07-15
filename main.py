"""CLI for the Korean AI-tone advisory pipeline.

Usage:
    python main.py --url <post-url-or-local-html> [--dry-run]
    python main.py --feed [--feed-url URL] [--dry-run]

Plain route→dispatch over pure pipeline functions — no agent frameworks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure Korean characters print correctly on Windows consoles (cp1252 → utf-8).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.dedupe import DEFAULT_SEEN_FILE, load_seen, save_seen
from src.extract import extract_post_text
from src.fetch_feed import DEFAULT_FEED_URL, fetch_feed
from src.models import PostText, ScoreReport
from src.rewrite import suggest_rewrites
from src.scorer import korean_ai_score
from src.segment import segment_korean
from src.slack import DISCLAIMER, build_slack_message, post_to_slack

HIGH_SCORE_THRESHOLD = 50.0


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal stdlib .env loader: KEY=VALUE lines, # comments, optional
    quotes. Real environment variables always win over .env values."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def _print_report(title: str, report: ScoreReport) -> None:
    print(f"\n=== Korean AI-tone advisory: {title} ===")
    print(
        f"score {report.score:.1f}/100 | Korean ratio {report.korean_ratio:.0%} "
        f"| {report.korean_chars} hangul chars | {report.sentence_count} Korean sentences"
    )
    for f in report.features:
        sev = f" [{f.severity}]" if f.severity else ""
        print(f"  - {f.name}{sev}: {f.contribution:.1f}/{f.max_contribution:.0f} — {f.evidence}")
    for note in report.notes:
        print(f"  note: {note}")
    if report.score >= HIGH_SCORE_THRESHOLD:
        print(
            "  ⚠️ High advisory score. If this text is known to be human-written, "
            "this is a FALSE-POSITIVE finding about the detector — report it, "
            "do not treat it as a verdict."
        )
    print(f"  {DISCLAIMER}")


def _run_pipeline(
    post: PostText, dry_run: bool, force_rewrite: bool
) -> ScoreReport:
    seg = segment_korean(post)
    report = korean_ai_score(seg)
    _print_report(post.title or post.url, report)
    rewrite = suggest_rewrites(seg.sentences, report, force=force_rewrite)
    message = build_slack_message(post, report, rewrite)
    if dry_run:
        print("\n--dry-run: Slack payload JSON --")
        print(json.dumps({"text": message.text, "blocks": message.blocks}, ensure_ascii=False, indent=2))
    else:
        post_to_slack(message)
        print("posted to Slack.")
    return report


def process_post(source: str, dry_run: bool, force_rewrite: bool = False) -> ScoreReport:
    """Run one post (URL or local HTML) through the full pipeline."""
    return _run_pipeline(extract_post_text(source), dry_run, force_rewrite)


def process_text_file(path: str, dry_run: bool, force_rewrite: bool = False) -> ScoreReport:
    """Run a plain-text file through the pipeline (bypasses HTML extraction)."""
    content = Path(path).read_text(encoding="utf-8")
    post = PostText(url=path, title=Path(path).stem, text=content)
    return _run_pipeline(post, dry_run, force_rewrite)


def run_url(args: argparse.Namespace) -> int:
    process_post(args.url, args.dry_run, args.force_rewrite)
    return 0


def run_text_file(args: argparse.Namespace) -> int:
    process_text_file(args.text_file, args.dry_run, args.force_rewrite)
    return 0


def run_feed(args: argparse.Namespace) -> int:
    entries = fetch_feed(args.feed_url)
    first_run = not Path(args.seen_file).exists()
    seen = load_seen(args.seen_file)
    new_entries = [e for e in entries if e.url not in seen]

    if first_run and not args.backfill:
        # Seed the dedupe store instead of blasting every historical post.
        print(
            f"feed: first run — marking {len(entries)} existing entries as seen "
            "without notifying (use --backfill to process them)."
        )
        if not args.dry_run:
            save_seen({e.url for e in entries}, args.seen_file)
        return 0

    print(f"feed: {len(entries)} entries, {len(new_entries)} new")
    for entry in new_entries:
        process_post(entry.url, args.dry_run, args.force_rewrite)
        seen.add(entry.url)
    if new_entries and not args.dry_run:
        save_seen(seen, args.seen_file)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Korean AI-tone advisory → Slack")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--url", help="post URL or local HTML file to score")
    mode.add_argument("--feed", action="store_true", help="scan the Atom feed for new posts")
    mode.add_argument("--text-file", metavar="PATH", help="plain-text file to score (bypasses HTML extraction)")
    parser.add_argument("--feed-url", default=DEFAULT_FEED_URL, help="Atom feed URL")
    parser.add_argument("--seen-file", default=DEFAULT_SEEN_FILE, help="dedupe store path")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the Slack payload instead of posting"
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="on first feed run, process all historical entries instead of seeding the seen-store",
    )
    parser.add_argument(
        "--force-rewrite",
        action="store_true",
        help="run the Gemini rewrite stage regardless of advisory score (for testing)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.feed:
        route = run_feed
    elif args.text_file:
        route = run_text_file
    else:
        route = run_url
    try:
        return route(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
