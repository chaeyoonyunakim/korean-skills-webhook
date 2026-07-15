# korean-skills-webhook

A minimal, dependency-light Python pipeline that scores Korean blog posts for
**AI-generated-sounding prose** and posts an **advisory** notification to a
Slack channel via incoming webhook.

> ⚠️ **Advisory flag only — human review required.** This tool never issues a
> verdict. Rule-based AI-text detection has well-known false-positive risks,
> especially on mixed-language and short text. Every notification lists the
> exact features that fired so the flag can be contested.

## How it works

Plain route→dispatch over pure functions, with Pydantic v2 models carrying
state between stages — no agent frameworks:

```
fetch_feed → extract_post_text → segment_korean → korean_ai_score
    → build_slack_message → post_to_slack
```

The detector reimplements linguistic feature *ideas* from
[KatFishNet](https://arxiv.org/abs/2503.00032) (ACL 2025) from scratch — the
reference repository has no licence, so no code was copied. Severity taxonomy
style (S1/S2/S3) is inspired by the MIT-licensed
[DaleSeo/korean-skills](https://github.com/DaleSeo/korean-skills) humanizer;
the patterns themselves are written fresh.

| Feature family | Weight | Signal |
| --- | --- | --- |
| Comma usage | 40 | LLM Korean puts commas in ~61% of sentences vs ~26% for humans (the paper's strongest signal); plus comma density, positional regularity, POS diversity around commas |
| Pattern tells | 25 | S1: double passive (되어지다), translation-ese (~에 있어서), formulaic closers · S2: rate-gated overuse (~에 대해, ~을 통해, 그것/이것), uniform ~니다 endings · S3: structural monotony (또한/그리고 starts, 첫째/둘째 scaffolding) |
| POS n-gram diversity | 20 | Low MATTR over POS 1/2/3-grams (kiwipiepy tagger) → repetitive grammar → AI-like |
| Word spacing | 15 | Rigid standard-compliant 띄어쓰기 is AI-like; humans are inconsistent (measured via Kiwi's spacing normaliser) |

Mixed-language posts are handled by scoring only Korean-bearing sentences
(hangul ≥ 30% of non-space chars); embedded English terms are kept but tagged
as foreign tokens and excluded where they would distort POS statistics. The
Korean-text ratio is always reported.

## Setup

Python 3.11+, CPU-only (kiwipiepy ships prebuilt wheels).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or requirements-dev.txt for tests
cp .env.example .env                   # fill in SLACK_WEBHOOK_URL
```

### Creating the Slack incoming webhook

1. Go to <https://api.slack.com/apps> → **Create New App** → *From scratch*.
2. In the app, open **Incoming Webhooks**, toggle it on, and click
   **Add New Webhook to Workspace**, picking the target channel.
3. Copy the generated `https://hooks.slack.com/services/…` URL into `.env`
   locally, and into the repo secret `SLACK_WEBHOOK_URL`
   (Settings → Secrets and variables → Actions) for the scheduled workflow.

The URL is a credential — never commit it. `.env` is gitignored.

## Usage

```bash
# Score one post (URL or local HTML file); print the Slack payload instead of posting
python main.py --url https://chaeyoonyunakim.github.io/2026/07/03/survey-sent-analysis-reflections/ --dry-run

# Scan the Atom feed for new posts and notify (dedupes via seen_posts.json)
python main.py --feed

# First feed run seeds seen_posts.json without notifying; force processing with
python main.py --feed --backfill
```

Console output shows the extraction stats, the 0–100 advisory score, and the
per-feature contributions with evidence.

## Tests

```bash
pytest
```

Unit tests verify each feature extractor's *direction* using two synthetic
fixtures — one human-like (irregular commas, mixed endings, inconsistent
spacing) and one AI-like (heavy commas, translation-ese, uniform ~니다
endings) — plus extraction against a saved copy of the real test post.

## Scheduled monitoring

`.github/workflows/monitor.yml` runs daily at 09:17 UTC (odd offset to dodge
the top-of-hour cron rush) and on manual `workflow_dispatch`. It restores
`seen_posts.json` from the Actions cache, scans the feed, and posts advisories
for new posts using the `SLACK_WEBHOOK_URL` repo secret.

## Ethics & known limitations

This is an experiment, currently pointed at a **human-written** post as a
false-positive stress test — the expected outcome is a *low* score, and a high
score is a finding about the detector, not about the author.

- **No verdicts.** The score is advisory; the Slack message and console output
  say so explicitly and list every fired feature so a human can contest it.
- **Short text is noisy.** Below ~300 hangul characters the statistical
  features are unreliable; the report carries an explicit warning and the
  spacing/POS features go neutral.
- **Mixed-language distortion.** Korean prose with embedded English is common
  in technical writing; only Korean spans are scored, but heavy code/term
  density still degrades the signal. Example from the test post: the S1
  pattern `~에 있어서` matched the literal locative "데이터가 손에 있어서"
  ("the data was at hand") — a genuine false positive, visible in the
  evidence snippet.
- **Rule-based ≠ ground truth.** Thresholds are calibrated from figures
  reported for essays in the KatFishNet paper; blogs are a different register.
- **GitHub cron is best-effort.** Scheduled runs can be delayed or skipped
  under load, and GitHub disables scheduled workflows after **60 days** of
  repository inactivity — re-enable from the Actions tab.
- **Dedupe state is best-effort.** `seen_posts.json` lives in the Actions
  cache; eviction (~7 days unused) can cause a repeat notification.
