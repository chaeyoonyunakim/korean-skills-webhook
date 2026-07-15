"""Stage 5 (optional): sentence selection + Gemini-powered rewrite suggestions.

Only runs when advisory_score >= REWRITE_THRESHOLD (env var, default 40).
All selected sentences are batched into a single Gemini call (free-tier friendly).
Degrades gracefully on any API or parse failure — never crashes the pipeline.
"""

from __future__ import annotations

import json
import os
import re
from statistics import mean

import requests
from pydantic import ValidationError

from .features.comma import LLM_COMMAS_PER_SENT
from .features.patterns import S1_PATTERNS, S2_RATE_PATTERNS, find_pattern_hits
from .models import RewriteResult, RewriteSuggestion, ScoreReport, SentenceSelection

DEFAULT_REWRITE_THRESHOLD = 40.0
_GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _redact(text: str, key: str) -> str:
    """Strip the API key from error strings so it is never logged or returned."""
    return text.replace(key, "***") if key else text
MAX_SUGGESTIONS = 5
_MAX_SENTENCE_CHARS = 200
_POLITE_ENDING = re.compile(r"[가-힣]니다\s*[.!?…]*\s*$")

PATTERN_GUIDANCE: dict[str, tuple[str, str]] = {
    "double_passive_되어지다": (
        "이중 피동 (double passive)",
        "Replace 되어지다 with simpler 되다",
    ),
    "translationese_에_있어서": (
        "번역 투 ~에 있어서 (〜において calque)",
        "Replace with ~에서 / ~의 / or drop entirely",
    ),
    "formulaic_closer": (
        "공식 결론 문구 (formulaic essay closer)",
        "Rephrase as a natural concluding thought",
    ),
    "overuse_에_대해": (
        "~에 대해/한 남용",
        "Use a more specific relational verb or restructure",
    ),
    "overuse_을_통해": (
        "~을/를 통해 남용",
        "Replace with a direct verb or a more specific preposition",
    ),
    "pronoun_overuse_그것_이것": (
        "그것/이것 남용",
        "Substitute the actual referent noun, or restructure",
    ),
    "uniform_politeness_endings": (
        "획일적 합니다체 어미",
        "Vary with 해요체, nominalization, or restructure some sentences",
    ),
    "high_comma_density": (
        "높은 쉼표 밀도",
        "Break into two sentences or restructure to remove unnecessary pauses",
    ),
}

_PROMPT_TEMPLATE = """\
You are a Korean writing assistant. An author wants light, targeted edits to
sentences that triggered AI-tone signals in a detector. Your job is to make
the *minimum change* that removes the flagged pattern while keeping everything
else identical.

Hard rules — violating any of these makes your response unusable:
1. Preserve meaning exactly. Do not add, omit, or infer any information.
2. Keep all embedded English technical terms unchanged (e.g. API, F1 score,
   survey, recall).
3. Keep the author's register. If the sentence is informal (반말/해체), keep
   it informal. If formal (합니다체), you may vary the ending but do not
   escalate formality.
4. Do NOT polish, beautify, or fix anything beyond the flagged pattern.
5. If you genuinely cannot improve a sentence, copy it unchanged into
   "revised" — that is a valid answer.

Return a JSON array with exactly one object per input sentence, in the same
order. Each object must have these four string fields and no others:
  "original"   — copy the sentence exactly as given
  "revised"    — your minimally revised version (or the original if no change)
  "reason"     — one sentence in Korean explaining what you changed and why
  "pattern_id" — copy the pattern_id exactly as given

Return ONLY the JSON array. No markdown fences, no commentary.

Sentences:
{sentences_json}"""


def _comma_count(s: str) -> int:
    return s.count(",") + s.count("、")


def select_sentences(
    sentences: list[str],
    report: ScoreReport,
    max_count: int = MAX_SUGGESTIONS,
) -> list[SentenceSelection]:
    """Select up to max_count high-signal sentences for the rewrite prompt.

    Tier 1: S1 pattern match (any S1 regex fires on the sentence).
    Tier 2: S2 / uniform-politeness pattern match that fired at corpus level.
    Tier 3: comma density >= LLM anchor, proportional to avg sentence length.
    Tier 4 (fallback): top-up to min 3 by raw comma count when tiers 1–3 are thin.
    """
    if not sentences:
        return []

    hits = find_pattern_hits(sentences)
    fired_s2_bases: set[str] = set()
    for h in hits:
        if h.severity == "S2":
            fired_s2_bases.add(h.pattern_name.split(" (")[0])

    avg_len = mean(len(s) for s in sentences)
    comma_threshold = LLM_COMMAS_PER_SENT / max(avg_len, 20.0)

    def _classify(s: str) -> tuple[str, int]:
        for name, regex in S1_PATTERNS:
            if regex.search(s):
                return name, 1
        for pat in S2_RATE_PATTERNS:
            if pat.name in fired_s2_bases and pat.regex.search(s):
                return pat.name, 2
        if "uniform_politeness_endings" in fired_s2_bases and _POLITE_ENDING.search(s):
            return "uniform_politeness_endings", 2
        if _comma_count(s) / max(len(s), 1) >= comma_threshold:
            return "high_comma_density", 3
        return "high_comma_density", 4

    def _make(s: str, pid: str, tier: int) -> SentenceSelection:
        desc, hint = PATTERN_GUIDANCE.get(pid, ("unknown pattern", "review manually"))
        return SentenceSelection(
            sentence=s, pattern_id=pid, pattern_description=desc, fix_hint=hint, tier=tier
        )

    classified = [(_classify(s), s) for s in sentences]
    selected: list[SentenceSelection] = []
    seen: set[int] = set()

    for target_tier in (1, 2, 3):
        for idx, ((pid, tier), s) in enumerate(classified):
            if len(selected) >= max_count:
                break
            if tier == target_tier and idx not in seen:
                selected.append(_make(s, pid, tier))
                seen.add(idx)

    if len(selected) < 3:
        fallback = sorted(
            [(idx, s) for idx, (_, s) in enumerate(classified) if idx not in seen],
            key=lambda x: _comma_count(x[1]),
            reverse=True,
        )
        for idx, s in fallback:
            if len(selected) >= 3:
                break
            pid, _ = classified[idx][0]
            selected.append(_make(s, pid, 4))
            seen.add(idx)

    return selected[:max_count]


def build_prompt(selections: list[SentenceSelection]) -> str:
    payload = [
        {
            "sentence": s.sentence,
            "pattern_id": s.pattern_id,
            "pattern_description": s.pattern_description,
            "fix_hint": s.fix_hint,
        }
        for s in selections
    ]
    return _PROMPT_TEMPLATE.format(
        sentences_json=json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _call_gemini_raw(prompt: str, api_key: str, timeout: int = 30) -> dict:
    model = os.environ.get("GEMINI_MODEL", _GEMINI_DEFAULT_MODEL)
    endpoint = f"{_GEMINI_BASE}/{model}:generateContent"
    resp = requests.post(
        endpoint,
        params={"key": api_key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_suggestions(raw_text: str) -> list[RewriteSuggestion]:
    data = json.loads(raw_text)
    if not isinstance(data, list):
        raise ValueError("expected a JSON array")
    return [RewriteSuggestion.model_validate(item) for item in data]


def _token_counts(response: dict) -> tuple[int, int]:
    meta = response.get("usageMetadata", {})
    return meta.get("promptTokenCount", 0), meta.get("candidatesTokenCount", 0)


def suggest_rewrites(
    sentences: list[str],
    report: ScoreReport,
    api_key: str | None = None,
    force: bool = False,
) -> RewriteResult:
    """Run the rewrite stage. Returns a skipped result if below threshold or on error."""
    threshold = float(os.environ.get("REWRITE_THRESHOLD", str(DEFAULT_REWRITE_THRESHOLD)))
    if not force and report.score < threshold:
        return RewriteResult(
            skipped=True,
            skip_reason=f"score {report.score:.0f} below threshold {threshold:.0f}",
        )

    resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not resolved_key:
        return RewriteResult(
            skipped=True,
            skip_reason="GEMINI_API_KEY not set — skipping rewrite stage",
        )

    selections = select_sentences(sentences, report)
    if not selections:
        return RewriteResult(skipped=True, skip_reason="no sentences selected")

    prompt = build_prompt(selections)
    retry_prompt = f"Return ONLY a valid JSON array with no other text.\n\n{prompt}"
    tokens_in = tokens_out = 0

    for attempt, current_prompt in enumerate([prompt, retry_prompt]):
        try:
            raw_response = _call_gemini_raw(current_prompt, resolved_key)
            tokens_in, tokens_out = _token_counts(raw_response)
            raw_text = raw_response["candidates"][0]["content"]["parts"][0]["text"]
            suggestions = _parse_suggestions(raw_text)
            suggestions = [s for s in suggestions if s.revised != s.original]
            print(
                f"rewrite: {len(suggestions)} suggestion(s) | "
                f"tokens in={tokens_in} out={tokens_out}"
            )
            return RewriteResult(
                suggestions=suggestions[:MAX_SUGGESTIONS],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        except (json.JSONDecodeError, ValidationError, ValueError, KeyError) as exc:
            if attempt == 1:
                print(f"rewrite: unparseable JSON after retry — {exc}")
                return RewriteResult(
                    skipped=True,
                    skip_reason="Gemini returned unparseable JSON after retry",
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )
        except Exception as exc:
            msg = _redact(str(exc), resolved_key)
            print(f"rewrite: API error — {msg}")
            return RewriteResult(skipped=True, skip_reason=f"API error: {msg}")

    return RewriteResult(skipped=True, skip_reason="unexpected retry exit")
