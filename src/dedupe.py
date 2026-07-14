"""Local seen-posts store so feed scans only notify about new posts."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SEEN_FILE = "seen_posts.json"


def load_seen(path: str | Path = DEFAULT_SEEN_FILE) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError):
        return set()


def save_seen(urls: set[str], path: str | Path = DEFAULT_SEEN_FILE) -> None:
    Path(path).write_text(
        json.dumps(sorted(urls), ensure_ascii=False, indent=2), encoding="utf-8"
    )
