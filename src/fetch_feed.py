"""Stage 1: discover posts from a Jekyll (jekyll-feed) Atom feed.

Uses stdlib XML parsing — jekyll-feed output is plain Atom, so feedparser
would be a needless dependency.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime

import requests

from .models import FeedEntry

ATOM_NS = "{http://www.w3.org/2005/Atom}"
DEFAULT_FEED_URL = "https://chaeyoonyunakim.github.io/feed.xml"


def parse_feed(xml_text: str) -> list[FeedEntry]:
    """Parse Atom XML into feed entries (pure function, testable offline)."""
    root = ET.fromstring(xml_text)
    entries: list[FeedEntry] = []
    for entry in root.iter(f"{ATOM_NS}entry"):
        title_el = entry.find(f"{ATOM_NS}title")
        link_el = entry.find(f"{ATOM_NS}link")
        published_el = entry.find(f"{ATOM_NS}published")
        if link_el is None or not link_el.get("href"):
            continue
        published = None
        if published_el is not None and published_el.text:
            try:
                published = datetime.fromisoformat(
                    published_el.text.replace("Z", "+00:00")
                )
            except ValueError:
                pass
        entries.append(
            FeedEntry(
                title=(title_el.text or "").strip() if title_el is not None else "",
                url=link_el.get("href", ""),
                published=published,
            )
        )
    return entries


def fetch_feed(feed_url: str = DEFAULT_FEED_URL, timeout: int = 30) -> list[FeedEntry]:
    """Fetch and parse the Atom feed."""
    resp = requests.get(feed_url, timeout=timeout)
    resp.raise_for_status()
    return parse_feed(resp.text)
