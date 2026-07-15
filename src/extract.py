"""Stage 2: extract the main prose text of a post from HTML.

Accepts an http(s) URL or a local file path (local paths keep tests and
offline development network-free). Extraction is prose-focused: tables,
code blocks, and site chrome are dropped because numeric tables and code
carry no Korean prose tone signal and would distort the Korean-text ratio.
"""

from __future__ import annotations

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .models import PostText

# Elements that are site chrome or non-prose content. Inline <code> is kept:
# embedded technical terms (TextBlob, NLTK, ...) are part of the prose in
# mixed-language posts; only block-level code (<pre>) is dropped.
_DROP_SELECTORS = ["script", "style", "nav", "header", "footer", "table", "pre"]

# Jekyll/minima-style content containers, most specific first.
_CONTENT_SELECTORS = ["div.post-content", "article", "main", "body"]


def _load_html(source: str, timeout: int = 30) -> str:
    if re.match(r"^https?://", source):
        resp = requests.get(source, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    return Path(source).read_text(encoding="utf-8")


def extract_post_text(source: str) -> PostText:
    """Extract title and main prose text from a post URL or local HTML file."""
    html = _load_html(source)
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    title_el = soup.select_one("h1.post-title") or soup.find("h1")
    if title_el is not None:
        title = title_el.get_text(strip=True)
    if not title:
        og = soup.find("meta", property="og:title")
        if og is not None and og.get("content"):
            title = str(og["content"]).strip()
    if not title and soup.title is not None:
        title = soup.title.get_text(strip=True)

    node = None
    for selector in _CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node is not None:
            break
    if node is None:
        node = soup

    for selector in _DROP_SELECTORS:
        for el in node.select(selector):
            el.decompose()

    text = node.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()

    return PostText(url=source, title=title, text=text)
