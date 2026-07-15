from __future__ import annotations

from src.dedupe import load_seen, save_seen
from src.fetch_feed import parse_feed

ATOM_SAMPLE = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>blog</title>
  <entry>
    <title>포스트 하나</title>
    <link href="https://example.com/2026/07/03/post-one/" rel="alternate"/>
    <published>2026-07-03T00:00:00+00:00</published>
  </entry>
  <entry>
    <title>Post two</title>
    <link href="https://example.com/2026/07/10/post-two/" rel="alternate"/>
    <published>2026-07-10T00:00:00+00:00</published>
  </entry>
</feed>
"""


def test_parse_feed():
    entries = parse_feed(ATOM_SAMPLE)
    assert len(entries) == 2
    assert entries[0].title == "포스트 하나"
    assert entries[0].url.endswith("/post-one/")
    assert entries[0].published is not None and entries[0].published.year == 2026


def test_seen_roundtrip(tmp_path):
    path = tmp_path / "seen.json"
    assert load_seen(path) == set()
    save_seen({"https://a", "https://b"}, path)
    assert load_seen(path) == {"https://a", "https://b"}


def test_seen_corrupt_file(tmp_path):
    path = tmp_path / "seen.json"
    path.write_text("not json")
    assert load_seen(path) == set()
