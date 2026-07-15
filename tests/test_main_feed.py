from __future__ import annotations

import json

import main as cli
from src.models import FeedEntry


def _entries() -> list[FeedEntry]:
    return [
        FeedEntry(title="one", url="https://example.com/one/"),
        FeedEntry(title="two", url="https://example.com/two/"),
    ]


def test_first_run_seeds_without_notifying(tmp_path, monkeypatch, capsys):
    seen_file = tmp_path / "seen.json"
    monkeypatch.setattr(cli, "fetch_feed", lambda url: _entries())
    monkeypatch.setattr(
        cli, "process_post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not notify"))
    )
    cli.main(["--feed", "--seen-file", str(seen_file)])
    assert "first run" in capsys.readouterr().out
    assert set(json.loads(seen_file.read_text())) == {e.url for e in _entries()}


def test_second_run_processes_only_new(tmp_path, monkeypatch):
    seen_file = tmp_path / "seen.json"
    seen_file.write_text(json.dumps(["https://example.com/one/"]))
    processed: list[str] = []
    monkeypatch.setattr(cli, "fetch_feed", lambda url: _entries())
    monkeypatch.setattr(cli, "process_post", lambda src, dry, force=False: processed.append(src))
    cli.main(["--feed", "--seen-file", str(seen_file)])
    assert processed == ["https://example.com/two/"]
    assert set(json.loads(seen_file.read_text())) == {e.url for e in _entries()}


def test_dry_run_does_not_write_state(tmp_path, monkeypatch):
    seen_file = tmp_path / "seen.json"
    monkeypatch.setattr(cli, "fetch_feed", lambda url: _entries())
    cli.main(["--feed", "--seen-file", str(seen_file), "--dry-run"])
    assert not seen_file.exists()
