from __future__ import annotations

from main import load_dotenv


def test_loads_values_and_strips_quotes(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nSLACK_WEBHOOK_URL="https://hooks.example/x"\nOTHER=plain\n\n')
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["SLACK_WEBHOOK_URL"] == "https://hooks.example/x"
    assert os.environ["OTHER"] == "plain"
    monkeypatch.delenv("SLACK_WEBHOOK_URL")
    monkeypatch.delenv("OTHER")


def test_real_environment_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("SLACK_WEBHOOK_URL=from-file\n")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "from-env")
    load_dotenv(env)
    import os

    assert os.environ["SLACK_WEBHOOK_URL"] == "from-env"


def test_missing_file_is_noop(tmp_path):
    load_dotenv(tmp_path / "nope.env")
