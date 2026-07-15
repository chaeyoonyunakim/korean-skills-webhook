from __future__ import annotations

from pathlib import Path

import pytest

from src.models import PostText
from src.segment import segment_korean

FIXTURES = Path(__file__).parent / "fixtures"


def _sentences(name: str) -> list[str]:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    seg = segment_korean(PostText(url="fixture", title=name, text=text))
    return seg.sentences


@pytest.fixture(scope="session")
def human_sentences() -> list[str]:
    return _sentences("human_like.txt")


@pytest.fixture(scope="session")
def ai_sentences() -> list[str]:
    return _sentences("ai_like.txt")


@pytest.fixture(scope="session")
def ai_toned_sentences() -> list[str]:
    return _sentences("ai_toned_ko.txt")


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES
