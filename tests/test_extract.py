from __future__ import annotations

from src.extract import extract_post_text


def test_real_post_extraction(fixtures_dir):
    post = extract_post_text(str(fixtures_dir / "real_post.html"))
    assert post.title == "Survey data analysis for free-text responses"
    # Site chrome and the numeric results table must be dropped.
    assert "footer text here" not in post.text
    assert "About" not in post.text
    assert "Lemmatization" not in post.text  # table header
    # Prose body must survive, including mixed-language spans.
    assert "감정 분석" in post.text
    assert "TextBlob" in post.text
    assert len(post.text) > 1000
