"""Tests for src/cleaner.py"""
from __future__ import annotations

import pytest

from src.cleaner import (
    _correct_ocr_artifacts,
    _detect_repeated_elements,
    _normalise_whitespace,
    _strip_repeated_elements,
    clean_document,
    clean_page_text,
)
from src.models import PageLayout, PageResult


def make_page(page_num: int, text: str) -> PageResult:
    return PageResult(page_num=page_num, layout=PageLayout.DIGITAL, raw_text=text, word_count=len(text.split()))


class TestOCRArtifacts:
    def test_ligature_fi(self):
        assert _correct_ocr_artifacts("ﬁle") == "file"

    def test_ligature_fl(self):
        assert _correct_ocr_artifacts("ﬂow") == "flow"

    def test_curly_quotes(self):
        curly = "‘hello’"
        assert _correct_ocr_artifacts(curly) == "'hello'"

    def test_em_dash_to_hyphen(self):
        assert _correct_ocr_artifacts("10–mm") == "10-mm"

    def test_zero_before_decimal(self):
        assert _correct_ocr_artifacts("O.5 mm") == "0.5 mm"

    def test_no_change_on_clean_text(self):
        text = "The bearing has a load rating of 12.5 kN."
        assert _correct_ocr_artifacts(text) == text


class TestWhitespaceNormalisation:
    def test_collapses_excess_blank_lines(self):
        result = _normalise_whitespace("line1\n\n\n\n\nline2")
        assert "\n\n\n\n" not in result  # no runs of 3+ blank lines

    def test_strips_trailing_spaces(self):
        result = _normalise_whitespace("line1   \nline2   ")
        for line in result.split("\n"):
            assert line == line.rstrip()

    def test_preserves_two_blank_lines(self):
        result = _normalise_whitespace("para1\n\npara2")
        assert "para1" in result
        assert "para2" in result


class TestRepeatedElements:
    def test_detects_common_footer(self):
        footer = "GGB Bearing Technology"
        pages = [make_page(i, f"Content {i}\n{footer}") for i in range(1, 5)]
        repeated = _detect_repeated_elements(pages, threshold=3)
        assert footer.lower() in repeated

    def test_does_not_flag_unique_content(self):
        pages = [make_page(i, f"Unique content on page {i}") for i in range(1, 5)]
        assert len(_detect_repeated_elements(pages, threshold=3)) == 0

    def test_strip_removes_flagged_lines(self):
        result = _strip_repeated_elements("Normal content\nPage Footer Text\nMore content", {"page footer text"})
        assert "Page Footer Text" not in result
        assert "Normal content" in result


class TestCleanPageText:
    def test_strips_copyright_line(self):
        result = clean_page_text("Bearing specs\n© 2023 GGB Inc. All rights reserved\nMore content")
        assert "All rights reserved" not in result

    def test_strips_page_number_line(self):
        result = clean_page_text("Section heading\nPage 3\nActual content")
        assert "Page 3" not in result

    def test_preserves_technical_content(self):
        text = "Load rating: 12.5 kN\nOperating temperature: -40°C to +180°C"
        result = clean_page_text(text)
        assert "12.5 kN" in result
        assert "-40°C to +180°C" in result


class TestCleanDocument:
    def test_updates_word_count(self):
        pages = [make_page(1, "hello world foo bar")]
        result = clean_document(pages)
        assert result[0].word_count >= 0

    def test_returns_same_page_count(self):
        pages = [make_page(i, f"Text on page {i}") for i in range(1, 6)]
        assert len(clean_document(pages)) == 5

    def test_repeated_element_removed_across_pages(self, monkeypatch):
        import config.settings as cfg_module
        monkeypatch.setattr(cfg_module.settings, "strip_repeated_elements", True)
        monkeypatch.setattr(cfg_module.settings, "repeated_element_threshold", 3)
        monkeypatch.setattr(cfg_module.settings, "pii_redaction_enabled", False)

        footer = "confidential footer"
        pages = [make_page(i, f"Real content {i}\n{footer}") for i in range(1, 6)]
        result = clean_document(pages)
        for page in result:
            assert footer not in page.raw_text.lower()
