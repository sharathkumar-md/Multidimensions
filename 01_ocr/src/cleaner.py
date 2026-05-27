from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Sequence

from loguru import logger

from config.settings import settings
from src.models import PageResult

# Common OCR misreads in engineering/technical PDFs
_OCR_SUBSTITUTIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?<!\w)O(?=\.\d)"), "0"),
    (re.compile(r"(?<=\d)l(?=\d)"), "1"),
    (re.compile(r"\brn\b"), "m"),
    (re.compile(r"(?<=\w)\|(?=\w)"), "l"),
    (re.compile(r"[–—]"), "-"),
    (re.compile(r"ﬁ"), "fi"),
    (re.compile(r"ﬂ"), "fl"),
    (re.compile(r"ﬀ"), "ff"),
    (re.compile(r"ﬃ"), "ffi"),
    (re.compile(r"ﬄ"), "ffl"),
    (re.compile("[‘’ʼ]"), "'"),
    (re.compile("[“”„]"), '"'),
]

_HEADER_FOOTER_RE = re.compile(
    r"^\s*(?:page\s*\d+|©.*|\d{4}\s+\w.*|all rights reserved.*|confidential.*)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _correct_ocr_artifacts(text: str) -> str:
    for pattern, replacement in _OCR_SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    return text


def _normalise_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    return text


def _normalise_whitespace(text: str) -> str:
    lines = text.split("\n")
    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        stripped = line.rstrip()
        if stripped == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(stripped)
    return "\n".join(cleaned).strip()


def _detect_repeated_elements(pages: Sequence[PageResult], threshold: int) -> set[str]:
    line_page_count: Counter[str] = Counter()
    for page in pages:
        seen_on_this_page: set[str] = set()
        for line in page.raw_text.split("\n"):
            norm = line.strip().lower()
            if len(norm) > 3 and norm not in seen_on_this_page:
                line_page_count[norm] += 1
                seen_on_this_page.add(norm)

    repeated = {line for line, count in line_page_count.items() if count >= threshold}
    if repeated:
        logger.debug(
            "Detected {n} repeated elements (threshold={t}): {samples}",
            n=len(repeated),
            t=threshold,
            samples=list(repeated)[:5],
        )
    return repeated


def _strip_repeated_elements(text: str, repeated: set[str]) -> str:
    lines = text.split("\n")
    filtered = [line for line in lines if line.strip().lower() not in repeated]
    return "\n".join(filtered)


def _strip_regex_headers_footers(text: str) -> str:
    return _HEADER_FOOTER_RE.sub("", text)


_presidio_analyzer = None
_presidio_anonymizer = None


def _get_presidio():
    global _presidio_analyzer, _presidio_anonymizer
    if _presidio_analyzer is None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            logger.info("Initialising Presidio PII engines")
            _presidio_analyzer = AnalyzerEngine()
            _presidio_anonymizer = AnonymizerEngine()
            logger.info("Presidio initialised successfully")
        except ImportError:
            logger.warning(
                "presidio-analyzer / presidio-anonymizer not installed. "
                "PII redaction will be skipped."
            )
    return _presidio_analyzer, _presidio_anonymizer


def _redact_pii(text: str) -> str:
    analyzer, anonymizer = _get_presidio()
    if analyzer is None or anonymizer is None:
        return text

    try:
        results = analyzer.analyze(
            text=text,
            entities=settings.pii_redaction_entities,
            language="en",
        )
        if results:
            logger.debug("Found {n} PII entities, redacting", n=len(results))
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
            return anonymized.text
    except Exception as exc:
        logger.warning("PII redaction failed, returning unredacted text: {err}", err=exc)

    return text


def clean_page_text(text: str) -> str:
    text = _normalise_unicode(text)
    text = _correct_ocr_artifacts(text)
    text = _strip_regex_headers_footers(text)
    text = _normalise_whitespace(text)
    return text


def clean_document(pages: list[PageResult]) -> list[PageResult]:
    logger.info("Starting document cleaning: {n} pages", n=len(pages))

    for page in pages:
        page.raw_text = clean_page_text(page.raw_text)

    if settings.strip_repeated_elements and len(pages) >= settings.repeated_element_threshold:
        repeated = _detect_repeated_elements(pages, settings.repeated_element_threshold)
        if repeated:
            for page in pages:
                page.raw_text = _strip_repeated_elements(page.raw_text, repeated)
                page.raw_text = _normalise_whitespace(page.raw_text)
                page.word_count = len(page.raw_text.split())
            logger.info("Stripped {n} repeated header/footer patterns", n=len(repeated))

    if settings.pii_redaction_enabled:
        for page in pages:
            original_len = len(page.raw_text)
            page.raw_text = _redact_pii(page.raw_text)
            if len(page.raw_text) != original_len:
                logger.debug("Page {n}: PII redaction changed text length", n=page.page_num)
            page.word_count = len(page.raw_text.split())

    total_words = sum(p.word_count for p in pages)
    logger.info("Document cleaning complete: {words} words across {n} pages", words=total_words, n=len(pages))
    return pages
