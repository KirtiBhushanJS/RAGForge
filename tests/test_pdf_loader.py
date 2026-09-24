"""Tests for PDF text extraction: page preservation, empty pages, malformed files."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.pdf_loader import (
    PdfLoadError,
    extract_pages,
    flatten_pages,
    get_pdf_files,
    load_documents,
)


def test_get_pdf_files_only_finds_pdfs(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"%PDF-fake")
    (tmp_path / "notes.txt").write_text("not a pdf")
    files = get_pdf_files(tmp_path)
    assert [f.name for f in files] == ["a.pdf"]


def test_extract_pages_preserves_page_numbers(tmp_path: Path, make_pdf) -> None:
    path = make_pdf(tmp_path / "two_pages.pdf", ["First page content here.", "Second page content here."])
    pages = extract_pages(path)
    assert len(pages) == 2
    assert pages[0].page == 1
    assert pages[1].page == 2
    assert "First page" in pages[0].text
    assert "Second page" in pages[1].text
    assert pages[0].document == "two_pages.pdf"
    assert pages[0].num_pages == 2


def test_empty_page_is_marked(tmp_path: Path, make_pdf) -> None:
    path = make_pdf(tmp_path / "blank_page.pdf", ["Only first page has text.", ""])
    pages = extract_pages(path)
    assert len(pages) == 2
    assert pages[0].is_empty is False
    assert pages[1].is_empty is True


def test_malformed_pdf_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is definitely not a pdf file")
    with pytest.raises(PdfLoadError):
        extract_pages(path)


def test_load_documents_skips_malformed_files(tmp_path: Path, make_pdf, caplog) -> None:
    good = make_pdf(tmp_path / "good.pdf", ["Hello world."])
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"garbage not a pdf")

    with caplog.at_level(logging.ERROR, logger="rag.pdf_loader"):
        docs, report = load_documents(tmp_path)

    assert report.files_attempted == 2
    assert report.files_loaded == 1
    assert report.files_failed == 1
    assert report.failed_files == ["bad.pdf"]
    assert [d.document for d in docs] == ["good.pdf"]
    assert report.pages_extracted == 1


def test_load_documents_empty_folder(tmp_path: Path, caplog) -> None:
    docs, report = load_documents(tmp_path)
    assert docs == []
    assert report.files_attempted == 0
    assert len(caplog.records) >= 0


def test_flatten_pages_order(tmp_path: Path, make_pdf) -> None:
    make_pdf(tmp_path / "one.pdf", ["Page one."])
    make_pdf(tmp_path / "two.pdf", ["Page one.", "Page two."])
    docs, _ = load_documents(tmp_path)
    pages = flatten_pages(docs)
    titles = [(p.document, p.page) for p in pages]
    assert titles == [("one.pdf", 1), ("two.pdf", 1), ("two.pdf", 2)]