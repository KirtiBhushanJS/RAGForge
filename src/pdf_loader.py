"""PDF loading utilities built on PyMuPDF (fitz).

Responsibilities:
- discover PDF files in `data/documents/`
- extract page-level text while preserving the page number
- tolerate and report empty pages and malformed PDFs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF: `fitz` alias is deprecated on the new API

logger = logging.getLogger("rag.pdf_loader")

MIN_PAGE_CHARS = 5  # pages with fewer chars are treated as empty / scanned only


class PdfLoadError(Exception):
    """Raised when a PDF cannot be read at all."""


@dataclass
class DocumentPage:
    """One extracted page.

    `page` is 1-based page number as printed on the document.
    """

    document: str            # file name, e.g. "manual.pdf"
    document_path: str       # absolute path to the PDF
    page: int                # 1-based page number
    text: str                # extracted text for this page
    num_pages: int           # total pages in the source document
    is_empty: bool = False


@dataclass
class LoadedDocument:
    """A successfully opened PDF with its extracted pages."""

    document: str
    path: Path
    num_pages: int
    pages: list[DocumentPage]


@dataclass
class IngestionReport:
    """Summary of an ingestion run (used by scripts/ingest.py)."""

    files_attempted: int = 0
    files_loaded: int = 0
    files_failed: int = 0
    pages_extracted: int = 0
    empty_pages_skipped: int = 0
    failed_files: list[str] | None = None

    def __post_init__(self) -> None:
        self.failed_files = [] if self.failed_files is None else self.failed_files


def get_pdf_files(documents_dir: Path) -> list[Path]:
    """Return a sorted list of `.pdf` files directly inside `documents_dir`."""
    if not documents_dir.exists():
        logger.warning("Documents directory does not exist: %s", documents_dir)
        return []
    return sorted(p for p in documents_dir.glob("*.pdf") if p.is_file())


def _is_empty_page(text: str) -> bool:
    return len(text.strip()) <= MIN_PAGE_CHARS


def extract_pages(path: Path) -> list[DocumentPage]:
    """Extract page-level text from a single PDF using PyMuPDF.

    Raises:
        PdfLoadError: if the file cannot be opened / has no pages.
    """
    try:
        doc = fitz.open(path)
    except Exception as exc:  # fitz raises various exceptions for bad files
        raise PdfLoadError(f"Could not open PDF '{path.name}': {exc}") from exc

    if doc.page_count == 0:
        doc.close()
        raise PdfLoadError(f"PDF '{path.name}' has no pages.")

    pages: list[DocumentPage] = []
    for index in range(doc.page_count):
        raw = doc[index].get_text("text")
        text = raw.strip()
        pages.append(
            DocumentPage(
                document=path.name,
                document_path=str(path.resolve()),
                page=index + 1,
                text=text,
                num_pages=doc.page_count,
                is_empty=_is_empty_page(text),
            )
        )
    doc.close()
    return pages


def load_documents(documents_dir: Path) -> tuple[list[LoadedDocument], IngestionReport]:
    """Load every PDF in the documents directory.

    Returns a tuple of (loaded_documents, report). Failed/malformed PDFs are
    logged and skipped without aborting the whole run.

    If two PDFs share the same file name (in nested folders or after copying),
    the report emits a warning and every chunk keeps its full relative path so
    citations remain unambiguous.
    """
    report = IngestionReport()
    loaded: list[LoadedDocument] = []
    seen_names: dict[str, Path] = {}

    for path in get_pdf_files(documents_dir):
        report.files_attempted += 1
        try:
            pages = extract_pages(path)
        except PdfLoadError as exc:
            logger.error("%s", exc)
            report.files_failed += 1
            report.failed_files.append(path.name)
            continue

        doc = LoadedDocument(document=path.name, path=path, num_pages=len(pages) or 1, pages=pages)
        report.pages_extracted += len(pages)
        report.empty_pages_skipped += sum(1 for p in pages if p.is_empty)
        loaded.append(doc)

        lowered = path.name.lower()
        if lowered in seen_names:
            logger.warning(
                "Duplicate document basename '%s' (also found at %s). "
                "Relative paths are kept in metadata to keep citations unique.",
                path.name,
                seen_names[lowered],
            )
        else:
            seen_names[lowered] = path

    report.files_loaded = len(loaded)
    if report.files_failed:
        logger.warning("Skipped %d malformed PDF(s): %s", report.files_failed, report.failed_files)
    if not loaded:
        logger.warning(
            "No readable PDFs found in %s. Place your *.pdf files there.",
            documents_dir,
        )
    return loaded, report


def flatten_pages(documents: list[LoadedDocument]) -> list[DocumentPage]:
    """Flatten loaded documents into a single page list (original order)."""
    pages: list[DocumentPage] = []
    for doc in documents:
        pages.extend(doc.pages)
    return pages