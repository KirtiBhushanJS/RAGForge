"""Tests for citation formatting and citation veracity checks."""

from __future__ import annotations

from src.citations import (
    build_source_map,
    citation_line,
    format_citation_list,
    format_context,
    verify_citations,
)
from src.retriever import RetrievedChunk
from src.utils import parse_citation_numbers


def chunk(document: str, page: int, text: str = "some chunk text", score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{document}_p{page}",
        document=document,
        document_path=f"/docs/{document}",
        page=page,
        text=text,
        score=score,
    )


def test_source_map_and_lines() -> None:
    chunks = [chunk("a.pdf", 4), chunk("b.pdf", 7), chunk("a.pdf", 4)]
    refs = build_source_map(chunks)
    assert list(refs) == [1, 2]          # duplicate (a.pdf, 4) collapsed
    assert refs[1].document == "a.pdf"
    assert refs[1].page == 4
    assert refs[2].document == "b.pdf"
    assert refs[2].page == 7
    assert citation_line(refs[1]) == "[1] a.pdf — Page 4"
    assert citation_line(refs[2]) == "[2] b.pdf — Page 7"


def test_format_citation_list() -> None:
    text = format_citation_list([chunk("a.pdf", 4), chunk("b.pdf", 7)])
    assert text == "Sources:\n[1] a.pdf — Page 4\n[2] b.pdf — Page 7"


def test_format_context_labels() -> None:
    context = format_context([chunk("a.pdf", 4, "Some text here")])
    assert "[1] Source: a.pdf, page 4" in context
    assert "Some text here" in context


def test_verify_citations_ok() -> None:
    chunks = [chunk("a.pdf", 4), chunk("b.pdf", 7)]
    answer = "The fleet is large [1] and revenue is high (b.pdf, page 7)."
    check = verify_citations(answer, chunks)
    assert check.ok is True
    assert check.cited_numbers == [1]


def test_verify_citations_catches_invented_number() -> None:
    chunks = [chunk("a.pdf", 4)]
    check = verify_citations("Unknown claim [9].", chunks)
    assert check.ok is False
    assert check.missing_numbers == [9]


def test_parse_citation_numbers() -> None:
    assert parse_citation_numbers("Answer [1] and [12] and plain 3.") == [1, 12]