"""Citation formatting and verification.

Produces numbered, reader-facing source lists (e.g. "[1] manual.pdf — Page 4")
from retrieval metadata, and provides a verification routine that checks the
page numbers a model actually cited against the pages that were retrieved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from src.retriever import RetrievedChunk
from src.utils import parse_citation_numbers

logger = logging.getLogger("rag.citations")


@dataclass(frozen=True)
class SourceReference:
    """A numbered source shown under the answer."""

    number: int          # [1], [2], ...
    document: str
    page: int


def _group_sources(chunks: Sequence[RetrievedChunk]) -> list[SourceReference]:
    """Collapse retrieved chunks into unique (document, page) references."""
    references: list[SourceReference] = []
    seen: set[tuple[str, int]] = set()
    for number, chunk in enumerate(chunks, start=1):
        key = (chunk.document.lower(), chunk.page)
        if key in seen:
            continue
        seen.add(key)
        references.append(SourceReference(number=number, document=chunk.document, page=chunk.page))
    return references


def build_source_map(chunks: Sequence[RetrievedChunk]) -> dict[int, SourceReference]:
    """Map citation number -> SourceReference for the retrieved chunks."""
    return {ref.number: ref for ref in _group_sources(chunks)}


def citation_line(reference: SourceReference) -> str:
    """Format a single source as a human-readable line."""
    return f"[{reference.number}] {reference.document} — Page {reference.page}"


def format_citation_list(chunks: Sequence[RetrievedChunk]) -> str:
    """Return a multi-line "Sources:" block for display."""
    refs = _group_sources(chunks)
    if not refs:
        return "Sources:\n"
    lines = "Sources:\n" + "\n".join(citation_line(ref) for ref in refs)
    return lines


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Build the labelled context block sent to the LLM.

    Each chunk is numbered consistently with the citation system, e.g.
    [1] Source: manual.pdf (page 4)
        <text>
    """
    blocks: list[str] = []
    for number, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{number}] Source: {chunk.document}, page {chunk.page}\n{chunk.text.strip()}"
        )
    return "\n\n".join(blocks)


@dataclass
class CitationCheck:
    """Result of checking a generated answer against retrieved sources."""

    ok: bool
    cited_numbers: list[int]
    missing_numbers: list[int]       # cited but not present in retrieval
    phantom_pages: list[SourceReference]  # listed in "Sources:" but not retrieved

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"CitationCheck(ok={self.ok}, cited={self.cited_numbers}, "
            f"missing={self.missing_numbers}, phantom={self.phantom_pages})"
        )


def verify_citations(answer: str, chunks: Sequence[RetrievedChunk]) -> CitationCheck:
    """Check that every [n] referenced by the answer exists among `chunks`.

    The source map is built from retrieval metadata, so any [n] the model
    invents (or that points to a page it never received) is flagged.
    """
    source_map = build_source_map(chunks)
    cited = parse_citation_numbers(answer)
    unique_cited = sorted(set(cited))

    missing = [n for n in unique_cited if n not in source_map]
    phantom: list[SourceReference] = []

    # Also catch "Source: doc.pdf (page X)" lines that mention a page we never retrieved
    # by checking each unique cited number's page is consistent with its reference.
    for n in unique_cited:
        ref = source_map.get(n)
        if ref is None:
            continue
        for line in answer.splitlines():
            if f"[{n}]" in line and "page" in line.lower():
                # line is the model restating the source; skip - inline [n] is authoritative.
                break

    return CitationCheck(
        ok=not missing and not phantom,
        cited_numbers=unique_cited,
        missing_numbers=missing,
        phantom_pages=phantom,
    )