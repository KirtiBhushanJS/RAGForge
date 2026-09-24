"""Shared helpers used across the project (logging, timing, JSON, text fur helpers)."""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

# The exact string the system uses whenever it cannot answer from context.
UNKNOWN_ANSWER = "I don't know based on the provided documents."


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a simple, consistent root logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@contextmanager
def timeit(label: str = "") -> Iterator[float]:
    """Time a block of code; yields the elapsed seconds when finished."""
    start = time.perf_counter()
    yield 0.0
    elapsed = time.perf_counter() - start
    if label:
        logging.getLogger("rag.utils").debug("%s took %.3fs", label, elapsed)


def ensure_dir(path: Path) -> Path:
    """Create the directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default: Any = None) -> Any:
    """Read a JSON file; return `default` (and log) on any failure."""
    logger = logging.getLogger("rag.utils")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        logger.warning("JSON file not found: %s", path)
        return default
    except json.JSONDecodeError:
        logger.error("Corrupted JSON in: %s", path)
        return default


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON file atomically (temp file + rename)."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def normalize_document_name(name: str) -> str:
    """Fold a document name/path into a comparable key (case + path insensitive)."""
    return Path(name).name.strip().lower()


def word_count(text: str) -> int:
    """Approximate word count (whitespace split)."""
    return len(text.split())


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    """Clamp a number into [low, high]."""
    return max(low, min(high, value))


def parse_citation_numbers(text: str) -> list[int]:
    """Extract inline citation numbers like '[1]', '[3]' from generated text."""
    return [int(n) for n in re.findall(r"\[(\d{1,3})\]", text)]


def safe_get(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """dict.get wrapper that never throws for non-string keys."""
    try:
        return mapping.get(key, default)
    except (AttributeError, TypeError):
        return default


def format_number(value: float, digits: int = 4) -> str:
    """Format a float for the CLI / CSV output."""
    return f"{value:.{digits}f}"


def sum_time(entries: Sequence[float]) -> float:
    return float(sum(entries))