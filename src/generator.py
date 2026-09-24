"""Answer generation with the Gemini API.

The generator is deliberately small: it receives a question plus a labelled,
numbered context block and produces an answer. Everything that decides *what*
context is relevant lives in the retriever/pipeline, not here.

All network access goes through a single `generate()` method so tests can
mock it (or use a stub generator) without a real API key.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("rag.generator")

SYSTEM_PROMPT = """You are a precise document Q&A assistant. You answer questions using ONLY the retrieved context provided below.

Rules:
1. Base your answer EXCLUSIVELY on the provided context. Never use outside knowledge.
2. Never invent facts, numbers, quotes, or page numbers.
3. Cite your sources inline using [n], where n matches the numbered context entries (e.g. "...the system uses X because ... [1]").
4. If the context does not support an answer, reply exactly: "I don't know based on the provided documents."
5. If you use more than one source, clearly attribute claims to each source, e.g. (research_paper_1.pdf, page 4) and (research_paper_2.pdf, page 7).
6. End your answer with a "Sources:" list naming every source you actually used, using the format:
   Sources:
   - <document>, page <page>
7. Only pages that appear in the provided context may be cited. Do not guess page numbers."""

DEFAULT_ANSWER_TEMPLATE = "I don't know based on the provided documents."


class GeminiError(Exception):
    """Raised when generation fails after retries (network, auth, quota)."""


class GeminiNotConfigured(Exception):
    """Raised when no API key is available."""


@dataclass
class GenerationResult:
    """Output of a single LLM call plus latency / token / cost bookkeeping."""

    text: str
    model: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None  # None => cost not calculated (no pricing configured)
    note: str = ""


@dataclass
class GeminiGenerator:
    """Minimal wrapper around google-generativeai."""

    api_key: str = ""
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.2
    max_retries: int = 3
    input_cost_per_1m: float = 0.0
    output_cost_per_1m: float = 0.0
    _client: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        if not self.model_name:
            self.model_name = "gemini-1.5-flash"

    def is_configured(self) -> bool:
        """True when an API key is present (does not test connectivity)."""
        return bool(self.api_key and self.api_key.strip() and self.api_key != "your_api_key_here")

    def _get_client(self):
        if self._client is not None:
            return self._client
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(self.model_name, system_instruction=SYSTEM_PROMPT)
        return self._client

    def generate(self, question: str, context: str, temperature: float | None = None) -> GenerationResult:
        """Generate an answer for `question` given the labelled `context`.

        Raises:
            GeminiNotConfigured: no API key configured.
            GeminiError: permanent failure after retries.
        """
        if not self.is_configured():
            raise GeminiNotConfigured(
                "GEMINI_API_KEY is not set. Copy .env.example to .env, add your key, and retry."
            )
        if not question.strip():
            raise ValueError("Question is empty.")

        temp = self.temperature if temperature is None else temperature
        prompt = (
            f"Question: {question.strip()}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer the question using ONLY the context above and follow the system rules."
        )

        model = self._get_client()
        start = time.perf_counter()
        for attempt in range(1, self.max_retries + 1):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": temp, "max_output_tokens": 1024},
                )
                elapsed = time.perf_counter() - start
                text = _extract_text(response)
                return GenerationResult(
                    text=text,
                    model=self.model_name,
                    latency_s=elapsed,
                    prompt_tokens=_tokens(response, "prompt"),
                    completion_tokens=_tokens(response, "completion"),
                    cost_usd=self._compute_cost(response),
                    note=self._cost_note(),
                )
            except Exception as exc:  # noqa: BLE001 - surface all provider errors
                if attempt < self.max_retries and _is_retryable(exc):
                    delay = 2.0 ** attempt
                    logger.warning(
                        "Gemini call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                        attempt,
                        self.max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                elapsed = time.perf_counter() - start
                logger.error("Gemini call failed permanently after %d attempts: %s", attempt, exc)
                raise GeminiError(f"Gemini API error after {attempt} attempt(s): {exc}") from exc

        # Unreachable - kept for type checkers.
        raise GeminiError("Gemini generation failed for an unknown reason.")

    def _compute_cost(self, response: Any) -> float | None:
        prompt_tokens = _tokens(response, "prompt")
        completion_tokens = _tokens(response, "completion")
        if prompt_tokens is None or completion_tokens is None:
            return None
        if self.input_cost_per_1m <= 0 or self.output_cost_per_1m <= 0:
            return None
        return (prompt_tokens * self.input_cost_per_1m + completion_tokens * self.output_cost_per_1m) / 1_000_000

    def _cost_note(self) -> str:
        if self.input_cost_per_1m <= 0 or self.output_cost_per_1m <= 0:
            return (
                "Cost is not calculated because no pricing is configured "
                "(set GEMINI_INPUT_COST_PER_1M / GEMINI_OUTPUT_COST_PER_1M in .env). "
                "Token usage is recorded instead."
            )
        return "Cost estimate uses the rates configured in .env."


def _extract_text(response: Any) -> str:
    try:
        return response.text.strip()
    except Exception:  # ValueErrors for blocked prompts, etc.
        return DEFAULT_ANSWER_TEMPLATE


def _tokens(response: Any, kind: str) -> int | None:
    """Best-effort token extraction from usage_metadata (shape varies by SDK)."""
    try:
        usage = response.usage_metadata
        if usage is None:
            return None
        if kind == "prompt":
            value = getattr(usage, "prompt_token_count", None)
        else:
            value = getattr(usage, "candidates_token_count", None)
        return int(value) if value is not None else None
    except Exception:
        return None


def _is_retryable(exc: Exception) -> bool:
    """True for quota/transient errors that may succeed on retry."""
    text = str(exc).lower()
    markers = ("resource_exhausted", "429", "503", "quota", "rate limit", "service unavailable", "server error")
    return any(marker in text for marker in markers)