"""End-to-end RAG pipeline: retrieve -> confidence gate -> generate -> cite.

The pipeline is the single entry point used by both the Streamlit app and the
evaluation/experiment scripts, so the app and the evaluation always measure the
exact same code path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.citations import SourceReference, build_source_map, format_context
from src.config import Settings, get_settings
from src.embeddings import EmbeddingModel, get_embedding_model
from src.generator import GeminiError, GeminiGenerator, GeminiNotConfigured
from src.retriever import RetrievedChunk, Retriever
from src.utils import UNKNOWN_ANSWER
from src.vector_store import IndexNotReadyError, load_index

logger = logging.getLogger("rag.pipeline")


@dataclass
class RAGResult:
    """Everything produced for one question (answer + sources + latencies)."""

    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    sources: list[SourceReference]
    max_similarity: float
    is_unknown: bool
    retrieval_latency_s: float = 0.0
    generation_latency_s: float = 0.0
    total_latency_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    note: str = ""          # human-readable warnings/messages
    generator_used: bool = False


class RAGPipeline:
    """Loads the persistent index lazily and answers questions with citations."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: EmbeddingModel | None = None,
        retriever: Retriever | None = None,
        generator: GeminiGenerator | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedding_model(self.settings.embedding_model)
        self._retriever = retriever
        self._generator = generator or GeminiGenerator(
            api_key=self.settings.gemini_api_key,
            model_name=self.settings.gemini_model,
            temperature=self.settings.gemini_temperature,
            max_retries=self.settings.gemini_max_retries,
            input_cost_per_1m=self.settings.gemini_input_cost_per_1m,
            output_cost_per_1m=self.settings.gemini_output_cost_per_1m,
        )
        self._index_payload: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    def index_ready(self) -> bool:
        if self._retriever is not None:
            return True
        try:
            self._load_index()
            return True
        except IndexNotReadyError:
            return False

    def index_summary(self) -> dict[str, Any]:
        """Sidebar summary (documents, chunks, model). Never raises."""
        try:
            self._load_index()
        except IndexNotReadyError:
            return {"ready": False}
        payload = self._index_payload or {}
        chunks = payload.get("chunks", [])
        docs = sorted({str(c["document"]) for c in chunks})
        return {
            "ready": True,
            "model": payload.get("model"),
            "dim": payload.get("dim"),
            "num_chunks": len(chunks),
            "num_documents": len(docs),
            "documents": docs,
            "created_at": payload.get("created_at"),
        }

    def generator_configured(self) -> bool:
        return self._generator.is_configured()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_index(self) -> Retriever:
        if self._retriever is not None:
            return self._retriever
        index, payload = load_index(
            self.settings.index_path,
            self.settings.metadata_path,
            settings=self.settings,
        )
        self._index_payload = payload
        self._retriever = Retriever(index, payload, self.embedder, settings=self.settings)
        logger.info("Loaded index: %d chunks across %d documents.",
                    self._retriever.num_chunks, len(self._retriever.documents))
        return self._retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> RAGResult:
        """Answer `question` with citations, or say "I don't know" when below threshold."""
        threshold = self.settings.similarity_threshold if similarity_threshold is None else similarity_threshold
        started = time.perf_counter()

        if not question or not question.strip():
            return RAGResult(
                question=question or "",
                answer="Please enter a question.",
                retrieved_chunks=[],
                sources=[],
                max_similarity=0.0,
                is_unknown=True,
                note="Empty question.",
            )

        # ---- 1. Retrieve -------------------------------------------------
        try:
            retriever = self._load_index()
        except IndexNotReadyError as exc:
            return RAGResult(
                question=question,
                answer=UNKNOWN_ANSWER,
                retrieved_chunks=[],
                sources=[],
                max_similarity=0.0,
                is_unknown=True,
                note=str(exc),
            )

        t0 = time.perf_counter()
        chunks = retriever.retrieve(question, top_k=top_k)
        retrieval_latency = time.perf_counter() - t0

        if not chunks:
            return RAGResult(
                question=question,
                answer=UNKNOWN_ANSWER,
                retrieved_chunks=[],
                sources=[],
                max_similarity=0.0,
                is_unknown=True,
                retrieval_latency_s=retrieval_latency,
                total_latency_s=time.perf_counter() - started,
                note="Retrieval returned no results for this question.",
            )

        max_sim = chunks[0].score
        sources = build_source_map(chunks)
        source_refs = list(sources.values())

        # ---- 2. Confidence gate ------------------------------------------
        if max_sim < threshold:
            return RAGResult(
                question=question,
                answer=UNKNOWN_ANSWER,
                retrieved_chunks=chunks,
                sources=source_refs,
                max_similarity=max_sim,
                is_unknown=True,
                retrieval_latency_s=retrieval_latency,
                total_latency_s=time.perf_counter() - started,
                note=(
                    f"Retrieval confidence {max_sim:.3f} is below the threshold "
                    f"{threshold:.3f}; no LLM call was made."
                ),
            )

        # ---- 3. Generate with LLM ----------------------------------------
        context = format_context(chunks)
        if not self._generator.is_configured():
            partial = RAGResult(
                question=question,
                answer=UNKNOWN_ANSWER,
                retrieved_chunks=chunks,
                sources=source_refs,
                max_similarity=max_sim,
                is_unknown=True,
                retrieval_latency_s=retrieval_latency,
                total_latency_s=time.perf_counter() - started,
                note=(
                    "Retrieval succeeded but GEMINI_API_KEY is not set, so "
                    "the answer was not generated. Sources shown below."
                ),
            )
            return partial

        t0 = time.perf_counter()
        try:
            result = self._generator.generate(question, context)
        except (GeminiNotConfigured, GeminiError) as exc:
            return RAGResult(
                question=question,
                answer=UNKNOWN_ANSWER,
                retrieved_chunks=chunks,
                sources=source_refs,
                max_similarity=max_sim,
                is_unknown=True,
                retrieval_latency_s=retrieval_latency,
                total_latency_s=time.perf_counter() - started,
                note=f"Answer generation failed: {exc}",
            )
        generation_latency = time.perf_counter() - t0

        answer_text = result.text or UNKNOWN_ANSWER
        return RAGResult(
            question=question,
            answer=answer_text,
            retrieved_chunks=chunks,
            sources=source_refs,
            max_similarity=max_sim,
            is_unknown=answer_text.strip() == UNKNOWN_ANSWER,
            retrieval_latency_s=retrieval_latency,
            generation_latency_s=generation_latency,
            total_latency_s=time.perf_counter() - started,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
            note=result.note,
            generator_used=True,
        )