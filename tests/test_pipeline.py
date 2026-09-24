"""End-to-end pipeline tests. The LLM is fully stubbed - no real API calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings
from src.generator import GenerationResult
from src.generator import GeminiNotConfigured
from src.rag_pipeline import RAGPipeline, UNKNOWN_ANSWER
from src.retriever import Retriever


class StubGenerator:
    """Deterministic fake LLM used to prove the pipeline mechanics."""

    def __init__(self, answer: str = "Base answer from context.", configured: bool = True):
        self.calls: list[tuple[str, str]] = []
        self.answer = answer
        self._configured = configured

    def is_configured(self) -> bool:
        return self._configured

    def generate(self, question: str, context: str) -> GenerationResult:
        self.calls.append((question, context))
        return GenerationResult(text=self.answer, model="stub", latency_s=0.0)


@pytest.fixture
def pipeline(sample_chunks, built_retriever) -> RAGPipeline:
    retriever: Retriever = built_retriever(sample_chunks)
    return RAGPipeline(retriever=retriever, generator=StubGenerator())


def test_answer_uses_retriever_and_generator(pipeline: RAGPipeline) -> None:
    result = pipeline.answer("fleet trucks", similarity_threshold=0.01)
    assert result.answer == "Base answer from context."
    assert result.is_unknown is False
    assert result.generator_used is True
    assert result.retrieved_chunks
    assert result.sources
    # Sources carry the real page numbers from the retrieval metadata.
    assert any("trucks.pdf" == s.document for s in result.sources)


def test_unknown_answer_when_below_threshold(pipeline: RAGPipeline) -> None:
    result = pipeline.answer("fleet trucks", similarity_threshold=0.99)
    assert result.is_unknown is True
    assert result.answer == UNKNOWN_ANSWER
    assert result.generator_used is False


def test_pipeline_handles_empty_question(pipeline: RAGPipeline) -> None:
    result = pipeline.answer("   ")
    assert result.answer == "Please enter a question."
    assert result.is_unknown is True


def test_pipeline_reports_missing_api_key(sample_chunks, built_retriever) -> None:
    retriever = built_retriever(sample_chunks)
    pipe = RAGPipeline(retriever=retriever, generator=StubGenerator(configured=False))
    result = pipe.answer("fleet trucks", similarity_threshold=0.01)
    assert result.answer == UNKNOWN_ANSWER
    assert "GEMINI_API_KEY" in result.note
    assert result.retrieved_chunks  # retrieval still works without a key


def test_pipeline_reports_missing_index(tmp_path: Path) -> None:
    settings = Settings(
        index_path=tmp_path / "missing.faiss",
        metadata_path=tmp_path / "missing.json",
    )
    pipe = RAGPipeline(settings=settings, generator=StubGenerator())
    result = pipe.answer("anything")
    assert result.is_unknown is True
    assert "ingest.py" in result.note


def test_pipeline_generation_failure(sample_chunks, built_retriever) -> None:
    from src.generator import GeminiError

    class FailingGenerator(StubGenerator):
        def generate(self, question, context):
            raise GeminiError("boom")

    retriever = built_retriever(sample_chunks)
    pipe = RAGPipeline(retriever=retriever, generator=FailingGenerator())
    result = pipe.answer("fleet trucks", similarity_threshold=0.01)
    assert result.answer == UNKNOWN_ANSWER
    assert "boom" in result.note


def test_generator_not_configured_raises() -> None:
    from src.generator import GeminiGenerator

    gen = GeminiGenerator(api_key="your_api_key_here")
    assert gen.is_configured() is False
    with pytest.raises(GeminiNotConfigured):
        gen.generate("Q", "ctx")