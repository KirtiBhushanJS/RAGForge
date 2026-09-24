"""Streamlit UI for the Document Q&A Assistant (RAG).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src.citations import citation_line  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.retriever import RetrievedChunk  # noqa: E402
from src.utils import read_json  # noqa: E402

st.set_page_config(page_title="Document Q&A Assistant", layout="wide")


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    """Cached pipeline (index + embedder loaded once per process)."""
    return RAGPipeline(settings=get_settings())


def load_pipeline() -> RAGPipeline:
    pipeline = get_pipeline()
    if not pipeline.index_ready():
        st.error(
            "No vector index found. Run `python scripts/ingest.py` (after placing "
            "your PDFs in `data/documents/`) and restart the app."
        )
        return None
    return pipeline


def example_questions() -> list[str]:
    path = PROJECT_ROOT / "evaluation" / "questions_sample.json"
    data = read_json(path, default=None)
    if not isinstance(data, list):
        return []
    return [str(q["question"]) for q in data if isinstance(q, dict) and q.get("question")][:6]


def render_sidebar() -> tuple[int, float, dict]:
    """Sidebar controls + index summary. Returns (top_k, threshold)."""
    pipeline = load_pipeline()
    if pipeline is None:
        st.stop()
    summary = pipeline.index_summary()

    st.sidebar.title("Configuration")
    top_k = st.sidebar.slider("Top-k retrieval", min_value=1, max_value=10, value=get_settings().top_k)
    default_threshold = float(get_settings().similarity_threshold)
    threshold = st.sidebar.slider(
        "Similarity threshold",
        min_value=0.00,
        max_value=1.00,
        value=default_threshold,
        step=0.05,
        help="If the best retrieved chunk scores below this, the assistant answers "
        "'I don't know based on the provided documents.'",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Index status")
    st.sidebar.write(f"Documents indexed: **{summary.get('num_documents', 0)}**")
    st.sidebar.write(f"Chunks indexed:     **{summary.get('num_chunks', 0)}**")
    st.sidebar.write(f"Embedding model:    `{summary.get('model')}`")
    st.sidebar.write(f"Embedding dim:      {summary.get('dim')}")
    st.sidebar.write(f"Index created:      {summary.get('created_at')}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Chunking config")
    settings = get_settings()
    st.sidebar.write(f"chunk size: **{settings.chunk_size}** words")
    st.sidebar.write(f"chunk overlap: **{settings.chunk_overlap}** words")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Generation")
    pipeline_proxy = pipeline
    api_ok = pipeline_proxy.generator_configured()
    st.sidebar.write(f"Model: `{settings.gemini_model}`")
    st.sidebar.write(f"API key: {'configured' if api_ok else 'MISSING (add to .env)'}")

    return top_k, threshold


def render_chunk_expander(chunk: RetrievedChunk) -> None:
    with st.expander(citation_line_for_chunk(chunk)):
        col1, col2 = st.columns(2)
        col1.metric("Document", chunk.document)
        col2.metric("Page", chunk.page)
        st.write(f"Similarity: **{chunk.score:.4f}**")
        st.write(f"Chunk ID: `{chunk.chunk_id}`")
        st.write(chunk.text)


def citation_line_for_chunk(chunk: RetrievedChunk) -> str:
    doc = chunk.document
    page = chunk.page
    score = chunk.score
    return f"{doc} — page {page}  (score {score:.3f})"


def render_result_header(question: str) -> None:
    st.markdown(f"### {question}")


def main() -> None:
    st.title("Document Q&A Assistant")
    st.caption(
        "Ask questions about the documents in `data/documents/`. Answers are grounded in "
        "retrieved chunks and cite the exact document + page."
    )

    top_k, threshold = render_sidebar()

    if "history" not in st.session_state:
        st.session_state["history"] = []

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_input("Your question", placeholder="e.g. What is the claim resolution rate?")
        asked = st.form_submit_button("Ask")

    if asked and question.strip():
        pipeline = get_pipeline()
        result = pipeline.answer(question.strip(), top_k=top_k, similarity_threshold=threshold)
        st.session_state["history"].append(
            {"question": question.strip(), "result": result, "top_k": top_k, "threshold": threshold}
        )

    if st.session_state["history"]:
        if st.button("Clear conversation"):
            st.session_state["history"] = []
            st.rerun()

    for item in reversed(st.session_state["history"]):
        result = item["result"]
        render_result_header(item["question"])

        if result.note:
            st.info(result.note)

        st.write(result.answer)
        st.write("---")

        if result.sources:
            st.markdown("#### Sources")
            for ref in result.sources:
                st.markdown("- " + citation_line(ref))

            st.markdown("#### Retrieved chunks")
            for chunk in result.retrieved_chunks:
                render_chunk_expander(chunk)
        else:
            st.caption("No sources were retrieved for this question.")

        st.caption(
            f"max similarity {result.max_similarity:.3f} | retrieval "
            f"{result.retrieval_latency_s * 1000:.1f} ms"
            + (f" | generation {result.generation_latency_s * 1000:.1f} ms" if result.generator_used else "")
            + (f" | total {result.total_latency_s * 1000:.0f} ms")
        )
        st.markdown("---")

    with st.expander("Try an example question"):
        for q in example_questions():
            if st.button(q, key=q):
                pipeline = get_pipeline()
                result = pipeline.answer(q, top_k=top_k, similarity_threshold=threshold)
                st.session_state["history"].append(
                    {"question": q, "result": result, "top_k": top_k, "threshold": threshold}
                )
                st.rerun()


if __name__ == "__main__":
    main()