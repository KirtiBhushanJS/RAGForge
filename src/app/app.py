import sys
from pathlib import Path

import streamlit as st


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# PROJECT IMPORTS
# =========================================================

from src.config import (
    TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)

from src.retrieval.retriever import retrieve_documents
from src.retrieval.answer_generator import generate_answer


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RAGForge",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

/* =====================================================
   MAIN APPLICATION
   ===================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 18% 8%,
            rgba(99, 102, 241, 0.10),
            transparent 30%
        ),
        radial-gradient(
            circle at 82% 82%,
            rgba(56, 189, 248, 0.06),
            transparent 32%
        ),
        #080a0f;

    color: #f5f7fa;
}


/* Hide Streamlit default UI */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* =====================================================
   SIDEBAR
   ===================================================== */

section[data-testid="stSidebar"] {
    background: #090c11;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}


.sidebar-brand {
    font-size: 25px;
    font-weight: 700;
    letter-spacing: -0.8px;
    margin-bottom: 4px;
}


.sidebar-subtitle {
    color: #8b93a7;
    font-size: 13px;
    margin-bottom: 28px;
}


/* Sidebar button */

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 46px;

    border-radius: 11px;

    border: 1px solid rgba(255, 255, 255, 0.08);

    background: rgba(255, 255, 255, 0.035);

    color: #dbeafe;

    text-align: center;
}


section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(129, 140, 248, 0.10);

    border-color:
        rgba(129, 140, 248, 0.35);
}


/* =====================================================
   MAIN HEADER
   ===================================================== */

.main-title {
    text-align: center;

    font-size: 46px;
    font-weight: 750;

    letter-spacing: -2px;

    margin-top: 70px;
    margin-bottom: 10px;

    background: linear-gradient(
        90deg,
        #ffffff,
        #a5b4fc,
        #67e8f9
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


.main-subtitle {
    text-align: center;

    color: #8b93a7;

    font-size: 16px;

    margin-bottom: 42px;
}


/* =====================================================
   STATISTICS CARDS
   ===================================================== */

.info-card {
    background: rgba(255, 255, 255, 0.035);

    border: 1px solid rgba(255, 255, 255, 0.07);

    border-radius: 16px;

    padding: 20px;

    height: 110px;

    box-sizing: border-box;

    display: flex;

    flex-direction: column;

    justify-content: center;

    backdrop-filter: blur(12px);
}


.info-label {
    color: #7f889d;

    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 0.8px;

    margin-bottom: 7px;
}


.info-value {
    color: #f5f7fa;

    font-size: 25px;

    font-weight: 650;
}


/* =====================================================
   TRY ASKING
   ===================================================== */

.try-heading {
    margin-top: 30px;

    margin-bottom: 14px;

    font-size: 22px;

    font-weight: 650;
}


div[data-testid="stHorizontalBlock"] .stButton > button {
    width: 100% !important;

    min-height: 64px !important;

    height: 64px !important;

    padding: 12px 18px !important;

    border-radius: 14px !important;

    border: 1px solid rgba(255, 255, 255, 0.07) !important;

    background:
        rgba(255, 255, 255, 0.025) !important;

    color: #cbd5e1 !important;

    text-align: left !important;

    font-size: 13px !important;

    line-height: 1.4 !important;

    white-space: normal !important;

    overflow: hidden !important;
}


div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background:
        rgba(129, 140, 248, 0.08) !important;

    border-color:
        rgba(129, 140, 248, 0.40) !important;

    color: white !important;
}


/* =====================================================
   ANSWER CARD
   ===================================================== */

.answer-card {
    background:
        rgba(255, 255, 255, 0.035);

    border:
        1px solid rgba(255, 255, 255, 0.08);

    border-radius: 20px;

    padding: 25px;

    margin-top: 20px;

    box-shadow:
        0 20px 60px rgba(0, 0, 0, 0.20);
}


.answer-header {
    font-size: 15px;

    font-weight: 600;

    color: #c7d2fe;

    margin-bottom: 16px;
}


/* =====================================================
   SOURCE SECTION
   ===================================================== */

.sources-heading {
    font-size: 18px;

    font-weight: 650;

    margin-top: 28px;

    margin-bottom: 12px;
}


/* Native source cards */

.source-card {
    background:
        rgba(255, 255, 255, 0.035);

    border:
        1px solid rgba(255, 255, 255, 0.07);

    border-radius: 14px;

    padding: 14px 16px;

    margin-bottom: 10px;
}


.source-title {
    color: #f1f5f9;

    font-size: 14px;

    font-weight: 600;
}


.source-details {
    color: #8b93a7;

    font-size: 12px;

    margin-top: 5px;
}


/* =====================================================
   CHAT INPUT
   ===================================================== */

div[data-testid="stChatInput"] {
    border-top: none !important;

    background: transparent !important;
}


div[data-testid="stChatInput"] > div {
    background:
        rgba(255, 255, 255, 0.045) !important;

    border:
        1px solid rgba(255, 255, 255, 0.10) !important;

    border-radius:
        10px !important;

    padding:
        4px 8px !important;

    box-shadow:
        0 8px 30px rgba(0, 0, 0, 0.18) !important;
}


div[data-testid="stChatInput"] textarea {
    background:
        transparent !important;

    border:
        none !important;

    border-radius:
        6px !important;

    color:
        white !important;

    box-shadow:
        none !important;
}


div[data-testid="stChatInput"] button {
    border-radius:
        8px !important;

    background:
        rgba(255, 255, 255, 0.08) !important;

    border:
        1px solid rgba(255, 255, 255, 0.08) !important;
}


div[data-testid="stChatInput"] button:hover {
    background:
        rgba(129, 140, 248, 0.20) !important;
}


/* =====================================================
   DIVIDERS
   ===================================================== */

hr {
    border-color:
        rgba(255, 255, 255, 0.08) !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">◈ RAGForge</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Document Intelligence Assistant'
        '</div>',
        unsafe_allow_html=True,
    )


    if st.button(
        "＋ New conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.session_state.pending_question = None

        st.rerun()


    st.markdown("---")


    st.markdown("### Project")

    st.caption("13 smartphone manuals")

    st.caption("5,359 indexed chunks")

    st.caption("Qdrant vector database")

    st.caption("Gemini-powered RAG")


    st.markdown("---")


    st.markdown("### Current configuration")

    st.caption(
        f"Chunk size · {CHUNK_SIZE}"
    )

    st.caption(
        f"Overlap · {CHUNK_OVERLAP}"
    )

    st.caption(
        f"Top-k · {TOP_K}"
    )

    st.caption(
        f"Embedding · {EMBEDDING_MODEL}"
    )


# =========================================================
# WELCOME SCREEN
# =========================================================

if not st.session_state.messages:

    st.markdown(
        '<div class="main-title">'
        'Ask your documents.'
        '</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        '<div class="main-subtitle">'
        'Search across your smartphone manuals '
        'using retrieval-augmented generation.'
        '</div>',
        unsafe_allow_html=True,
    )


    # -----------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(
        3,
        gap="medium",
    )


    with col1:

        st.markdown(
            """
<div class="info-card">
    <div class="info-label">Documents</div>
    <div class="info-value">13</div>
</div>
""",
            unsafe_allow_html=True,
        )


    with col2:

        st.markdown(
            """
<div class="info-card">
    <div class="info-label">Indexed chunks</div>
    <div class="info-value">5,359</div>
</div>
""",
            unsafe_allow_html=True,
        )


    with col3:

        st.markdown(
            """
<div class="info-card">
    <div class="info-label">Manufacturers</div>
    <div class="info-value">3</div>
</div>
""",
            unsafe_allow_html=True,
        )


    # -----------------------------------------------------
    # TRY ASKING
    # -----------------------------------------------------

    st.markdown(
        '<div class="try-heading">Try asking</div>',
        unsafe_allow_html=True,
    )


    suggestion_questions = [
        "How do I take a screenshot on the Galaxy S26 Ultra?",
        "How do I insert a SIM card?",
        "How can I transfer data to my new phone?",
    ]


    qcol1, qcol2, qcol3 = st.columns(
        3,
        gap="medium",
    )


    with qcol1:

        if st.button(
            suggestion_questions[0],
            key="suggestion_1",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                suggestion_questions[0]
            )

            st.rerun()


    with qcol2:

        if st.button(
            suggestion_questions[1],
            key="suggestion_2",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                suggestion_questions[1]
            )

            st.rerun()


    with qcol3:

        if st.button(
            suggestion_questions[2],
            key="suggestion_3",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                suggestion_questions[2]
            )

            st.rerun()


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        if message["role"] == "assistant":

            sources = message.get(
                "sources",
                [],
            )


            if sources:

                st.markdown(
                    '<div class="sources-heading">'
                    'Sources'
                    '</div>',
                    unsafe_allow_html=True,
                )


                shown_sources = set()


                for source in sources:

                    key = (
                        source["document"],
                        source["page"],
                    )


                    if key in shown_sources:
                        continue


                    shown_sources.add(key)


                    # Native Streamlit container
                    # instead of raw HTML card

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"**{source['document']}**"
                        )

                        st.caption(
                            f"{source['manufacturer']} "
                            f"· Page {source['page']}"
                        )


# =========================================================
# CHAT INPUT
# =========================================================

typed_question = st.chat_input(
    "Ask anything about your documents..."
)


# =========================================================
# DETERMINE QUESTION
# =========================================================

question = None


if st.session_state.pending_question:

    question = (
        st.session_state.pending_question
    )

    st.session_state.pending_question = None


elif typed_question:

    question = typed_question


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    with st.chat_message("user"):

        st.markdown(
            question
        )


    # -----------------------------------------------------
    # ASSISTANT
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        # Retrieval

        with st.spinner(
            "Searching your documents..."
        ):

            retrieved_documents = (
                retrieve_documents(
                    question,
                    top_k=TOP_K,
                )
            )


        # Generation

        with st.spinner(
            "Generating answer..."
        ):

            raw_answer = generate_answer(
                question,
                retrieved_documents,
            )


        # -------------------------------------------------
        # REMOVE GEMINI'S SOURCE SECTION
        # -------------------------------------------------

        if "Sources:" in raw_answer:

            answer = raw_answer.split(
                "Sources:",
                1
            )[0].strip()

        else:

            answer = raw_answer.strip()


        # -------------------------------------------------
        # ANSWER
        # -------------------------------------------------

        with st.container(border=True):

            st.markdown(
                "#### ◈ RAGForge"
            )

            st.markdown(
                answer
            )

        # -------------------------------------------------
        # SOURCES
        # -------------------------------------------------

        st.markdown(
            '<div class="sources-heading">'
            'Sources'
            '</div>',
            unsafe_allow_html=True,
        )


        sources = []


        for document in retrieved_documents:

            manufacturer = (
                document.get("manufacturer")
                or "Manufacturer"
            )


            sources.append(
                {
                    "document":
                        document.get(
                            "document",
                            "Unknown document"
                        ),

                    "manufacturer":
                        manufacturer,

                    "page":
                        document.get(
                            "page",
                            "Unknown"
                        ),
                }
            )


        shown_sources = set()


        for source in sources:

            key = (
                source["document"],
                source["page"],
            )


            if key in shown_sources:
                continue


            shown_sources.add(key)


            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{source['document']}**"
                )

                st.caption(
                    f"{source['manufacturer']} "
                    f"· Page {source['page']}"
                )


    # -----------------------------------------------------
    # SAVE RESPONSE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",

            "content": answer,

            "sources": sources,
        }
    )