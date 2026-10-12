import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")


gemini_client = genai.Client(api_key=GEMINI_API_KEY)

LLM_MODEL = "gemini-3.5-flash-lite"

ABSTENTION_MESSAGE = (
    "I couldn't find enough information in the provided documents."
)


def generate_answer(question, retrieved_documents):
    """
    Generate a grounded answer using retrieved document chunks.

    Answer from supported facts, even when the context does not
    provide every detail. Abstain when the requested answer cannot
    be established from the retrieved evidence.
    """

    if not retrieved_documents:
        return ABSTENTION_MESSAGE

    context_parts = []

    for index, document in enumerate(retrieved_documents, start=1):
        context_parts.append(
            f"""
Source {index}
Document: {document.get('document', 'Unknown')}
Page: {document.get('page', 'Unknown')}

Content:
{document.get('text', '')}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a careful document question-answering assistant.

Answer the user's question using ONLY the retrieved document
context below. Do not use outside knowledge or assumptions.

Follow these rules:

1. First, check whether the context contains evidence that directly
   answers the question.
2. If it contains enough evidence for a useful answer, answer clearly
   and concisely. Do not abstain merely because the wording of the
   question differs from the wording in the manual.
3. If the context supports only part of the answer, provide that
   supported part and clearly state what the documents do not specify.
   Do not invent missing details.
4. If the context is related to the topic but does not support the
   requested fact, use this exact response:
   "{ABSTENTION_MESSAGE}"
5. Never infer private information, exact prices, future dates,
   device-specific specifications, or other precise facts that the
   retrieved documents do not establish.
6. Use only source document names and page numbers supplied in the
   context. Cite sources that actually support your answer.
7. If multiple sources disagree, describe the disagreement rather
   than silently choosing one.

At the end of a substantive answer, use this format:

Sources:
- Document Name — Page X

Do not add a Sources section to an abstention.

User question:
{question}

Retrieved document context:
{context}
"""

    response = gemini_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    if not response.text:
        return ABSTENTION_MESSAGE

    return response.text.strip()


if __name__ == "__main__":
    from .retriever import retrieve_documents

    question = "How do I take a screenshot on the Galaxy S26 Ultra?"

    print("\nQuestion:")
    print(question)

    retrieved_documents = retrieve_documents(question, top_k=5)
    answer = generate_answer(question, retrieved_documents)

    print("\n==============================")
    print("RAG Answer")
    print("==============================")
    print(answer)
