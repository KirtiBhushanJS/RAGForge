import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found in .env"
    )


gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


LLM_MODEL = "gemini-3.5-flash-lite"


def generate_answer(question, retrieved_documents):
    """
    Generate an answer using only the retrieved document chunks.

    Args:
        question: User's question.
        retrieved_documents: Chunks returned by the retriever.

    Returns:
        Generated answer from Gemini.
    """

    if not retrieved_documents:
        return (
            "I couldn't find relevant information "
            "in the provided documents."
        )

    context_parts = []

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context_parts.append(
            f"""
Source {index}
Document: {document['document']}
Manufacturer: {document['manufacturer']}
Page: {document['page']}

Content:
{document['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the retrieved document context below.

Do not use outside knowledge.

If the retrieved context does not contain enough
information to answer the question, say:

"I couldn't find enough information in the provided documents."

Keep the answer clear and concise.

At the end of the answer, provide the sources using
this format:

Sources:
- Document — Page X

Do not invent document names or page numbers.

User question:
{question}

Retrieved document context:
{context}
"""

    response = gemini_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return response.text


if __name__ == "__main__":

    from .retriever import retrieve_documents

    question = (
        "How do I take a screenshot "
        "on the Galaxy S26 Ultra?"
    )

    print("\n🔎 Question:")
    print(question)

    retrieved_documents = retrieve_documents(
        question,
        top_k=5
    )

    answer = generate_answer(
        question,
        retrieved_documents
    )

    print("\n==============================")
    print("RAG Answer")
    print("==============================")

    print(answer)