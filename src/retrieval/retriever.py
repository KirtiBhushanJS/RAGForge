from ..ingestion.embeddings import generate_embeddings
from ..ingestion.vector_store import (
    qdrant_client,
    COLLECTION_NAME,
)


def retrieve_documents(query, top_k=5):
    """
    Retrieve the most relevant document chunks from Qdrant.

    Args:
        query: User's question.
        top_k: Number of relevant chunks to retrieve.

    Returns:
        List of retrieved chunks with metadata and similarity scores.
    """

    # Generate embedding for the user's question
    query_embedding = generate_embeddings(
        [query]
    )[0]

    # Search Qdrant for the most similar chunks
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results = []

    for point in search_result.points:

        payload = point.payload

        results.append({
            "score": point.score,
            "document": payload.get("document"),
            "manufacturer": payload.get("manufacturer"),
            "page": payload.get("page"),
            "chunk_id": payload.get("chunk_id"),
            "text": payload.get("text"),
        })

    return results


if __name__ == "__main__":

    question = (
        "How do I take a screenshot "
        "on the Galaxy S26 Ultra?"
    )

    print("\n🔎 Question:")
    print(question)

    results = retrieve_documents(
        question,
        top_k=5
    )

    print("\n==============================")
    print("Retrieved Documents")
    print("==============================")

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n--- Result {index} ---"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Document: "
            f"{result['document']}"
        )

        print(
            f"Manufacturer: "
            f"{result['manufacturer']}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Text:\n"
            f"{result['text']}"
        )