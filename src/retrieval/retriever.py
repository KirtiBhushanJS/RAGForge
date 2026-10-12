
from ..ingestion.embeddings import generate_embeddings
from ..ingestion.vector_store import (
    qdrant_client,
    COLLECTION_NAME,
)


def retrieve_documents(query, top_k=5, document_filter=None):
    """
    Retrieve relevant chunks from Qdrant.

    Args:
        query: User's question.
        top_k: Maximum number of chunks to return.
        document_filter: Optional exact document name, such as
            "Galaxy A57 5G". When omitted, search all documents.

    Returns:
        List of retrieved chunks with metadata and similarity scores.
    """
    query_embedding = generate_embeddings([query])[0]

    search_args = {
        "collection_name": COLLECTION_NAME,
        "query": query_embedding,
        "limit": top_k,
        "with_payload": True,
        "with_vectors": False,
    }

    if document_filter:
        search_args["query_filter"] = {
            "must": [
                {
                    "key": "document",
                    "match": {"value": document_filter},
                }
            ]
        }

    search_result = qdrant_client.query_points(**search_args)

    results = []

    for point in search_result.points:
        payload = point.payload or {}

        results.append({
            "score": point.score,
            "document": payload.get("document"),
            "manufacturer": payload.get("manufacturer"),
            "page": payload.get("page"),
            "chunk_id": payload.get("chunk_id"),
            "text": payload.get("text"),
        })

    return results
