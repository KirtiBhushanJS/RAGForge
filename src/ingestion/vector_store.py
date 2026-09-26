import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Qdrant credentials not found in .env")


qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


COLLECTION_NAME = "ragforge_documents"


def create_point_id(chunk):
    unique_text = (
        f"{chunk['document']}|"
        f"{chunk['page']}|"
        f"{chunk['chunk_id']}"
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            unique_text
        )
    )


def upload_chunks(chunks, embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must match."
        )

    points = []

    for chunk, embedding in zip(chunks, embeddings):

        point_id = create_point_id(chunk)

        payload = {
            "text": chunk["text"],
            "document": chunk["document"],
            "manufacturer": chunk["manufacturer"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"]
        }

        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
        )

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
        timeout=120
    )

    return len(points)