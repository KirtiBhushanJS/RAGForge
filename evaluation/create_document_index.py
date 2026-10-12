
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client.models import PayloadSchemaType
from src.ingestion.vector_store import (
    qdrant_client,
    COLLECTION_NAME,
)


def main():
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="document",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )

    print("Keyword index created for the 'document' field.")

    info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Points: {info.points_count}")


if __name__ == "__main__":
    main()
