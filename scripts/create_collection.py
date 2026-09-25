import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

if not url or not api_key:
    print("❌ Qdrant credentials not found")
    raise SystemExit

client = QdrantClient(
    url=url,
    api_key=api_key
)

collection_name = "ragforge_documents"

# Check whether collection already exists
collections = client.get_collections().collections
existing_names = [collection.name for collection in collections]

if collection_name in existing_names:
    print(f"ℹ️ Collection '{collection_name}' already exists.")
else:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )

    print(f"✅ Collection '{collection_name}' created successfully!")

print("\nCurrent collections:")
for collection in client.get_collections().collections:
    print("-", collection.name)