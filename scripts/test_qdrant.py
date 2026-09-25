import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

if not url or not api_key:
    print("❌ Qdrant credentials not found")
    exit()

client = QdrantClient(
    url=url,
    api_key=api_key
)

collections = client.get_collections()

print("✅ Qdrant connection successful!")
print("Collections:", collections)