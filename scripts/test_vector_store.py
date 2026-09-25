import sys

sys.path.insert(0, "src")

from ingestion.extract_pdf import extract_pages
from ingestion.chunking import create_chunks
from ingestion.embeddings import generate_embeddings
from ingestion.vector_store import upload_chunks, qdrant_client, COLLECTION_NAME


PDF_PATH = "data/documents/Galaxy S26 Ultra.pdf"


# Extract pages
pages = extract_pages(PDF_PATH)

# Create chunks
chunks = create_chunks(
    pages,
    "Galaxy S26 Ultra",
    "Samsung"
)

# Use only the first chunk for this test
test_chunk = chunks[:1]

# Generate embedding
embeddings = generate_embeddings(
    [test_chunk[0]["text"]]
)

print("✅ Chunk created")
print("Page:", test_chunk[0]["page"])
print("Embedding dimension:", len(embeddings[0]))

# Upload one test chunk
uploaded = upload_chunks(
    test_chunk,
    embeddings
)

print("✅ Vector uploaded to Qdrant")
print("Points uploaded:", uploaded)

# Check collection
collection_info = qdrant_client.get_collection(
    COLLECTION_NAME
)

print("Qdrant points:", collection_info.points_count)