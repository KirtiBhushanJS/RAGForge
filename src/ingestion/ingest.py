import time
from pathlib import Path

from .extract_pdf import extract_pages
from .chunking import create_chunks
from .embeddings import generate_embeddings
from .vector_store import (
    upload_chunks,
    qdrant_client,
    COLLECTION_NAME,
    create_point_id,
)


DOCUMENTS_DIR = Path("data/documents")

# Number of chunks sent to Gemini in one API request
BATCH_SIZE = 50

# Wait time when Gemini rate limit is reached
RATE_LIMIT_WAIT = 65


def get_manufacturer(document_name):
    if document_name.startswith("Galaxy"):
        return "Samsung"

    if document_name.startswith("Motorola"):
        return "Motorola"

    if document_name.startswith("One Plus") or document_name.startswith("OnePlus"):
        return "OnePlus"

    return "Unknown"


def process_pdf(pdf_path):
    document_name = pdf_path.stem
    manufacturer = get_manufacturer(document_name)

    pages = extract_pages(pdf_path)

    chunks = create_chunks(
        pages,
        document_name,
        manufacturer
    )

    return chunks


def process_all_pdfs():
    pdf_files = sorted(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        print("❌ No PDF files found.")
        return []

    print(f"Found {len(pdf_files)} PDF files.")

    all_chunks = []

    for pdf_path in pdf_files:

        print(f"\n📄 Processing: {pdf_path.name}")

        chunks = process_pdf(pdf_path)

        print(f"   Generated {len(chunks)} chunks")

        all_chunks.extend(chunks)

    print("\n--------------------------------")
    print(f"Total chunks: {len(all_chunks)}")
    print("--------------------------------")

    return all_chunks


def get_existing_point_ids():
    existing_ids = set()

    offset = None

    while True:

        points, offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        for point in points:
            existing_ids.add(str(point.id))

        if offset is None:
            break

    return existing_ids


def remove_already_ingested(chunks):
    existing_ids = get_existing_point_ids()

    print(f"\nExisting Qdrant points: {len(existing_ids)}")

    remaining_chunks = []

    for chunk in chunks:
        point_id = create_point_id(chunk)

        if point_id not in existing_ids:
            remaining_chunks.append(chunk)

    skipped = len(chunks) - len(remaining_chunks)

    print(f"Already ingested chunks skipped: {skipped}")
    print(f"Chunks remaining to ingest: {len(remaining_chunks)}")

    return remaining_chunks


def ingest_chunks(chunks):

    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):

        batch = chunks[
            start:start + BATCH_SIZE
        ]

        print(
            f"\n🔄 Embedding chunks "
            f"{start + 1}-{min(start + BATCH_SIZE, total)} "
            f"of {total}"
        )

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        while True:

            try:

                embeddings = generate_embeddings(texts)

                uploaded = upload_chunks(
                    batch,
                    embeddings
                )

                print(
                    f"   ✅ Uploaded {uploaded} vectors"
                )

                break

            except Exception as error:

                error_text = str(error)

                if (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                ):

                    print(
                        "\n⚠️ Gemini rate limit reached."
                    )

                    print(
                        f"Waiting {RATE_LIMIT_WAIT} seconds..."
                    )

                    time.sleep(RATE_LIMIT_WAIT)

                    print(
                        "Retrying the same batch..."
                    )

                else:
                    raise

        # Small delay between successful batches
        time.sleep(1)


if __name__ == "__main__":

    print("🚀 Starting RAGForge ingestion...\n")

    all_chunks = process_all_pdfs()

    if not all_chunks:
        raise SystemExit

    chunks_to_ingest = remove_already_ingested(
        all_chunks
    )

    if not chunks_to_ingest:

        print(
            "\n🎉 All chunks are already present in Qdrant!"
        )

    else:

        ingest_chunks(chunks_to_ingest)

        print(
            "\n🎉 Ingestion completed successfully!"
        )

    collection_info = qdrant_client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"\nQdrant points: {collection_info.points_count}"
    )