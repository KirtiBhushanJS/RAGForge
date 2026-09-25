from pathlib import Path

from .extract_pdf import extract_pages
from .chunking import create_chunks
from .embeddings import generate_embeddings
from .vector_store import upload_chunks


DOCUMENTS_DIR = Path("data/documents")
BATCH_SIZE = 10


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


def ingest_chunks(chunks):
    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):

        batch = chunks[start:start + BATCH_SIZE]

        print(
            f"\n🔄 Embedding chunks "
            f"{start + 1}-{min(start + BATCH_SIZE, total)} "
            f"of {total}"
        )

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        embeddings = generate_embeddings(texts)

        uploaded = upload_chunks(
            batch,
            embeddings
        )

        print(f"   ✅ Uploaded {uploaded} vectors")


if __name__ == "__main__":

    chunks = process_all_pdfs()

    if chunks:
        ingest_chunks(chunks)

        print("\n🎉 Full ingestion completed successfully!")