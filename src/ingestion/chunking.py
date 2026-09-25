CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def create_chunks(pages, document_name, manufacturer):
    all_chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"]

        chunks = chunk_text(
            text,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        for chunk_number, chunk in enumerate(chunks, start=1):
            all_chunks.append({
                "document": document_name,
                "manufacturer": manufacturer,
                "page": page_number,
                "chunk_id": chunk_number,
                "text": chunk
            })

    return all_chunks