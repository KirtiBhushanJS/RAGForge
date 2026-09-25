import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")


gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSION = 768


def generate_embeddings(texts):
    contents = []

    for text in texts:
        embedding_text = (
            f"title: smartphone user manual | "
            f"text: {text}"
        )

        contents.append(
            types.Content(
                parts=[
                    types.Part.from_text(
                        text=embedding_text
                    )
                ]
            )
        )

    result = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=contents,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    return [
        embedding.values
        for embedding in result.embeddings
    ]