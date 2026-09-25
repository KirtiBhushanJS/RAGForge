import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found")
    raise SystemExit

client = genai.Client(api_key=api_key)

text = "The Galaxy S26 Ultra supports Wi-Fi connectivity."

result = client.models.embed_content(
    model="gemini-embedding-2",
    contents=text,
    config=types.EmbedContentConfig(
        output_dimensionality=768
    )
)

embedding = result.embeddings[0].values

print("✅ Gemini embedding successful!")
print("Dimension:", len(embedding))
print("First 5 values:", embedding[:5])