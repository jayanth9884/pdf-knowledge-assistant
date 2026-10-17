import os
from dotenv import load_dotenv
from google import genai

print("Step 1: Python file is running!")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: API key not found in .env")
else:
    print("Step 2: API key found!")

    client = genai.Client(api_key=api_key)

    print("Step 3: Sending request to embedding model...")

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents="Java supports inheritance between classes."
    )

    embedding = response.embeddings[0].values

    print("Step 4: Embedding created successfully!")
    print("Number of values:", len(embedding))
    print("First 10 values:", embedding[:10])