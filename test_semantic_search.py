
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

chunks = [
    "Java supports inheritance, allowing one class to acquire properties and methods from another class.",
    "Python lists can store multiple items, and you can access them using indexes.",
    "In object-oriented programming, inheritance helps classes reuse existing code."
]

question = "How can a class reuse another class's functionality?"


def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# Embed every chunk
chunk_embeddings = [
    get_embedding(chunk) for chunk in chunks
]

# Embed the question
question_embedding = get_embedding(question)

# Compare the question against every chunk
results = []

for chunk, embedding in zip(chunks, chunk_embeddings):
    score = cosine_similarity(question_embedding, embedding)
    results.append((score, chunk))

# Sort from most similar to least similar
results.sort(key=lambda item: item[0], reverse=True)

print("\nQuestion:", question)
print("\nMost relevant chunks:\n")

for score, chunk in results:
    print(f"Similarity: {score:.4f}")
    print("Text:", chunk)
    print("-" * 50)
