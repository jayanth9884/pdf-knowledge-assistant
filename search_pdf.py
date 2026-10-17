
import os
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai

# 1. Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# 2. Read PDF and create chunks
pdf_path = os.path.join("data", "sample.pdf")
reader = PdfReader(pdf_path)

chunks = []
chunk_size = 500
overlap = 50

for page_number, page in enumerate(reader.pages, start=1):
    page_text = (page.extract_text() or "").strip()

    if not page_text:
        continue

    start = 0

    while start < len(page_text):
        end = start + chunk_size
        chunk_text = page_text[start:end].strip()

        if chunk_text:
            chunks.append({
                "page": page_number,
                "text": chunk_text
            })

        start += chunk_size - overlap

if not chunks:
    raise ValueError("No readable text found in the PDF.")

print("PDF loaded:", len(reader.pages), "pages")
print("Chunks created:", len(chunks))


# 3. Generate embeddings
def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values


print("\nCreating chunk embeddings...")
chunk_embeddings = []

for index, chunk in enumerate(chunks, start=1):
    chunk_embeddings.append(get_embedding(chunk["text"]))

    if index % 10 == 0 or index == len(chunks):
        print(f"Processed {index}/{len(chunks)} chunks")

print("Embeddings ready!")


# 4. Compare vectors using cosine similarity
def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# 5. Ask a question
question = input("\nEnter your question about the PDF: ").strip()

if not question:
    raise ValueError("Question cannot be empty.")

question_embedding = get_embedding(question)


# 6. Retrieve the three most similar chunks
results = []

for chunk, embedding in zip(chunks, chunk_embeddings):
    score = cosine_similarity(question_embedding, embedding)

    results.append({
        "score": score,
        "page": chunk["page"],
        "text": chunk["text"]
    })

results.sort(key=lambda item: item["score"], reverse=True)
top_results = results[:3]

print("\nRetrieved passages:")
for result in top_results:
    print(f"Page {result['page']} | Similarity: {result['score']:.4f}")
    print(result["text"])
    print("-" * 50)


# 7. Ask Gemini to answer using only the retrieved text
context_parts = []

for result in top_results:
    context_parts.append(
        f"[PDF page {result['page']}]\n{result['text']}"
    )

context = "\n\n".join(context_parts)

prompt = f"""
You are a PDF question-answering assistant.

Answer the question using only the supplied PDF passages.
Do not use outside knowledge or invent facts.
If the passages do not contain enough information to answer,
clearly say that the retrieved passages do not provide the answer.
Cite the PDF page number for claims you support.

PDF passages:
{context}

Question:
{question}
"""

print("\nGenerating answer with Gemini...")

try:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    print("\nAnswer:")
    print(response.text)

except Exception as error:
    print("\nGemini request failed.")
    print("Technical details:", error)
