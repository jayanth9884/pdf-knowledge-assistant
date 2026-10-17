
import os
import math
import hashlib
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader


# -----------------------------------------
# 1. Page setup and API configuration
# -----------------------------------------
st.set_page_config(
    page_title="PDF Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Knowledge Assistant")
st.write("Upload a PDF and ask questions about its contents.")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key or api_key == "paste_your_actual_key_here":
    st.error("Please add your Gemini API key to the .env file.")
    st.stop()

client = genai.Client(api_key=api_key)


# -----------------------------------------
# 2. Create text chunks with page numbers
# -----------------------------------------

def create_chunks(text, page_number, chunk_size=1000, overlap=150):
    chunks = []

    # Split the page into paragraphs
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    current_chunk = ""

    for paragraph in paragraphs:
        # Keep paragraphs together when possible
        if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
            current_chunk = (
                f"{current_chunk}\n{paragraph}"
                if current_chunk
                else paragraph
            )
        else:
            if current_chunk:
                chunks.append({
                    "page": page_number,
                    "text": current_chunk
                })

            # If a paragraph is too large, split it into smaller pieces
            while len(paragraph) > chunk_size:
                piece = paragraph[:chunk_size].strip()

                if piece:
                    chunks.append({
                        "page": page_number,
                        "text": piece
                    })

                paragraph = paragraph[chunk_size - overlap:]

            current_chunk = paragraph

    if current_chunk.strip():
        chunks.append({
            "page": page_number,
            "text": current_chunk.strip()
        })

    return chunks



# -----------------------------------------
# 3. Generate and cache embeddings
# -----------------------------------------
@st.cache_data(show_spinner=False)
def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


# -----------------------------------------
# 4. Prepare and cache an entire PDF
# -----------------------------------------
@st.cache_data(show_spinner=False)
def prepare_pdf(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()

        if page_text:
            chunks.extend(
                create_chunks(page_text, page_number)
            )

    if not chunks:
        raise ValueError(
            "No readable text found. This PDF may contain scanned images."
        )

    embeddings = []

    for index, chunk in enumerate(chunks, start=1):
        embeddings.append(get_embedding(chunk["text"]))

    return len(reader.pages), chunks, embeddings


# -----------------------------------------
# 5. Compare embeddings using cosine similarity
# -----------------------------------------
def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(value * value for value in vector_a)
    )
    magnitude_b = math.sqrt(
        sum(value * value for value in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# -----------------------------------------
# 6. Upload the PDF
# -----------------------------------------
uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()

    # Identify the document by its actual contents.
    document_hash = hashlib.sha256(pdf_bytes).hexdigest()

    st.caption(f"Selected file: {uploaded_file.name}")

    # -------------------------------------
    # 7. Prepare the document
    # -------------------------------------
    if st.button("Prepare PDF for questions", type="primary"):
        try:
            with st.spinner(
                "Preparing PDF and checking cached embeddings..."
            ):
                page_count, chunks, embeddings = prepare_pdf(pdf_bytes)

                st.session_state["prepared_document_hash"] = document_hash
                st.session_state["prepared_chunks"] = chunks
                st.session_state["prepared_embeddings"] = embeddings
                st.session_state["prepared_page_count"] = page_count

            st.success(
                f"PDF prepared successfully: {page_count} pages, "
                f"{len(chunks)} chunks."
            )

        except Exception as error:
            st.error("Could not prepare this PDF.")
            st.caption(str(error))

    # -------------------------------------
    # 8. Show preparation status
    # -------------------------------------
    is_prepared = (
        st.session_state.get("prepared_document_hash")
        == document_hash
    )

    if is_prepared:
        chunks = st.session_state["prepared_chunks"]
        embeddings = st.session_state["prepared_embeddings"]

        st.success("Your PDF is ready for questions.")

        with st.expander("Preview the first text chunk"):
            st.write(chunks[0]["text"])
            st.caption(f"PDF page {chunks[0]['page']}")

        # ---------------------------------
        # 9. Ask questions
        # ---------------------------------
        with st.form("question_form"):
            question = st.text_input(
                "Ask a question about your PDF",
                placeholder="For example: What technologies are mentioned?"
            )
            submitted = st.form_submit_button("Ask AI")

        if submitted:
            if not question.strip():
                st.warning("Please enter a question.")

            else:
                try:
                    with st.spinner(
                        "Searching the PDF and generating an answer..."
                    ):
                        question_embedding = get_embedding(
                            question.strip()
                        )

                        results = []

                        for chunk, embedding in zip(chunks, embeddings):
                            score = cosine_similarity(
                                question_embedding,
                                embedding
                            )

                            results.append({
                                "score": score,
                                "page": chunk["page"],
                                "text": chunk["text"]
                            })

                        results.sort(
                            key=lambda item: item["score"],
                            reverse=True
                        )

                        top_results = results[:3]
                        
                        # Reject irrelevant questions
                        SIMILARITY_THRESHOLD = 0.50

                        if (
                            not top_results
                            or top_results[0]["score"] < SIMILARITY_THRESHOLD
                        ):
                            st.warning(
                                "I couldn't find sufficiently relevant "
                                "information in this PDF to answer your question."
                            )
                        else:
                            # Build context with source page numbers.
                            context = "\n\n".join(
                                f"[PDF page {item['page']}]\n{item['text']}"
                                for item in top_results
                            )

                            prompt = f"""
You are a PDF question-answering assistant.

Answer using only the supplied PDF passages.
Do not invent facts or use outside knowledge.
If the passages do not contain enough information,
say so clearly.
Cite supporting page numbers.

PDF passages:
{context}

Question:
{question.strip()}
"""

                            response = client.models.generate_content(
                                model="gemini-3.8-flash",
                                contents=prompt
                            )

                            st.subheader("Answer")
                            if response.text:
                                st.markdown(response.text)
                            else:
                                st.warning("The model returned an empty answer.")

                            with st.expander("View retrieved passages"):
                                for index, result in enumerate(top_results, start=1):
                                    st.markdown(
                                        f"**Passage {index} — PDF page "
                                        f"{result['page']}**"
                                    )
                                    st.caption(
                                        f"Similarity score: {result['score']:.4f}"
                                    )
                                    st.write(result["text"])
                                    st.divider()

                        

                        # Build context with source page numbers.
                        context = "\n\n".join(
                            f"[PDF page {item['page']}]\n{item['text']}"
                            for item in top_results
                        )

                        prompt = f"""
You are a PDF question-answering assistant.

Rules:
1. Answer using only the supplied PDF passages.
2. Do not invent facts or use outside knowledge.
3. Use the full context of the passages to identify the project
   being described, even if the question uses a different title.
4. If a passage describes a project by its responsibilities and
   technologies, use those details to answer the question when relevant.
5. If the exact project identity cannot be established, explain
   what the passages support without claiming certainty.
6. Cite page numbers using this format: [PDF page 1].

PDF passages:
{context}

Question:
{question.strip()}
"""

                        response = client.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=prompt
                        )

                    st.subheader("Answer")
                    if response.text:
                        st.markdown(response.text)
                    else:
                        st.warning("The model returned an empty answer.")

                    with st.expander("View retrieved passages"):
                        for index, result in enumerate(top_results, start=1):
                            st.markdown(
                                f"**Passage {index} — PDF page "
                                f"{result['page']}**"
                            )
                            st.caption(
                                f"Similarity score: {result['score']:.4f}"
                            )
                            st.write(result["text"])
                            st.divider()

                except Exception as error:
                    error_message = str(error)

                    if "503" in error_message or "UNAVAILABLE" in error_message:
                        st.warning(
                            "Gemini is temporarily unavailable. "
                            "Please wait and try again."
                        )
                    else:
                        st.error("Could not answer your question.")
                        st.caption(error_message)

    else:
        st.info(
            "Click 'Prepare PDF for questions' before asking questions. "
            "If you select a different PDF, prepare that document too."
        )

else:
    st.info("Upload a PDF to get started.")
