# PDF Knowledge Assistant

**Ask questions about your PDF documents and get AI-generated answers grounded in relevant passages.**

The PDF Knowledge Assistant is a Python application built with Streamlit and Google's Gemini API. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant document passages and generate answers based on the uploaded PDF.

##  Features

- **PDF Upload:** Upload PDF documents through a simple web interface.
- **Text Extraction:** Extract text from PDF pages using PyPDF.
- **Text Chunking:** Split extracted text into smaller passages for retrieval.
- **Semantic Search:** Generate embeddings using Google's `gemini-embedding-001` model.
- **Similarity Ranking:** Rank document passages using cosine similarity.
- **AI-Powered Answers:** Use Gemini to answer questions based on retrieved passages.
- **Source References:** Display PDF page numbers and retrieved passages to help users verify answers.
- **Caching:** Cache document preparation and embeddings to reduce repeated API calls.
- **Error Handling:** Display helpful messages for API quota limits and temporary service failures.

##  Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Web interface |
| Google Gemini API | Answer generation |
| Gemini Embeddings | Text and question embeddings |
| PyPDF | PDF text extraction |
| NumPy-free cosine similarity | Similarity calculation implemented in Python |
| python-dotenv | Environment variable configuration |

##  How It Works

The application follows a basic Retrieval-Augmented Generation workflow.

1. **Upload:** The user uploads a PDF document.
2. **Extract:** Text is extracted from each readable PDF page.
3. **Chunk:** The text is divided into manageable passages while preserving page numbers.
4. **Embed:** Each passage is converted into a numerical embedding.
5. **Retrieve:** The question is embedded and compared with document passage embeddings using cosine similarity.
6. **Generate:** The most relevant passages are sent to Gemini as context.
7. **Respond:** The application displays an answer and the retrieved source passages.

### Architecture

```text
       PDF Document
             |
             v
       Text Extraction
             |
             v
        Text Chunking
             |
             v
      Gemini Embeddings
             |
             v
      Document Vectors
             |
User Question ---> Question Embedding
                         |
                         v
                Cosine Similarity
                         |
                         v
                Top Relevant Chunks
                         |
                         v
                 Gemini LLM
                         |
                         v
              Answer + Page Sources
```

##  Getting Started

### Prerequisites

- Python 3.11 or compatible Python version
- Git
- A Google Gemini API key
- An internet connection for API requests

### 1. Clone the repository

```bash
git clone https://github.com/jayanth9884/pdf-knowledge-assistant.git
cd pdf-knowledge-assistant
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment on Windows

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure your Gemini API key

Create a file named `.env` in the project root directory:

```text
GEMINI_API_KEY=your_actual_api_key
```

Replace the placeholder with your own API key.

**Security:** Never commit your `.env` file or expose your API key in source code, screenshots, or public repositories.

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

### 7. Ask questions

1. Upload a PDF.
2. Click **Prepare PDF for questions**.
3. Enter a question about the document.
4. Click **Ask AI**.
5. Review the generated answer and retrieved passages.

##  Project Structure

```text
pdf-knowledge-assistant/
├── app.py                   # Main Streamlit application
├── read_pdf.py              # PDF reading utility
├── search_pdf.py            # PDF retrieval and question-answering prototype
├── test_embeddings.py       # Embedding experiment
├── test_semantic_search.py  # Semantic search experiment
├── requirements.txt         # Python dependencies
├── .gitignore               # Excluded files
├── README.md                # Project documentation
└── data/                    # Local test documents (not committed)
```

##  Limitations

- Scanned PDFs without an embedded text layer may require OCR.
- Answer quality depends on PDF text quality and retrieval relevance.
- Similarity thresholds may need adjustment for different documents.
- Gemini API usage is subject to the applicable model quotas and rate limits.
- The application is designed for document-grounded question answering, but generated answers should still be checked against their cited passages.

##  Future Improvements

- Improve chunk overlap and retrieval quality.
- Add automated unit and integration tests.
- Support OCR for scanned PDFs.
- Add multi-document search.
- Add conversation history.
- Explore a vector database for larger document collections.
- Deploy the application for public access.

##  Author

**Jayanth Janjanam**

- GitHub: [@jayanth9884](https://github.com/jayanth9884)
- Project: [PDF Knowledge Assistant](https://github.com/jayanth9884/pdf-knowledge-assistant)

---

*Built as a hands-on project to explore document processing, embeddings, semantic retrieval, and Retrieval-Augmented Generation.*
