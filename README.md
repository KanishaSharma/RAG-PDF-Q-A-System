# RAG-PDF-Q-A-System
Ask questions about any PDF using LangChain · FAISS · HuggingFace Embeddings · Groq LLM · FastAPI

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green?style=flat-square)](https://langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow?style=flat-square&logo=huggingface)](https://huggingface.co)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)](https://docker.com)

---

## What it does

Upload any PDF — a research paper, contract, report, or textbook — and ask natural language questions about it. The system retrieves the most relevant sections using semantic search and generates a grounded, source-cited answer using an LLM.

**No hallucination by design** — the LLM is constrained to answer only from the retrieved document chunks. If the answer isn't in the PDF, it says so.

---

## Demo

```
POST /upload  →  index a 45-page research paper  →  23 seconds, 312 chunks

POST /ask  →  "What evaluation metrics were used?"
→  "The authors used F1-score, precision, and recall on a held-out test set
    of 2,000 samples, reporting a final F1 of 0.84."
   Sources: page 8, page 12
```

---

## Architecture

```
User uploads PDF
      │
      ▼
┌──────────────────────┐
│   PyMuPDF Parser     │  ← Extracts raw text page by page
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Text Chunker        │  ← RecursiveCharacterTextSplitter
│  chunk=500, overlap=50│    preserves sentence boundaries
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  HuggingFace         │  ← all-MiniLM-L6-v2
│  Embeddings          │    384-dim dense vectors, runs locally
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  FAISS Vector Store  │  ← In-memory index, cosine similarity
│  (in-memory)         │    top-k=4 chunk retrieval
└──────────┬───────────┘
           │  User asks a question
           ▼
┌──────────────────────┐
│  Query Embedding     │  ← Same HuggingFace model
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  FAISS Retrieval     │  ← Cosine similarity search
│  (top-4 chunks)      │    returns most relevant context
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  LangChain RetrievalQA│  ← Stuffs context into prompt
│  + Groq LLM          │    llama3-8b-8192, temp=0.2
│  (llama3-8b)         │
└──────────┬───────────┘
           │
           ▼
     Answer + Source Pages
      served via FastAPI
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| PDF parsing | PyMuPDF | Fast, accurate text extraction |
| Text splitting | LangChain RecursiveCharacterTextSplitter | Respects sentence/paragraph boundaries |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Free, runs locally, 384-dim, strong on Q&A |
| Vector store | FAISS (CPU) | Sub-millisecond similarity search |
| LLM | Groq (llama3-8b-8192) | Free API, ~200 tokens/sec, low latency |
| Orchestration | LangChain RetrievalQA | Handles retrieval-to-prompt pipeline |
| API | FastAPI (async) | Production-grade, auto-docs at /docs |
| Language | Python 3.10+ | — |

---

## Key Features

- **Source citations** — every answer includes the page numbers the context came from
- **Hallucination-resistant** — system prompt strictly limits LLM to retrieved context only
- **Local embeddings** — HuggingFace model runs on CPU, no embedding API cost
- **Fast inference** — Groq serves llama3 at ~200 tokens/sec; typical answer in < 3 seconds
- **Any PDF** — works on research papers, contracts, textbooks, reports, manuals
- **Clean REST API** — `/upload`, `/ask`, `/status`, `/reset` endpoints with auto-docs

---

## Project Structure

```
rag-pdf-qa/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app — all endpoints
│   └── rag_engine.py    # Core RAG logic — load, embed, retrieve, generate
├── tests/
│   └── test_rag.py      # Unit + integration tests (pytest)
├── data/                # Drop PDFs here for testing (gitignored)
├── .env.example         # Environment variable template
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Get a free Groq API key

Go to [console.groq.com](https://console.groq.com) → sign up free → create an API key.
Groq gives you a generous free tier (14,400 requests/day).

### 2. Clone and install

```bash
git clone https://github.com/your-username/rag-pdf-qa.git
cd rag-pdf-qa

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Open .env and set your GROQ_API_KEY
```

### 4. Run the API

```bash
uvicorn app.main:app --reload
```

API is live at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---

## API Reference

### `POST /upload`
Upload and index a PDF.

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your-document.pdf"
```

**Response:**
```json
{
  "filename": "your-document.pdf",
  "chunks_indexed": 312,
  "message": "PDF indexed successfully into 312 chunks. You can now ask questions."
}
```

---

### `POST /ask`
Ask a question about the loaded PDF.

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings of this document?"}'
```

**Response:**
```json
{
  "question": "What are the main findings of this document?",
  "answer": "The main findings indicate that the proposed approach outperforms baseline models by 18% on the benchmark dataset, with particular gains in recall for rare categories.",
  "sources": [
    { "page": 7, "snippet": "Our method achieves 18% improvement over the strongest baseline..." },
    { "page": 11, "snippet": "Recall for rare categories improved from 0.61 to 0.79..." }
  ]
}
```

---

### `GET /status`
Check if a PDF is loaded and the system is ready.

```json
{
  "ready": true,
  "loaded_pdf": "your-document.pdf"
}
```

---

### `DELETE /reset`
Clear the current index and load a new PDF.

```json
{ "message": "Index cleared. Upload a new PDF to continue." }
```

---

## Core RAG Logic

```python
# app/rag_engine.py — simplified

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

def load_pdf(pdf_path: str) -> int:
    # 1. Parse PDF pages
    pages = PyMuPDFLoader(pdf_path).load()

    # 2. Split into overlapping chunks
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(pages)

    # 3. Embed with HuggingFace (runs locally, free)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Store in FAISS for fast similarity search
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5. Build retrieval QA chain with Groq LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(model_name="llama3-8b-8192", temperature=0.2),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )

    return len(chunks)
```

---

## Run with Docker

```bash
docker build -t rag-pdf-qa .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key_here rag-pdf-qa
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Why This Architecture

| Decision | Alternative | Reason chosen |
|---|---|---|
| FAISS (in-memory) | Pinecone / ChromaDB | Zero infra cost, sufficient for single-session use |
| HuggingFace embeddings | OpenAI embeddings | Free, local, no API cost, comparable quality for Q&A |
| Groq LLM | OpenAI GPT-4 | Free tier, significantly faster (~200 tok/s vs ~40) |
| chunk_size=500 | Larger chunks | Balances context richness vs retrieval precision |
| overlap=50 | No overlap | Prevents answers being cut at chunk boundaries |

---

## Possible Extensions

- **Persistent vector store** — swap FAISS in-memory for ChromaDB on disk to survive restarts
- **Multi-PDF support** — maintain a named index per document, query across all
- **Streaming responses** — use LangChain streaming + FastAPI `StreamingResponse` for real-time token output
- **Re-ranking** — add a cross-encoder re-ranker after FAISS retrieval for higher precision
- **Frontend** — add a React/HTML upload + chat UI on top of the existing API

---

## License

MIT License — see [LICENSE](LICENSE) for details.
