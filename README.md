# Medical RAG Assistant

A  Retrieval-Augmented Generation (RAG) system for querying medical guidelines documents. Upload any medical PDF and ask questions & answers are grounded strictly in the document with source citations and page references.

Built with ChromaDB, hybrid BM25 + semantic search, MMR deduplication, and Llama-3.3-70b via Groq. Deployed as a local Streamlit web app.

---

## Features

- **PDF Upload via UI** — Upload any medical guidelines PDF directly from the browser. No hardcoded paths.
- **Intelligent Chunking** — `RecursiveCharacterTextSplitter` preserves context across chunk boundaries.
- **Local Embeddings** — `all-MiniLM-L6-v2` runs fully offline via `sentence-transformers`.
- **ChromaDB Vector Store** — In-memory ephemeral store (no disk writes, resets per session).
- **Hybrid Search** — Combines BM25 keyword search + semantic vector search for better recall.
- **Query Expansion** — Automatically rewrites short/vague queries into multiple variants before retrieval.
- **MMR Deduplication** — Maximal Marginal Relevance removes near-duplicate chunks from results.
- **Structured Answers** — Responses follow a consistent Summary → Details → Sources format.
- **Low-Confidence Warning** — Flags answers when the top retrieval score falls below 0.5.
- **Source Transparency** — Every answer shows page number, relevance score, and retrieval method (bm25/semantic).
- **Streamlit Chat UI** — Full chat interface with message history and collapsible source viewer.
- **50MB Upload Limit** — Enforced at both server and application level.

---

## Project Structure

```
medical-rag-assistant/
│
├── app/
│   ├── loader.py            # PDF loading via PyPDFLoader
│   ├── chunker.py           # Text splitting
│   ├── retriever.py         # ChromaDB + BM25 hybrid retrieval + MMR
│   ├── generator.py         # Groq LLM answer generation
│   ├── streamlit_app.py     # Streamlit web UI (main entry point)
│   └── main.py              # Legacy CLI entry point
│
├── data/                    # PDF storage (not committed to git)
│
├── .streamlit/
│   └── config.toml          # Streamlit theme + upload size config
│
├── .env                     # API keys (not committed to git)
├── requirements.txt
└── README.md
```

---

## Architecture

```
User uploads PDF
      │
      ▼
PDFLoader → TextChunker → ChromaDB (EphemeralClient)
                                │
                         ┌──────┴──────┐
                    BM25 index    Vector index
                         └──────┬──────┘
                                │
User query → Query Expansion → Hybrid Search
                                │
                          MMR Deduplication
                                │
                         Top-5 diverse chunks
                                │
                        Groq Llama-3.3-70b
                                │
                    Structured answer + citations
```

---

## Setup

### 1. Get a Groq API Key

1. Go to the [Groq Console](https://console.groq.com/keys).
2. Sign in and click **Create API Key**.
3. Name it (e.g., `medical-rag`) and copy the key immediately.

### 2. Create `.env` file

Create a `.env` file in the project root (same level as `app/`):

```text
GROQ_API_KEY=your_gsk_key_here
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Streamlit (optional)

The `.streamlit/config.toml` file sets the theme and enforces the 50MB upload limit:

```toml
[theme]
primaryColor = "#0F6E56"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#1A1A1A"
font = "sans serif"

[server]
headless = false
port = 8501
maxUploadSize = 50
```

---

## Running the App

```bash
cd app
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

**First run:** The embedding model (`all-MiniLM-L6-v2`) downloads automatically (~80MB, one time only).

---

## How It Works

### 1. PDF Indexing (on upload)

When a PDF is uploaded:
- Pages are extracted and split into overlapping chunks (size: 1000 chars, overlap: 200).
- Each chunk is embedded using `all-MiniLM-L6-v2` and stored in ChromaDB's in-memory vector store.
- A BM25 keyword index is built lazily on the first query.
- Previous document data is wiped automatically on new upload.

### 2. Retrieval Pipeline (on each query)

Every query goes through three stages:

**Stage 1 — Query Expansion**  
Short or vague queries (≤ 3 words) are automatically expanded into multiple variants to improve recall. Example: `"guidelines"` becomes `["guidelines", "definition and explanation of guidelines", "guidelines and rules regarding guidelines", ...]`.

**Stage 2 — Hybrid Search**  
Both search methods run in parallel across all query variants:
- **Semantic search** — ChromaDB cosine similarity against dense embeddings.
- **BM25 search** — Keyword frequency matching for exact medical terminology.

Results from both methods are merged, with duplicates removed.

**Stage 3 — MMR Deduplication**  
Maximal Marginal Relevance re-ranks the merged candidates to balance relevance with diversity. This prevents the same page from appearing 3× in the top results.

### 3. Answer Generation

The top-5 diverse chunks are sent to `llama-3.3-70b-versatile` via Groq with a strict prompt:
- Must use only the provided context.
- Must cite sources inline (`[Source 1]`, `[Source 2]`).
- Must admit if the answer is not found.
- Temperature set to `0.3` for consistency.

Responses are structured as:

```
## Summary
Direct one or two sentence answer.

## Details
Detailed explanation with bullet points and inline citations.

## Sources referenced
- [Source N] Page X — what this source contributed.
```

---

## Retrieval Quality Notes

| Metric | Value |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (384-dim) |
| Chunk size | 1000 chars, 200 overlap |
| Top-k retrieved | 5 |
| Search method | BM25 + semantic (hybrid) |
| Deduplication | MMR (diversity=0.4) |
| Low-confidence threshold | Score < 0.5 |
| Max upload size | 50 MB |

---

## Validation & Safety

| Layer | Check |
|---|---|
| Upload | File size enforced at ≤ 50MB (server + app level) |
| PDF | Warns if very few characters per page (likely scanned/image PDF) |
| Retrieval | Low-confidence warning shown if top score < 0.5 |
| Generation | LLM strictly prohibited from using prior knowledge |
| Generation | Returns a clear "not found" message rather than hallucinating |
| API key | App stops immediately with a clear error if `GROQ_API_KEY` is missing |

---

## Requirements

```
streamlit
chromadb
sentence-transformers
langchain
langchain-community
langchain-text-splitters
rank-bm25
groq
pypdf
python-dotenv
```

---

## Notes

- `embedder.py` is retained in the codebase but is no longer used — ChromaDB handles embedding internally.
- `listModels.py` is a development utility and should not be committed with API keys.
- ChromaDB uses an `EphemeralClient` (RAM only) — data does not persist across app restarts by design.
- The CLI (`main.py`) still works independently if you prefer terminal usage.