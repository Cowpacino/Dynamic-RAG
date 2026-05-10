# 🤖 Multi-Modal RAG Agent

A modular, multi-modal Retrieval-Augmented Generation (RAG) system built with **FastAPI** and **LangGraph**. Supports ingestion and querying of PDFs, webpages, and videos through a ReAct agent with tool-use capabilities.

---

## ✨ Features

- **PDF Ingestion** — Upload multiple PDFs; content is extracted as Markdown (preserving structure) and indexed in ChromaDB.
- **Webpage Indexing** — Fetch and index any public URL into the vector store for later retrieval.
- **Video Understanding** — Upload videos; frames are extracted and embedded using OpenCLIP for visual similarity search.
- **ReAct Agent** — A LangGraph-powered agent that intelligently chooses between:
  - Local vector store retrieval (PDFs & web pages)
  - Live internet search (DuckDuckGo)
  - Direct webpage browsing (Selenium)
  - Visual video frame retrieval (OpenCLIP + Vision LLM)
- **Dual Embedding Strategy**
  - Text: OpenAI embeddings for PDF/web content
  - Visual: OpenCLIP (local) for video frames
- **Background Video Processing** — Large video uploads are handled asynchronously without blocking the server.

---

## 🏗️ Project Structure

```
DYNAMIC-RAG/
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── vector_store.py
│   ├── models/
│   ├── services/
│   │   ├── __init__.py
│   │   ├── agent_service.py
│   │   ├── cleanup_temp.py
│   │   ├── frame_extraction.py
│   │   ├── pdf_service.py
│   │   └── search_service.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── browse_tool.py
│   │   ├── retrieval_tool.py
│   │   ├── video_retrieval_tool.py
│   │   └── web_search_tool.py
│   ├── __init__.py
│   └── main.py
├── chroma_db/
├── extracted_frames/
├── .env
├── .gitignore
├── .python-version
├── document.pdf
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## ⚙️ Setup

### 1. Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Google Chrome (for Selenium-based web browsing)
- [Ollama](https://ollama.com/) running locally (for local LLM inference)

### 2. Install Dependencies

```bash
pip install uv
uv sync
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Pull Ollama Models

```bash
ollama pull qwen2.5:3b       # Text agent
ollama pull qwen3-vl:2b      # Vision-language model (video queries)
```

### 4. Configure Environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key

# Optional — defaults shown
USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64)
CHROMA_DB_DIR=./chroma_db
COLLECTION_NAME=modular_rag
MODEL_NAME=gpt-4o-mini
```

> **Note:** `OPENAI_API_KEY` is required for text/PDF embedding. The agent LLM itself runs locally via Ollama.

---

## 🚀 Running the Server

```bash
python -m app.main
```

The API will be available at `http://localhost:8000`.

Interactive docs: `http://localhost:8000/docs`

---

## 📡 API Endpoints

### `POST /chat`
Send a message to the agent. It will automatically choose the best tool (local retrieval, web search, browsing, or video frames).

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What does the uploaded PDF say about the project timeline?"}'
```

```json
{
  "message": "Your question here",
  "thread_id": "optional-session-id"
}
```

---

### `POST /index/pdfs`
Upload and index one or more PDF files.

```bash
curl -X POST "http://localhost:8000/index/pdfs" \
     -F "files=@doc1.pdf" \
     -F "files=@doc2.pdf"
```

---

### `POST /index/url`
Fetch and index a webpage URL.

```bash
curl -X POST "http://localhost:8000/index/url" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/article"}'
```

---

### `POST /index/video`
Upload a video file. Frames are extracted and indexed in the background — the endpoint returns immediately.

```bash
curl -X POST "http://localhost:8000/index/video" \
     -F "file=@my_video.mp4"
```

Video queries are automatically routed to the vision-language model when the message contains keywords like `video`, `frame`, `clip`, `scene`, or `timestamp`.

---

## 🧠 How It Works

```
User Query
    │
    ▼
AgentService.chat()
    │
    ├─ Contains video keywords? ──► OpenCLIP frame retrieval → VLM (qwen3-vl)
    │
    └─ Otherwise ────────────────► LangGraph ReAct Agent (qwen2.5)
                                        │
                                        ├─ retrieve_from_vector_store  (PDFs / web)
                                        ├─ search_the_internet          (DuckDuckGo)
                                        ├─ browse_webpage               (Selenium)
                                        └─ retrieve_video_content       (OpenCLIP)
```

**PDF pipeline:** `pymupdf4llm` → Markdown → `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter` → OpenAI embeddings → ChromaDB

**Video pipeline:** `OpenCV` frame extraction (0.5 fps default) → OpenCLIP visual embeddings → ChromaDB (`pure_visual_frames` collection)

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `langgraph` | ReAct agent orchestration |
| `langchain-chroma` | Vector store integration |
| `langchain-openai` | OpenAI text embeddings |
| `langchain-ollama` | Local LLM inference |
| `pymupdf4llm` | PDF → Markdown extraction |
| `open-clip-torch` | Visual embeddings for video frames |
| `opencv-python` | Video frame extraction |
| `selenium` | Headless web browsing |
| `duckduckgo-search` | Internet search |

---

## 🗒️ Notes

- Video processing runs in the background; large files may take time to index. Query video content after processing completes.
- The `extracted_frames/` directory stores frame images locally — ensure sufficient disk space for long videos.
- ChromaDB persists to `./chroma_db` by default; delete this folder to reset the knowledge base.
- Consecutive `ws1` in combos require precise timing — just kidding, wrong README.
