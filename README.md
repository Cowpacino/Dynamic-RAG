# Modular RAG Agent with FastAPI

This project is a highly modularized RAG (Retrieval-Augmented Generation) system built with FastAPI and LangGraph. It allows for dynamic ingestion of multiple PDFs, automated web parsing, and real-time internet searching.

## Key Features

- **Modular Architecture**: Separated into `core`, `services`, `models`, and `tools` for better maintainability.
- **FastAPI Endpoints**:
    - `POST /index/pdfs`: Upload and index multiple PDF files simultaneously.
    - `POST /index/url`: Manually trigger indexing of a specific webpage.
    - `POST /chat`: Interact with the AI agent.
- **Automated Web Parsing**: Uses `pymupdf4llm` for PDFs (preserving Markdown structure) and `WebBaseLoader` with `BeautifulSoup` for clean web content extraction.
- **Intelligent Agent (ReAct)**:
    - Automatically searches the internet if local context is insufficient.
    - Can visit specific URLs provided by the user or found during search.
    - Manages its own "browsing" and "indexing" logic.

## Project Structure

```text
rag-agent/app/
├── api/            # Future API route versioning
├── core/           # Configuration and Vector Store initialization
├── models/         # Pydantic schemas for request/response
├── services/       # Business logic (PDF processing, Web searching, Agent management)
├── tools/          # LangChain tools for the agent
└── main.py         # FastAPI application entry point
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install fastapi uvicorn langchain-openai langchain-chroma pymupdf4llm duckduckgo-search langgraph bs4 python-multipart
    ```

2.  **Environment Variables**:
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=your_openai_key
    USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64)
    CHROMA_DB_DIR=./chroma_db
    COLLECTION_NAME=modular_rag
    MODEL_NAME=gpt-4o-mini
    ```

3.  **Run the Server**:
    ```bash
    python -m app.main
    ```

## Usage Examples

### 1. Uploading PDFs
Use an API client like Postman or `curl` to send multiple files:
```bash
curl -X POST "http://localhost:8000/index/pdfs" -F "files=@doc1.pdf" -F "files=@doc2.pdf"
```

### 2. Chatting with the Agent
The agent will decide whether to use local documents or the web:
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What does the uploaded PDF say about the project timeline?"}'
```

### 3. Automated Web Access
Ask the agent to look at a specific site:
```json
{
  "message": "Can you check https://example.com and summarize their latest news?"
}
```
The agent will use the `browse_webpage` tool, parse the HTML automatically, and provide a summary.
