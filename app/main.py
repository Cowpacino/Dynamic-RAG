from typing import List

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.core.config import settings
from app.core.vector_store import vector_store_manager
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    IndexResponse,
    IndexUrlRequest,
)
from app.services.agent_service import agent_service
from app.services.pdf_service import pdf_service
from app.services.search_service import search_service

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A modular RAG system allowing multiple PDF uploads and automated web parsing/search.",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME} API"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint to interact with the AI agent.
    The agent can retrieve info from indexed PDFs/Webpages or search the live internet.
    """
    response = await agent_service.chat(request.message, request.thread_id or "default")
    return ChatResponse(response=response)


@app.post("/index/pdfs", response_model=IndexResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Endpoint to upload and index multiple PDF files.
    The content is parsed as Markdown and stored in the vector database.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    files_to_process = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            continue
        content = await file.read()
        files_to_process.append({"content": content, "filename": file.filename})

    if not files_to_process:
        raise HTTPException(
            status_code=400, detail="No valid PDF files found in upload"
        )

    summary = await pdf_service.upload_and_index_pdfs(files_to_process)
    return IndexResponse(
        status="success",
        message=f"Processed {len(files_to_process)} files.",
        summary=summary,
    )


@app.post("/index/url", response_model=IndexResponse)
async def index_url(request: IndexUrlRequest):
    """
    Endpoint to fetch, parse, and index a specific webpage URL.
    """
    try:
        docs = await search_service.fetch_and_parse_webpage(request.url)
        if not docs:
            return IndexResponse(
                status="error",
                message=f"No content could be extracted from {request.url}",
                summary=None,
            )

        vector_store_manager.add_documents(docs)
        return IndexResponse(
            status="success",
            message=f"Successfully indexed content from {request.url}",
            summary={request.url: f"Indexed {len(docs)} chunks."},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
