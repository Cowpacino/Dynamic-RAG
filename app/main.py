from typing import List
import tempfile
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile,BackgroundTasks
import aiofiles

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
from app.services.cleanup_temp import process_video_heavy_lifting,cleanup_temporary_file




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

@app.post("/index/video")
async def upload_video(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    # 1. Validate the file type
    if not file.content_type.startswith("video/"):
        return {"error": "Invalid file type. Please upload a video."}

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}")
    tmp_file_path = tmp_file.name

    try:
        # 2. Async chunked writing for large files
        # This prevents the server from freezing while saving a 500MB video
        async with aiofiles.open(tmp_file_path, 'wb') as out_file:
            # Read in 1MB chunks
            while content := await file.read(1024 * 1024):  
                await out_file.write(content)

        # 3. Schedule the heavy processing as a background task
        # Do NOT await the heavy processing here, or the request will time out
        background_tasks.add_task(process_video_heavy_lifting, tmp_file_path)
        
        # 4. Schedule cleanup to happen AFTER processing is done
        background_tasks.add_task(cleanup_temporary_file, tmp_file_path)

        return {
            "filename": file.filename,
            "status": "Video uploaded successfully and is now processing in the background."
        }

    except Exception as e:
        background_tasks.add_task(cleanup_temporary_file, tmp_file_path)
        return HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
