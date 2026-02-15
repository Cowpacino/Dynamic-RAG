import os
import tempfile
from typing import List

import pymupdf4llm
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.core.vector_store import vector_store_manager


class PDFService:
    """
    Service for processing PDF files and indexing them in the vector store.
    It uses pymupdf4llm to extract Markdown content, which preserves document structure.
    """

    def __init__(self):
        # Define headers to split on for better contextual chunking
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        # Secondary splitter to ensure chunks are within LLM context window limits
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    async def process_pdf_content(self, content: bytes, filename: str) -> int:
        """
        Processes PDF content:
        1. Saves bytes to a temporary file.
        2. Extracts Markdown text using pymupdf4llm.
        3. Splits text by headers and then by character count.
        4. Adds resulting documents to the vector store.
        """
        # Create a temporary file because pymupdf4llm requires a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # 1. Extract text as Markdown
            md_output = pymupdf4llm.to_markdown(tmp_path)

            # Ensure we have a string (pymupdf4llm can return list of dicts)
            if isinstance(md_output, list):
                md_text = "\n\n".join(
                    [
                        str(page.get("text", ""))
                        for page in md_output
                        if isinstance(page, dict)
                    ]
                )
            else:
                md_text = str(md_output) if md_output else ""

            if not md_text.strip():
                raise ValueError("No text could be extracted from the PDF.")

            # 2. Structural split based on Markdown headers
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.headers_to_split_on
            )
            header_splits = header_splitter.split_text(md_text)

            # 3. Recursive split into manageable chunks
            final_docs = self.text_splitter.split_documents(header_splits)

            # Enrich metadata
            for doc in final_docs:
                doc.metadata["source"] = filename
                doc.metadata["type"] = "pdf"

            # 4. Index in vector store
            vector_store_manager.add_documents(final_docs)

            return len(final_docs)
        finally:
            # Ensure the temporary file is deleted
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def upload_and_index_pdfs(self, files_to_process: List[dict]) -> dict:
        """
        Process multiple uploaded files.
        files_to_process: List of dictionaries with "content" (bytes) and "filename" (str).
        """
        summary = {}
        for item in files_to_process:
            content = item["content"]
            filename = item["filename"]
            try:
                num_chunks = await self.process_pdf_content(content, filename)
                summary[filename] = f"Successfully indexed {num_chunks} chunks."
            except Exception as e:
                summary[filename] = f"Error processing file: {str(e)}"

        return summary


# Global instance
pdf_service = PDFService()
