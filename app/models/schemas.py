from typing import Dict, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Schema for a chat request.
    """

    message: str = Field(..., description="The user's query or message to the agent.")
    thread_id: Optional[str] = Field(
        "default", description="Conversation thread ID for maintaining state."
    )


class ChatResponse(BaseModel):
    """
    Schema for a chat response.
    """

    response: str = Field(..., description="The AI agent's response message.")


class IndexUrlRequest(BaseModel):
    """
    Schema for indexing a specific URL.
    """

    url: str = Field(..., description="The URL of the webpage to parse and index.")


class IndexResponse(BaseModel):
    """
    Schema for indexing operation results.
    """

    status: str = Field(..., description="Success or error status.")
    message: str = Field(..., description="Detailed message about the indexing result.")
    summary: Optional[Dict[str, str]] = Field(
        None, description="Summary of chunks indexed per file/source."
    )
