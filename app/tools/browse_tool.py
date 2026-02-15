import asyncio
from typing import List, Tuple

from langchain_core.tools import tool

from app.core.vector_store import vector_store_manager
from app.services.search_service import search_service


@tool(response_format="content_and_artifact")
def browse_webpage(url: str, index_for_later: bool = True):
    """
    Accesses a specific webpage URL, parses its content automatically, and extracts relevant information.
    Use this when you have a specific URL and need to know what's on that page.

    Args:
        url: The full URL of the webpage to browse.
        index_for_later: If True, the content will be saved to the internal knowledge base for future queries.
    """
    try:
        # Create an event loop to run the async service method in a sync tool context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Fetch and parse the webpage content
        docs = loop.run_until_complete(search_service.fetch_and_parse_webpage(url))

        if not docs:
            loop.close()
            return (
                f"Failed to extract content from {url}. The page might be protected or empty.",
                [],
            )

        # Optionally index the documents into our vector store
        if index_for_later:
            vector_store_manager.add_documents(docs)
            indexing_status = (
                " (This content has also been indexed to your local knowledge base.)"
            )
        else:
            indexing_status = ""

        # Format the content for the LLM to read
        page_content = "\n\n".join([doc.page_content for doc in docs])

        # Limit content length to avoid context window overflow in the initial response
        # The agent can always ask for more or we can rely on the full artifact
        truncated_content = (
            (page_content[:8000] + "...") if len(page_content) > 8000 else page_content
        )

        message_content = (
            f"Successfully browsed: {url}{indexing_status}\n\n"
            f"--- Page Content ---\n"
            f"{truncated_content}"
        )

        loop.close()
        return (message_content, docs)

    except Exception as e:
        return (f"Error browsing {url}: {str(e)}", [])
