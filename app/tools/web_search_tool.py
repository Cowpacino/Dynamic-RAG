import asyncio

from langchain_core.tools import tool

from app.services.search_service import search_service


@tool(response_format="content_and_artifact")
def search_the_internet(query: str):
    """
    Search the internet for real-time information, news, or general knowledge.
    Use this tool when the internal knowledge base does not have the answer or when
    the user explicitly asks for information from the web.
    """
    try:
        # Since we are in a synchronous tool definition but calling an async service,
        # we use asyncio.run to execute the search.
        # In a fully async LangGraph/LangChain setup, you would use 'async def'.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Get snippets from search results
        search_results_docs = loop.run_until_complete(
            search_service.get_web_context(query)
        )
        loop.close()

        if not search_results_docs:
            return (f"No search results found for: '{query}'.", [])

        # Format the snippets for the LLM
        formatted_results = []
        for i, doc in enumerate(search_results_docs, start=1):
            source = doc.metadata.get("source", "Search Engine")
            content = doc.page_content
            formatted_results.append(
                f"--- Search Result {i} (Source: {source}) ---\n{content}"
            )

        message_content = "\n\n".join(formatted_results)

        # Add a note that this is live data
        header = f"Live Search Results for: {query}\n\n"

        return (header + message_content, search_results_docs)
    except Exception as e:
        error_message = f"Error performing web search: {str(e)}"
        return (error_message, [])
