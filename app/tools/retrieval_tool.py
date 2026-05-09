from langchain_core.tools import tool
from langsmith import traceable
from app.core.vector_store import vector_store_manager
import logging 



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
@traceable(name="Retrieval_tool")
def retrieve_from_vector_store(query: str):
    """
    Search for relevant information in the internal knowledge base (PDFs and indexed web pages).
    Use this tool when the user asks questions about uploaded documents or previously stored information.
    """
    try:
        # Perform similarity search using the manager
        retrieved_docs = vector_store_manager.similarity_search(query, k=2)

        if not retrieved_docs:
            message_content = f"No relevant information found in the local knowledge base for: '{query}'."
            return (message_content, [])
        
        logger.info(f"Retrieved Docs: {retrieved_docs}")
        
        # Format the documents for the LLM
        formatted_documents = []
        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "Unknown source")
            doc_type = doc.metadata.get("type", "unknown")
            formatted_doc = f"--- Document {i} (Source: {source}, Type: {doc_type}) ---\n{doc.page_content}"
            formatted_documents.append(formatted_doc)

        message_content = "\n\n".join(formatted_documents)
    
        # Return tuple: (message_content for LLM, raw documents as artifact)
        return (message_content, retrieved_docs)
    except Exception as e:
        error_message = f"Error retrieving context from vector store: {str(e)}"
        return (error_message, [])
