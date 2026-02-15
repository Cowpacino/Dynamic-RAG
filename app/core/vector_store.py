from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


class VectorStoreManager:
    """
    Manages the ChromaDB vector store lifecycle and operations.
    """

    def __init__(self):
        # Initialize embeddings with the API key from settings
        self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

        # Initialize ChromaDB instance
        self.vector_store = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_DB_DIR,
        )

    def add_documents(self, documents: List[Document]):
        """
        Add a list of LangChain Document objects to the vector store.
        """
        if not documents:
            return
        self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform a similarity search against the vector store.
        """
        return self.vector_store.similarity_search(query, k=k)

    def get_retriever(self, search_kwargs: dict = None):
        """
        Return a retriever interface for LangChain tools/chains.
        """
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)


# Singleton instance to be used across the application
vector_store_manager = VectorStoreManager()
