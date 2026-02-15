import os
from typing import List, Optional

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


class SearchService:
    """
    Service for searching the internet, fetching web content, and parsing it.
    It can be used to dynamically expand the RAG system's knowledge base.
    """

    def __init__(self):
        self.search_tool = DuckDuckGoSearchRun()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        # Strainer to filter common clutter from webpages
        self.bs4_strainer = bs4.SoupStrainer(
            ["article", "main", "div", "h1", "h2", "h3", "p"],
            class_=["post-content", "content", "main-content", "entry-content"],
        )

    async def search_internet(self, query: str) -> str:
        """
        Perform a web search using DuckDuckGo.
        Returns a summary/snippet of the search results.
        """
        try:
            return self.search_tool.run(query)
        except Exception as e:
            return f"Search failed: {str(e)}"

    async def fetch_and_parse_webpage(self, url: str) -> List[Document]:
        """
        Fetches the content of a specific URL and parses it into LangChain Documents.
        """
        try:
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs={"parse_only": self.bs4_strainer},
                requests_kwargs={"headers": {"User-Agent": settings.USER_AGENT}},
            )

            docs = loader.load()

            # Fallback if the strainer was too aggressive and returned nothing
            has_content = any(
                doc.page_content and doc.page_content.strip() for doc in docs
            )
            if not has_content:
                loader = WebBaseLoader(
                    web_paths=(url,),
                    requests_kwargs={"headers": {"User-Agent": settings.USER_AGENT}},
                )
                docs = loader.load()

            # Split content into manageable chunks
            split_docs = self.text_splitter.split_documents(docs)

            # Add metadata
            for doc in split_docs:
                doc.metadata["source"] = url
                doc.metadata["type"] = "web"

            return split_docs
        except Exception as e:
            print(f"Error fetching webpage {url}: {str(e)}")
            return []

    async def get_web_context(self, query: str, max_results: int = 3) -> List[Document]:
        """
        A high-level method that searches the web and then fetches the top results
        to provide real-time context.
        """
        # In a production scenario, you might use a search API that returns URLs directly
        # For this modular example, we'll focus on the pattern of search -> fetch
        search_results = await self.search_internet(query)

        # Note: DuckDuckGoSearchRun returns a string.
        # For more complex agents, we'd use DuckDuckGoSearchResults to get specific URLs.
        # Here we return the search summary as a document.
        return [
            Document(
                page_content=search_results,
                metadata={"source": "duckduckgo", "query": query},
            )
        ]


# Global instance
search_service = SearchService()
