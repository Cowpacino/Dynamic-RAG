from typing import List,Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import torch
import logging


from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages the ChromaDB vector store lifecycle and operations.
    """

    def __init__(self,is_api=True,is_video_processing=False):
        if is_api and not is_video_processing:
        # Initialize embeddings with the API key from settings
            self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
            # Initialize ChromaDB instance
            self.vector_store = Chroma(
                collection_name=settings.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=settings.CHROMA_DB_DIR,
            )
        else:
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"CUDA version (torch): {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            if torch.cuda.is_available():
                logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("CUDA not available - reason unknown, check below")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device} for embeddings")
            
            logger.info("Initializing OpenCLIP Embedding Function...")
            self.embeddings = OpenCLIPEmbeddingFunction(device=device)

            # We need an ImageLoader so Chroma knows how to read the physical files
            logger.info("Initializing Image Loader...")
            image_loader = ImageLoader()

            # Create the database connection
            logger.info("Connecting to ChromaDB persistent client...")
            client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)

            logger.info("Getting or creating Chroma collection 'pure_visual_frames'...")
            self.vector_store = client.get_or_create_collection(
                name="pure_visual_frames",
                embedding_function=self.embeddings,
                data_loader=image_loader
            )
        

    def add_documents(self, documents: Optional[List[Document]],extracted_metadata:Optional[dict]=None,is_video_processing=False):
        """
        Add a list of LangChain Document objects to the vector store.
        """
        if is_video_processing:
            
            # Add your frames (using the list we made in the previous step)
            ids = [f"frame_{i}" for i in range(len(extracted_metadata))]
            paths = [item['path'] for item in extracted_metadata]
            metadatas = [{"timestamp": item['timestamp']} for item in extracted_metadata]

            # Notice we use 'uris' instead of 'documents'. 
            logger.info(f"Adding {len(ids)} frames to the Chroma collection...")
            # Chroma will open the images, pass them through the local Vision model, and save the vectors!
            self.vector_store.add(
                ids=ids,
                uris=paths,
                metadatas=metadatas
            )
            
        else:
            if not documents:
                return
            self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 2) -> List[Document]:
        """
        Perform a similarity search against the vector store.
        """
        if isinstance(self.vector_store,Chroma):
            # LangChain wrapper — used for text/PDF RAG
            result = self.vector_store.similarity_search(query, k=k)

        else:
            # Raw ChromaDB collection — used for video frames
            result = self.vector_store.query(
                query_texts=[query],
                n_results=k,
                include=["uris", "metadatas"]
            )
        logger.info(result)
        return result

    def get_retriever(self, search_kwargs: dict = None):
        """
        Return a retriever interface for LangChain tools/chains.
        """
        if search_kwargs is None:
            search_kwargs = {"k": 2}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

# Singleton for text/PDF RAG (OpenAI text embeddings)
vector_store_manager = VectorStoreManager()

# Singleton for video RAG (OpenCLIP visual embeddings)
video_store_manager = VectorStoreManager(is_api=False, is_video_processing=True)