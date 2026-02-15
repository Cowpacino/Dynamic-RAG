import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    PROJECT_NAME: str = "Modular RAG Agent"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    USER_AGENT: str = os.getenv("USER_AGENT", "Mozilla/5.0")
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "modular_rag")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")


settings = Settings()
