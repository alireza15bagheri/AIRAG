"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # App
    app_name: str = "AIRAG"
    debug: bool = True
    log_level: str = "INFO"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Document processing
    allowed_extensions: list[str] = ["pdf", "docx", "md", "txt"]
    max_file_size_mb: int = 50
    upload_dir: str = "./data/uploads"
    
    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunk_strategy: str = "sentence_aware"
    
    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    
    # Vector DB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "documents"
    chroma_distance_function: str = "cosine"
    
    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.7
    
    # LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_temperature: float = 0.1
    
    # Generation
    max_context_tokens: int = 4000
    system_prompt: str = (
        "You are a helpful assistant that answers questions based only on "
        "the provided context. If the context doesn't contain enough "
        "information, say so."
    )
    enable_streaming: bool = True


# Singleton instance - import this everywhere
settings = Settings()
