from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "models/text-embedding-004"

    # Offline smoke mode (hashing embeddings + extractive answers). Use only for local demos.
    demo_mode: bool = False

    chunk_size: int = 800
    chunk_overlap: int = 150

    top_k: int = 3
    similarity_threshold: float = 0.15

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    data_dir: Path = BASE_DIR / "data"
    docs_dir: Path = BASE_DIR / "data" / "sample_docs"
    uploads_dir: Path = BASE_DIR / "data" / "uploads"
    vectorstore_dir: Path = BASE_DIR / "data" / "vectorstore"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    return settings
