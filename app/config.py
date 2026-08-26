from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Universal Model Configuration (LiteLLM Gateway & Any Provider)
    llm_model: str = ""
    embed_model: str = ""
    embed_dim: int = 1536

    # Azure OpenAI (Legacy / Standard)
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_chat_deployment: str = "gpt-4o"
    azure_embed_deployment: str = "text-embedding-3-small"

    # Direct Provider Keys / LiteLLM Proxy
    openai_api_key: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    litellm_api_base: str = ""
    litellm_api_key: str = ""

    # Stores
    pg_dsn: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Pipeline
    concurrency: int = 5
    window_size: int = 10
    window_overlap: int = 2
    naive_chunk_chars: int = 1200
    naive_chunk_overlap: int = 200
    naive_top_k: int = 8
    pass2_batch_size: int = 2

    # Paths
    upload_dir: str = "./data/uploads"
    cache_dir: str = "./data/cache"

    @property
    def upload_path(self) -> Path:
        p = (ROOT / self.upload_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_path(self) -> Path:
        p = (ROOT / self.cache_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
