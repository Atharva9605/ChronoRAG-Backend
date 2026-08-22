from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2024-10-21"
    azure_chat_deployment: str = "gpt-4o"
    azure_embed_deployment: str = "text-embedding-3-small"
    embed_dim: int = 1536

    # Stores
    pg_dsn: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    qdrant_url: str
    qdrant_api_key: str | None = None

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

# Canonical story-stage taxonomy. stage_order drives chronological sorting.
# Tuned for Hemingway's The Old Man and the Sea (swap for another book).
TAXONOMY: list[str] = [
    "Shore / Setup",
    "Out to Sea / The Hunt",
    "Struggle with the Marlin",
    "Return / The Sharks",
    "Homecoming / Aftermath",
]
STAGE_ORDER = {name: i for i, name in enumerate(TAXONOMY)}


def stage_index(anchor: str) -> int:
    """Map a free-text anchor onto the taxonomy, tolerating minor LLM drift."""
    if anchor in STAGE_ORDER:
        return STAGE_ORDER[anchor]
    low = anchor.lower()
    for name, idx in STAGE_ORDER.items():
        if name.lower() in low or low in name.lower():
            return idx
    for key, idx in (
        ("shore", 0), ("setup", 0), ("skiff", 0), ("manolin", 0), ("boy", 0),
        ("hunt", 1), ("out to sea", 1), ("bait", 1), ("hook", 1),
        ("marlin", 2), ("struggle", 2), ("fight", 2), ("line", 2),
        ("shark", 3), ("return", 3), ("skeleton", 3), ("carcass", 3),
        ("home", 4), ("aftermath", 4), ("tourist", 4), ("sleep", 4), ("dream", 4),
    ):
        if key in low:
            return idx
    return 1  # safe default: main story body
