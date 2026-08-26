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

# Fallback only — used if Pass 0 has not produced a per-document taxonomy yet.
# Prefer document.taxonomy from the DB for any real build.
DEFAULT_TAXONOMY: list[str] = [
    "Beginning / Setup",
    "Rising Action",
    "Midpoint / Complication",
    "Climax",
    "Resolution / Aftermath",
]

# Back-compat alias for older imports / health endpoints.
TAXONOMY = DEFAULT_TAXONOMY


def stage_names(taxonomy: list) -> list[str]:
    """Normalize taxonomy records (strings or {name,...} dicts) to stage names."""
    names: list[str] = []
    for item in taxonomy or []:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(item.get("name") or "").strip()
        else:
            name = str(getattr(item, "name", "") or "").strip()
        if name:
            names.append(name)
    return names or list(DEFAULT_TAXONOMY)


def stage_index(anchor: str, taxonomy: list | None = None) -> int:
    """Map a free-text anchor onto the given taxonomy, tolerating minor LLM drift."""
    names = stage_names(taxonomy if taxonomy is not None else DEFAULT_TAXONOMY)
    order = {name: i for i, name in enumerate(names)}
    if anchor in order:
        return order[anchor]
    low = (anchor or "").lower().strip()
    if not low:
        return min(1, len(names) - 1)
    for name, idx in order.items():
        nl = name.lower()
        if nl in low or low in nl:
            return idx
    # Token overlap fallback (no book-specific keyword table).
    tokens = {t for t in low.replace("/", " ").replace("-", " ").split() if len(t) > 2}
    best_idx, best_score = 0, 0
    for name, idx in order.items():
        ntokens = {t for t in name.lower().replace("/", " ").replace("-", " ").split() if len(t) > 2}
        score = len(tokens & ntokens)
        if score > best_score:
            best_idx, best_score = idx, score
    if best_score > 0:
        return best_idx
    return min(1, len(names) - 1)
