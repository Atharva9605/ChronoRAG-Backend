from contextlib import contextmanager

import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .config import settings, ROOT

_pool: ConnectionPool | None = None
_neo4j = None
_qdrant = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.pg_dsn,
            min_size=1,
            max_size=settings.concurrency + 4,
            open=True,
        )
    return _pool


@contextmanager
def pg():
    """Yields a dict-row cursor inside a transaction."""
    with pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            yield cur


def neo4j():
    global _neo4j
    if _neo4j is None:
        _neo4j = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _neo4j


def qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    return _qdrant


def init_neo4j() -> None:
    """Apply constraints/indexes. Safe to call on every startup."""
    path = ROOT / "infra" / "neo4j" / "init.cypher"
    statements = [s.strip() for s in path.read_text().split(";") if s.strip()]
    with neo4j().session() as sess:
        for stmt in statements:
            sess.run(stmt)


def init_qdrant() -> None:
    """Create collections if they don't exist."""
    client = qdrant()
    collections = [c.name for c in client.get_collections().collections]
    
    if "naive_chunks" not in collections:
        client.create_collection(
            collection_name="naive_chunks",
            vectors_config=models.VectorParams(
                size=settings.embed_dim,
                distance=models.Distance.COSINE,
            ),
        )
        # We index doc_id for fast filtering during retrieval
        client.create_payload_index(
            collection_name="naive_chunks",
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
    if "events" not in collections:
        client.create_collection(
            collection_name="events",
            vectors_config=models.VectorParams(
                size=settings.embed_dim,
                distance=models.Distance.COSINE,
            ),
        )
        client.create_payload_index(
            collection_name="events",
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def close() -> None:
    global _pool, _neo4j, _qdrant
    if _pool is not None:
        _pool.close()
        _pool = None
    if _neo4j is not None:
        _neo4j.close()
        _neo4j = None
    if _qdrant is not None:
        _qdrant.close()
        _qdrant = None
