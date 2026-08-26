import logging
from typing import Callable

from . import engine, extractor, graph
from .schemas import ExtractedEvent, WindowExtractionResult, RollingMemory

logger = logging.getLogger(__name__)


def build_timeline(
    doc_id: str,
    pages: list[str],
    on_progress: Callable[[float, str], None] | None = None,
) -> dict:
    """
    Main entrypoint to run the Kaalkram 2.0 pipeline:
    1. 10-page sliding window with 2-page overlap + rolling memory.
    2. Zero-cost causal & temporal graph DAG construction.
    3. Topological ordering via Kahn's algorithm.
    4. PostgreSQL (pgvector) + Neo4j persistence.
    """
    # 1. Extraction with Rolling Memory
    events = extractor.extract_timeline_events(doc_id, pages, on_progress=on_progress)
    
    if on_progress:
        on_progress(0.85, "Constructing causal timeline graph and topological order")
        
    # 2. Graph Construction & Topological Sorting
    edges, topo_order = graph.build_timeline_graph(events)
    
    # Map topological order integer back to events
    order_map = {node_id: rank for rank, node_id in enumerate(topo_order)}
    for e in events:
        e["topological_order"] = order_map.get(e["id"], 0)
        
    if on_progress:
        on_progress(0.92, "Persisting timeline to PostgreSQL and Neo4j")
        
    # 3. Persistence
    extractor.persist_events_to_postgres(doc_id, events, on_progress=on_progress)
    graph.push_to_neo4j(doc_id, events, edges)
    
    story_count = sum(1 for e in events if e.get("classification") == "story_progression")
    flashback_count = len(events) - story_count
    
    return {
        "events": len(events),
        "edges": len(edges),
        "story_progression": story_count,
        "flashbacks_or_backstory": flashback_count,
    }


def answer_query(doc_id: str, question: str):
    """Timeline-aware retrieval and QA."""
    return engine.answer(doc_id, question)


def get_graph(doc_id: str):
    """Retrieve full event graph for UI."""
    return graph.fetch_graph(doc_id)
