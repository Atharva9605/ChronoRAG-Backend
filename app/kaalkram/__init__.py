import json
import logging
from typing import Callable

from ..config import settings
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
    5. JSON checkpoints saved to data/cache/.
    """
    # 1. Extraction with Rolling Memory and Window Checkpoints
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
    
    # Save graph & topological order checkpoint
    try:
        cache_dir = settings.cache_path
        cache_dir.mkdir(parents=True, exist_ok=True)
        graph_checkpoint_file = cache_dir / f"{doc_id}_checkpoint_graph_and_topology.json"
        graph_payload = {
            "doc_id": doc_id,
            "total_events": len(events),
            "total_edges": len(edges),
            "topological_sequence": topo_order,
            "edges": edges,
            "events_ordered": sorted(events, key=lambda x: x["topological_order"]),
        }
        graph_checkpoint_file.write_text(
            json.dumps(graph_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning(f"Failed to write graph checkpoint: {exc}")
    
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


def get_events(doc_id: str) -> list[dict]:
    """Retrieve all chronological events for document."""
    from ..db import pg
    with pg() as cur:
        cur.execute(
            """
            SELECT id, event_name, category, chronological_clue AS timeline_anchor,
                   topological_order AS stage_order, location, characters,
                   core_event, antecedent_cause, consequent_effect,
                   source_pages, first_page, merge_count
            FROM events
            WHERE doc_id = %s
            ORDER BY topological_order, first_page
            """,
            (doc_id,),
        )
        return [dict(r) for r in cur.fetchall()]
