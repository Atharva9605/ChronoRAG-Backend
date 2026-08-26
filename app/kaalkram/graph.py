import json
import logging
from typing import Callable

from .. import llm
from ..config import settings
from ..db import neo4j
from .schemas import GlobalTimelineOrderingResponse

logger = logging.getLogger(__name__)

ORDERING_SYSTEM_PROMPT = """You are an expert narrative chronologist specializing in true timeline reconstruction.

You are given a comprehensive list of all events extracted from a document or book.
Your task is to organize these events into their TRUE CHRONOLOGICAL ORDER (Fabula), representing the actual historical/narrative time they occurred, not merely the order they were printed on the page.

CRITICAL CHRONOLOGY RULES:
1. TRUE STORY-TIME VS DOCUMENT PRINT ORDER:
   - If an event is a flashback, historical memory, origin backstory, or prior life milestone that occurred in the past, it MUST be placed in its true past position BEFORE the primary real-time storyline begins.
   - The primary real-time narrative must flow forward in strict sequential chronological order.
2. STRICT UNIQUE RANKS:
   - Assign every single event an exact integer `chronological_rank` starting at 1, 2, 3, ... up to N.
   - Every event ID provided in the input MUST be included in the output.
3. RATIONALE:
   - For every event, provide a 1-sentence `chronological_rationale` explaining why it occurs at this position in true chronology.
"""


def _save_json_checkpoint(filename: str, data: dict) -> None:
    """Helper to save inspection checkpoints to cache directory."""
    try:
        cache_dir = settings.cache_path
        cache_dir.mkdir(parents=True, exist_ok=True)
        target_file = cache_dir / filename
        target_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to write checkpoint {filename}: {e}")


def order_timeline_with_llm(
    doc_id: str,
    events: list[dict],
    on_progress: Callable[[float, str], None] | None = None,
) -> tuple[list[dict], list[str]]:
    """
    Feeds the full list of extracted events to the LLM to arrange them into
    true story-world chronological sequence, replacing heuristic DAG logic.
    """
    if not events:
        return [], []

    if on_progress:
        on_progress(0.85, "LLM arranging events into true chronological story order")

    # Format all events for the LLM
    event_blocks = []
    for e in events:
        actions_str = ""
        if e.get("point_wise_actions"):
            actions_str = " | Actions: " + "; ".join(
                f"[p. {a['page_no']}] {a['action']}" for a in e["point_wise_actions"]
            )
        
        timestamps_str = ", ".join(e.get("exact_dates_and_timestamps", [])) or "None"
        rel_context = e.get("relative_time_context") or "N/A"
        
        block = (
            f"- EVENT ID: {e['id']}\n"
            f"  Name: {e['event_name']}\n"
            f"  Classification: {e.get('classification', 'story_progression')}\n"
            f"  Era / Period: {e.get('story_era', 'Present')}\n"
            f"  Page Range: pp. {e['page_start']}-{e['page_end']}\n"
            f"  Time Anchor: {e.get('temporal_anchor') or 'Not specified'}\n"
            f"  Dates / Timestamps: {timestamps_str}\n"
            f"  Relative Chronology: {rel_context}\n"
            f"  Summary: {e['summary']}{actions_str}"
        )
        event_blocks.append(block)

    user_prompt = (
        f"EXTRACTED EVENTS TO ORDER CHRONOLOGICALLY ({len(events)} total events):\n\n"
        + "\n\n".join(event_blocks)
        + "\n\nArrange all events above into their TRUE story-world chronological order from earliest past backstories/memories to the final resolution."
    )

    # Call LLM for global timeline ordering
    ordering_result: GlobalTimelineOrderingResponse = llm.chat_structured(
        system=ORDERING_SYSTEM_PROMPT,
        user=user_prompt,
        model_cls=GlobalTimelineOrderingResponse,
        temperature=0.1,
        max_tokens=4000,
    )

    # Save LLM ordering checkpoint to data/cache/
    _save_json_checkpoint(
        f"{doc_id}_checkpoint_global_llm_ordering.json",
        ordering_result.model_dump(),
    )

    # Map the LLM's chronological rank back to the events
    rank_map = {}
    for item in ordering_result.ordered_events:
        rank_map[item.event_id] = {
            "rank": item.chronological_rank,
            "period": item.story_time_period,
            "era": item.story_era,
            "rationale": item.chronological_rationale,
        }

    id_to_event = {e["id"]: e for e in events}
    
    # Assign ranks (fall back to narrative order if missing)
    for i, e in enumerate(events):
        info = rank_map.get(e["id"])
        if info:
            e["topological_order"] = info["rank"]
            e["story_time_period"] = info["period"]
            e["story_era"] = info.get("era") or e.get("story_era", "Present")
            e["chronological_rationale"] = info["rationale"]
        else:
            e["topological_order"] = 1000 + i

    # Sort events strictly by LLM's chronological rank
    events.sort(key=lambda x: x["topological_order"])
    topo_order = [e["id"] for e in events]

    # Construct single sequential forward timeline edges: (Rank 1) -> (Rank 2) -> ... -> (Rank N)
    edges: list[dict] = []
    for i in range(len(events) - 1):
        edges.append({
            "src": events[i]["id"],
            "dst": events[i + 1]["id"],
            "rel": "HAPPENS_BEFORE",
        })

    return edges, topo_order


def push_to_neo4j(doc_id: str, events: list[dict], edges: list[dict]) -> None:
    """Pushes nodes and relationships into Neo4j graph database."""
    with neo4j().session() as session:
        # Clear existing nodes for document
        session.run("MATCH (n:Event {doc_id: $d}) DETACH DELETE n", d=doc_id)

        # Batch create Event nodes
        node_payload = [
            {
                "id": e["id"],
                "doc_id": doc_id,
                "name": e["event_name"],
                "summary": e["summary"],
                "classification": e.get("classification", "story_progression"),
                "story_era": e.get("story_era", "Present"),
                "page_start": e["page_start"],
                "page_end": e["page_end"],
                "source_pages": e.get("page_numbers", []),
                "characters": e.get("characters", []),
                "temporal_anchor": e.get("temporal_anchor", ""),
                "topological_order": e.get("topological_order", 0),
            }
            for e in events
        ]

        session.run(
            """
            UNWIND $nodes AS n
            CREATE (e:Event {
                id: n.id,
                doc_id: n.doc_id,
                name: n.name,
                summary: n.summary,
                classification: n.classification,
                story_era: n.story_era,
                page_start: n.page_start,
                page_end: n.page_end,
                source_pages: n.source_pages,
                characters: n.characters,
                temporal_anchor: n.temporal_anchor,
                topological_order: n.topological_order
            })
            """,
            nodes=node_payload,
        )

        # Batch create single directional forward edges
        if edges:
            session.run(
                """
                UNWIND $edges AS r
                MATCH (a:Event {id: r.src}), (b:Event {id: r.dst})
                MERGE (a)-[rel:HAPPENS_BEFORE]->(b)
                SET rel.relation_type = r.rel
                """,
                edges=edges,
            )


def fetch_graph(doc_id: str) -> dict:
    """Retrieves full graph representation for UI visualization."""
    with neo4j().session() as session:
        result = session.run(
            """
            MATCH (e:Event {doc_id: $d})
            OPTIONAL MATCH (e)-[r:HAPPENS_BEFORE]->(target:Event {doc_id: $d})
            RETURN e, r, target
            ORDER BY e.topological_order, e.page_start
            """,
            d=doc_id,
        )
        
        nodes_dict = {}
        links = []
        for record in result:
            n = record["e"]
            if n["id"] not in nodes_dict:
                nodes_dict[n["id"]] = {
                    "id": n["id"],
                    "name": n.get("name", "Event"),
                    "summary": n.get("summary", ""),
                    "classification": n.get("classification", "story_progression"),
                    "pages": n.get("source_pages", []),
                    "page_start": n.get("page_start", 0),
                    "page_end": n.get("page_end", 0),
                    "topological_order": n.get("topological_order", 0),
                    "temporal_anchor": n.get("temporal_anchor", ""),
                }
            r = record["r"]
            t = record["target"]
            if r and t:
                links.append({
                    "source": n["id"],
                    "target": t["id"],
                    "relation": r.get("relation_type", "HAPPENS_BEFORE"),
                })

        nodes = []
        for n in nodes_dict.values():
            category = "major" if n.get("classification") == "story_progression" else "minor"
            nodes.append({
                "id": n["id"],
                "name": n.get("name", "Event"),
                "category": category,
                "anchor": n.get("temporal_anchor", ""),
                "stage_order": n.get("topological_order", 0),
                "first_page": n.get("page_start", 0),
                "pages": n.get("pages", []),
                "core": n.get("summary", ""),
            })

        edges = []
        for r in links:
            edges.append({
                "src": r["source"],
                "dst": r["target"],
                "confidence": 1.0,
                "kind": "causal" if r.get("relation") == "CAUSES" else "temporal",
            })

        nodes.sort(key=lambda x: x["stage_order"])
        return {"nodes": nodes, "edges": edges, "links": links}


def neighbours(seed_ids: list[str], hops: int = 1) -> list[str]:
    """Expands seed nodes along graph relationships."""
    if not seed_ids:
        return []
    with neo4j().session() as session:
        result = session.run(
            """
            MATCH (s:Event) WHERE s.id IN $seeds
            MATCH path = (s)-[*1..%d]-(n:Event)
            RETURN DISTINCT n.id AS id
            """ % hops,
            seeds=seed_ids,
        )
        return [r["id"] for r in result]
