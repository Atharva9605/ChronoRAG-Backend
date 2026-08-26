import json
import logging
from collections import defaultdict
from typing import Callable

from .. import llm
from ..config import settings
from ..db import neo4j
from .schemas import (
    GlobalTimelineOrderingResponse,
    MacroEraOrderingResponse,
)

logger = logging.getLogger(__name__)

ORDERING_SYSTEM_PROMPT = """You are an expert narrative chronologist specializing in true timeline reconstruction.

You are given a list of events extracted from a document or book.
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

MACRO_ERA_SYSTEM_PROMPT = """You are a master literary chronologist.
You are given a list of narrative eras and story epochs extracted from a large document.
Your task is to order these ERAS into their TRUE chronological sequence from earliest historical past/backstory to the final narrative resolution.
Assign each era a strict integer `era_rank` (1 = earliest in history/past, N = final resolution).
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


def _format_event_card(e: dict) -> str:
    """Generates a compact, high-signal event card optimized for LLM context limits."""
    actions_str = ""
    if e.get("point_wise_actions"):
        # Select top 2-3 concise actions
        actions_str = " | Actions: " + "; ".join(
            f"[p.{a['page_no']}] {a['action'][:90]}" for a in e["point_wise_actions"][:3]
        )
    
    timestamps_str = ", ".join(e.get("exact_dates_and_timestamps", [])) or "None"
    rel_context = e.get("relative_time_context") or "N/A"
    
    return (
        f"- EVENT ID: {e['id']}\n"
        f"  Name: {e['event_name']}\n"
        f"  Classification: {e.get('classification', 'story_progression')} | Era: {e.get('story_era', 'Present')}\n"
        f"  Pages: pp. {e['page_start']}-{e['page_end']} | Time Anchor: {e.get('temporal_anchor') or 'None'}\n"
        f"  Dates/Timestamps: {timestamps_str} | Context: {rel_context}\n"
        f"  Summary: {e['summary'][:160]}{actions_str}"
    )


def _order_direct_batch(events: list[dict], batch_label: str = "all") -> dict[str, dict]:
    """Orders a single batch (<= 30 events) using structured LLM response."""
    event_blocks = [_format_event_card(e) for e in events]
    user_prompt = (
        f"EVENTS TO ORDER CHRONOLOGICALLY ({len(events)} events in batch '{batch_label}'):\n\n"
        + "\n\n".join(event_blocks)
        + "\n\nArrange all events above into their TRUE story-world chronological order from earliest to latest."
    )

    result: GlobalTimelineOrderingResponse = llm.chat_structured(
        system=ORDERING_SYSTEM_PROMPT,
        user=user_prompt,
        model_cls=GlobalTimelineOrderingResponse,
        temperature=0.1,
        max_tokens=4000,
    )

    rank_map = {}
    for item in result.ordered_events:
        rank_map[item.event_id] = {
            "rank": item.chronological_rank,
            "period": item.story_time_period,
            "era": item.story_era,
            "rationale": item.chronological_rationale,
        }
    return rank_map


def _order_hierarchical(
    doc_id: str,
    events: list[dict],
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, dict]:
    """
    Scalable hierarchical ordering for large documents (100-500+ pages / 50-300+ events):
    1. Clusters events by story_era / narrative epoch.
    2. Orders events within each era locally in parallel/batches.
    3. Orders the eras globally via a macro-ordering prompt.
    4. Stitches all batches into a single globally-ranked chronological timeline.
    """
    if on_progress:
        on_progress(0.85, f"Hierarchical scaling: Grouping {len(events)} events into chronological eras")

    # Group events by era
    era_groups = defaultdict(list)
    for e in events:
        era = e.get("story_era") or "Present Storyline"
        era_groups[era].append(e)

    # 1. Order eras globally
    distinct_eras = list(era_groups.keys())
    era_list_str = "\n".join(
        f"- Era: '{era}' ({len(era_groups[era])} events spanning pages {min(ev['page_start'] for ev in era_groups[era])}-{max(ev['page_end'] for ev in era_groups[era])})"
        for era in distinct_eras
    )
    
    macro_prompt = (
        f"STORY ERAS IDENTIFIED ACROSS DOCUMENT ({len(distinct_eras)} total eras):\n\n"
        + era_list_str
        + "\n\nOrder these eras into their TRUE story-world chronological sequence from earliest historical past to final resolution."
    )

    macro_res: MacroEraOrderingResponse = llm.chat_structured(
        system=MACRO_ERA_SYSTEM_PROMPT,
        user=macro_prompt,
        model_cls=MacroEraOrderingResponse,
        temperature=0.1,
        max_tokens=1500,
    )

    # Sort eras by LLM macro rank
    ordered_era_names = [item.era_name for item in sorted(macro_res.ordered_eras, key=lambda x: x.era_rank)]
    # Append any unranked eras safely
    for era in distinct_eras:
        if era not in ordered_era_names:
            ordered_era_names.append(era)

    # 2. Order events within each era
    global_rank_map = {}
    global_counter = 1

    for era_name in ordered_era_names:
        era_events = era_groups.get(era_name, [])
        if not era_events:
            continue

        if on_progress:
            on_progress(0.88, f"Ordering events in era: {era_name} ({len(era_events)} events)")

        # Sub-batch if single era has more than 25 events
        sub_batch_size = 25
        for i in range(0, len(era_events), sub_batch_size):
            chunk = era_events[i:i + sub_batch_size]
            if len(chunk) == 1:
                global_rank_map[chunk[0]["id"]] = {
                    "rank": global_counter,
                    "period": era_name,
                    "era": era_name,
                    "rationale": f"Sequential progression in era {era_name}.",
                }
                global_counter += 1
            else:
                batch_ranks = _order_direct_batch(chunk, batch_label=f"{era_name}_part_{i//sub_batch_size+1}")
                # Sort chunk by local rank
                sorted_chunk = sorted(chunk, key=lambda ev: batch_ranks.get(ev["id"], {}).get("rank", 999))
                for ev in sorted_chunk:
                    local_info = batch_ranks.get(ev["id"], {})
                    global_rank_map[ev["id"]] = {
                        "rank": global_counter,
                        "period": local_info.get("period", era_name),
                        "era": era_name,
                        "rationale": local_info.get("rationale", f"Order in {era_name}"),
                    }
                    global_counter += 1

    return global_rank_map


def order_timeline_with_llm(
    doc_id: str,
    events: list[dict],
    on_progress: Callable[[float, str], None] | None = None,
) -> tuple[list[dict], list[str]]:
    """
    Feeds extracted events to the LLM to arrange them into true story-world
    chronological sequence. Automatically uses Direct Mode (<=30 events) or
    Hierarchical Scaling Mode (>30 events) to prevent token overflow on large books.
    """
    if not events:
        return [], []

    if len(events) <= 30:
        if on_progress:
            on_progress(0.85, f"Arranging {len(events)} events into true chronological story order")
        rank_map = _order_direct_batch(events, batch_label="global")
    else:
        rank_map = _order_hierarchical(doc_id, events, on_progress=on_progress)

    # Assign ranks
    for i, e in enumerate(events):
        info = rank_map.get(e["id"])
        if info:
            e["topological_order"] = info["rank"]
            e["story_time_period"] = info.get("period", "")
            e["story_era"] = info.get("era") or e.get("story_era", "Present")
            e["chronological_rationale"] = info.get("rationale", "")
        else:
            e["topological_order"] = 1000 + i

    # Sort events strictly by assigned chronological rank
    events.sort(key=lambda x: x["topological_order"])
    topo_order = [e["id"] for e in events]

    # Save LLM ordering checkpoint to data/cache/
    _save_json_checkpoint(
        f"{doc_id}_checkpoint_global_llm_ordering.json",
        {
            "doc_id": doc_id,
            "mode": "direct" if len(events) <= 30 else "hierarchical",
            "total_events": len(events),
            "events_ordered": [
                {
                    "rank": e["topological_order"],
                    "id": e["id"],
                    "name": e["event_name"],
                    "era": e.get("story_era", ""),
                    "pages": f"{e['page_start']}-{e['page_end']}",
                    "rationale": e.get("chronological_rationale", ""),
                }
                for e in events
            ],
        },
    )

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
                    "story_era": n.get("story_era", "Present"),
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
