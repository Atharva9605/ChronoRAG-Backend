import json
import time
from pgvector.psycopg import Vector
from pydantic import BaseModel, Field

from .. import llm
from ..db import pg
from ..schemas import Citation, PipelineAnswer
from . import graph

SYSTEM_PROMPT = """You are a chronological literary intelligence assistant for educational textual analysis.

You are given a chronological sequence of narrative events extracted from a book.
The events have been deterministically placed in their TRUE story order by a graph engine.

STRICT ANSWERING RULES:
1. Answer the question using ONLY the provided events, point-wise actions, and dialogue.
2. Cite the exact page number for every factual claim, formatted as (p. 2) or (pp. 2-3).
3. If comparing before vs after, explain the chronological sequence clearly referencing the pages.
4. If the provided events do not contain the answer, state that plainly.
5. Provide a direct, concise, and complete prose answer."""


class StructuredAnswer(BaseModel):
    answer: str = Field(description="Direct narrative answer citing specific pages as (p. N)")
    used_event_ids: list[str] = Field(description="IDs of events referenced in this answer")


def search_events(doc_id: str, query: str, k: int = 10) -> list[dict]:
    """Semantic vector search on events table."""
    vec = llm.embed([query])[0]
    with pg() as cur:
        cur.execute(
            """
            SELECT id, event_name, category, chronological_clue, topological_order,
                   location, characters, core_event, antecedent_cause, consequent_effect,
                   source_pages, first_page,
                   1 - (embedding <=> %s) AS similarity
            FROM events
            WHERE doc_id = %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (Vector(vec), doc_id, Vector(vec), k),
        )
        return [dict(r) for r in cur.fetchall()]


def _fetch_by_ids(doc_id: str, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    with pg() as cur:
        cur.execute(
            """
            SELECT id, event_name, category, chronological_clue, topological_order,
                   location, characters, core_event, antecedent_cause, consequent_effect,
                   source_pages, first_page
            FROM events
            WHERE doc_id = %s AND id = ANY(%s)
            """,
            (doc_id, ids),
        )
        return [dict(r) for r in cur.fetchall()]


def answer(doc_id: str, question: str, k: int = 10) -> PipelineAnswer:
    t0 = time.perf_counter()
    before_usage = llm.usage_snapshot()
    trace: list[str] = []

    # 1. Semantic search for candidate seed events
    seeds = search_events(doc_id, question, k=k)
    trace.append(f"Identified {len(seeds)} candidate events by semantic similarity.")

    # 2. Graph expansion (1 hop along temporal DAG)
    seed_ids = [s["id"] for s in seeds]
    expanded_ids = graph.neighbours(seed_ids, hops=1)
    extra = [e for e in _fetch_by_ids(doc_id, expanded_ids) if e["id"] not in set(seed_ids)]
    trace.append(f"Expanded along graph relationships, adding {len(extra)} adjacent timeline events.")

    pool = seeds + extra

    # 3. Deterministic Topological Sorting (The Graph dictates true story order)
    pool.sort(key=lambda e: (e.get("topological_order", 0), e.get("first_page", 0)))
    trace.append("Topologically ordered working set into true story-world chronology.")

    # Format event context with point-wise actions
    context_blocks = []
    for i, e in enumerate(pool, start=1):
        pages_str = f"pp. {e['source_pages'][0]}-{e['source_pages'][-1]}" if len(e.get("source_pages", [])) > 1 else f"p. {e.get('first_page', '?')}"
        
        actions_detail = ""
        if e.get("consequent_effect"):
            try:
                actions_list = json.loads(e["consequent_effect"])
                if isinstance(actions_list, list):
                    actions_detail = "\n   Actions: " + "; ".join(f"[p. {a.get('page_no')}] {a.get('action')}" for a in actions_list)
            except Exception:
                pass

        block = (
            f"[{i}] EVENT: {e['event_name']} ({pages_str})\n"
            f"   Classification: {e.get('location', 'story_progression')} | Time: {e.get('chronological_clue') or 'N/A'}\n"
            f"   Characters: {', '.join(e.get('characters', [])) or 'None'}\n"
            f"   Summary: {e.get('core_event')}{actions_detail}"
        )
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)
    user_prompt = (
        f"EVENTS IN TRUE STORY CHRONOLOGY:\n\n{context_text}\n\n"
        f"QUESTION: {question}"
    )

    result = llm.chat_structured(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        model_cls=StructuredAnswer,
        temperature=0.1,
        max_tokens=1500,
    )
    after_usage = llm.usage_snapshot()

    used_ids = set(result.used_event_ids)
    cited_events = [e for e in pool if e["id"] in used_ids] or pool[:4]

    return PipelineAnswer(
        pipeline="kaalkram",
        answer=result.answer,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        prompt_tokens=after_usage["prompt"] - before_usage["prompt"],
        completion_tokens=after_usage["completion"] - before_usage["completion"],
        citations=[Citation(label=e["event_name"], pages=e.get("source_pages", [e.get("first_page", 1)])) for e in cited_events],
        retrieved=[
            {
                "rank": i + 1,
                "id": e["id"],
                "name": e["event_name"],
                "stage": e.get("chronological_clue", ""),
                "category": e.get("category", "major"),
                "pages": e.get("source_pages", [e.get("first_page", 1)]),
                "preview": e.get("core_event", "")[:280],
            }
            for i, e in enumerate(pool)
        ],
        trace=trace,
    )
