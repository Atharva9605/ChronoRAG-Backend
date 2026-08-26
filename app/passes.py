import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from pgvector.psycopg import Vector

from . import llm
from .config import settings
from .db import pg
from .ingest import parse_pages_from_bullet
from .schemas import BatchExtractResponse

# ============================================================
# PASS 1 — parallel skim
# ============================================================
PASS1_SYSTEM = """You are a rapid-scanning literary agent. Skim the text slice provided.
The text contains inline [PAGE N] markers showing exactly which PDF page each passage comes from.

Write a concise, chronological bulleted list of all plot events, character actions, background
details, subplots, and key interactions. Capture raw observations only - no analysis, no summary
of the whole slice.

IMPORTANT: For every bullet point, record the [PAGE N] number(s) where that event occurs.
Format each bullet exactly as: [PAGE N] <observation>
If an event spans multiple pages: [PAGE N, M] <observation>
Do not output anything except the bullet list."""


def _cache_file(doc_id: str, name: str):
    return settings.cache_path / f"{doc_id}_{name}.json"


def _load_cache(doc_id: str, name: str) -> dict:
    f = _cache_file(doc_id, name)
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(doc_id: str, name: str, data: dict) -> None:
    _cache_file(doc_id, name).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_pass1(doc_id: str, windows: list[dict], on_progress=None) -> list[dict]:
    """Skim every window concurrently. Checkpointed and resumable."""
    cache = _load_cache(doc_id, "pass1")
    cache_lock = threading.Lock()
    pending = [w for w in windows if w["id"] not in cache]
    done = len(windows) - len(pending)

    if on_progress and done:
        on_progress(done / len(windows), f"resumed: {done}/{len(windows)} windows cached")

    def work(w: dict) -> tuple[str, str]:
        user = f"[Text Window: Pages {w['start']} to {w['end']}]\n\n{w['text']}"
        return w["id"], llm.chat(PASS1_SYSTEM, user, max_tokens=1600)

    if pending:
        with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
            futures = {pool.submit(work, w): w for w in pending}
            for fut in as_completed(futures):
                w = futures[fut]
                wid, text = fut.result()
                with cache_lock:
                    cache[wid] = {"start": w["start"], "end": w["end"], "content": text}
                    _save_cache(doc_id, "pass1", cache)
                    done += 1
                if on_progress:
                    on_progress(done / len(windows),
                                f"pass 1: {done}/{len(windows)} windows read")

    # Persist observations for the audit panel
    with pg() as cur:
        cur.execute("DELETE FROM observations WHERE doc_id = %s", (doc_id,))
        cur.executemany(
            """INSERT INTO observations (doc_id, window_id, page_start, page_end, raw_text)
               VALUES (%s,%s,%s,%s,%s)""",
            [(doc_id, w["id"], cache[w["id"]]["start"], cache[w["id"]]["end"],
              cache[w["id"]]["content"]) for w in windows if w["id"] in cache],
        )

    return [
        {"window_id": w["id"], "start": w["start"], "end": w["end"],
         "content": cache[w["id"]]["content"]}
        for w in windows if w["id"] in cache
    ]


# ============================================================
# PASS 2 — stateful merge and deduplication
# ============================================================
PASS2_SYSTEM = """You are a structural data engineer managing a narrative timeline database.
You receive a set of raw chronological observations from a portion of a book.
Extract distinct events and output a list of them.

CHRONOLOGICAL CLUE:
If the text provides explicit or relative time markers (e.g., '1922', 'the next morning', 'after the war', 'later that day'), capture them in chronological_clue. Leave empty if none.

CATEGORY:
- "major": core structural milestones that drive the main plot.
- "minor": subplots, backstory reveals, secondary interactions, atmosphere.

CAUSALITY: antecedent_cause = what triggered this event. consequent_effect = what it leads to.
Both must be grounded in the text, not invented.

SOURCE PAGES: copy the [PAGE N] numbers from the observation bullet verbatim."""

def run_pass2(doc_id: str, observations: list[dict], on_progress=None) -> list[dict]:
    """
    Concurrent extraction, sequential deduplication via vector similarity.
    """
    bs = settings.pass2_batch_size
    batches = [
        "\n\n".join(
            f"--- Observations (pages {o['start']}-{o['end']}) ---\n{o['content']}"
            for o in observations[i:i + bs]
        )
        for i in range(0, len(observations), bs)
    ]
    total = len(batches)
    events: list[dict] = []
    stats = {"extracted": 0, "added": 0, "merged": 0, "failed": 0}

    def fetch(idx: int, text: str):
        user = f"NEW OBSERVATIONS TO EXTRACT:\n{text}"
        try:
            return idx, llm.chat_structured(PASS2_SYSTEM, user, BatchExtractResponse)
        except Exception:
            return idx, None

    # Step 1: Concurrent extraction
    results: dict[int, BatchExtractResponse | None] = {}
    with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
        futures = [pool.submit(fetch, i, text) for i, text in enumerate(batches)]
        for fut in as_completed(futures):
            i, parsed = fut.result()
            results[i] = parsed
            if on_progress:
                on_progress(len(results) / (total * 2), f"extracting events: batch {len(results)}/{total}")

    # Step 2: Sequential deduplication via embeddings
    # To avoid embedding one-by-one which is slow, we gather all extracted events, embed them in bulk, then merge.
    all_extracted = []
    for i in range(total):
        parsed = results.get(i)
        if parsed is None:
            stats["failed"] += 1
        else:
            all_extracted.extend(parsed.events)
    
    stats["extracted"] = len(all_extracted)
    
    if not all_extracted:
        return []

    # Bulk embed all extracted events
    texts_to_embed = [f"{e.event_name} {e.core_event}" for e in all_extracted]
    embeddings = []
    for i in range(0, len(texts_to_embed), 64):
        embeddings.extend(llm.embed(texts_to_embed[i:i + 64]))
        if on_progress:
            on_progress(0.5 + (i / len(texts_to_embed)) * 0.2, "embedding extracted events...")

    # Sequential merge
    import numpy as np
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for idx, e in enumerate(all_extracted):
        emb = embeddings[idx]
        pages = sorted(set(int(p) for p in e.source_pages if p and p > 0))
        
        # Find best match in currently accepted events
        best_match_idx = -1
        best_sim = 0.0
        for e_idx, existing in enumerate(events):
            sim = cosine_sim(emb, existing['embedding'])
            if sim > best_sim:
                best_sim = sim
                best_match_idx = e_idx
        
        if best_sim > 0.98:  # Very high threshold because ada-002 clusters tightly
            target = events[best_match_idx]
            target["source_pages"] = sorted(set(target["source_pages"] + pages))
            if e.antecedent_cause and e.antecedent_cause not in target["antecedent_cause"]:
                target["antecedent_cause"] = f"{target['antecedent_cause']} | {e.antecedent_cause}".strip(" |")
            if e.consequent_effect and e.consequent_effect not in target["consequent_effect"]:
                target["consequent_effect"] = f"{target['consequent_effect']} | {e.consequent_effect}".strip(" |")
            for c in e.characters_present:
                if c not in target["characters"]:
                    target["characters"].append(c)
            if not target.get("chronological_clue") and getattr(e, "chronological_clue", ""):
                target["chronological_clue"] = e.chronological_clue
            target["merge_count"] += 1
            stats["merged"] += 1
        else:
            record = {
                "id": f"ev_{uuid.uuid4().hex[:12]}",
                "event_name": e.event_name,
                "category": e.category,
                "chronological_clue": getattr(e, "chronological_clue", ""),
                "location": e.location,
                "characters": list(e.characters_present),
                "core_event": e.core_event,
                "antecedent_cause": e.antecedent_cause,
                "consequent_effect": e.consequent_effect,
                "source_pages": pages,
                "merge_count": 1,
                "embedding": emb
            }
            events.append(record)
            stats["added"] += 1
            
        if on_progress and idx % 10 == 0:
            on_progress(0.7 + (idx / len(all_extracted)) * 0.3, f"merging events: {idx}/{len(all_extracted)}")

    _save_cache(doc_id, "pass2", {"events": events, "stats": stats})
    return events


# ============================================================
# PASS 3 — Normalization
# ============================================================
def run_pass3(events: list[dict]) -> list[dict]:
    """
    Normalize pages and setup defaults.
    Topological sort is now handled by the graph engine.
    """
    for e in events:
        e["source_pages"] = sorted(set(e.get("source_pages") or []))
        e["first_page"] = e["source_pages"][0] if e["source_pages"] else 0
        
    return events

# ============================================================
# Persistence
# ============================================================
def persist_events(doc_id: str, events: list[dict], on_progress=None) -> None:
    # Ensure they have embeddings if not already present
    missing = [e for e in events if 'embedding' not in e]
    if missing:
        texts = [
            f"{e['event_name']}. {e['core_event']} Characters: {', '.join(e['characters'])}. Location: {e['location']}."
            for e in missing
        ]
        vectors = []
        for i in range(0, len(texts), 64):
            vectors.extend(llm.embed(texts[i:i + 64]))
            if on_progress:
                on_progress(min(1.0, (i + 64) / max(1, len(texts))), f"embedding events {min(i + 64, len(texts))}/{len(texts)}")
        for e, v in zip(missing, vectors):
            e['embedding'] = v

    from pgvector.psycopg import Vector
    with pg() as cur:
        cur.execute("DELETE FROM events WHERE doc_id = %s", (doc_id,))
        cur.executemany(
            """INSERT INTO events
               (id, doc_id, event_name, category, chronological_clue, topological_order,
                location, characters, core_event, antecedent_cause, consequent_effect,
                source_pages, first_page, merge_count, embedding)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [
                (e["id"], doc_id, e["event_name"], e["category"], e.get("chronological_clue", ""),
                 e.get("topological_order", 0), e["location"], e["characters"], e["core_event"],
                 e["antecedent_cause"], e["consequent_effect"], e["source_pages"],
                 e["first_page"], e["merge_count"], Vector(e["embedding"]))
                for e in events
            ],
        )

def load_events(doc_id: str) -> list[dict]:
    with pg() as cur:
        cur.execute(
            """SELECT id, event_name, category, chronological_clue, topological_order, location,
                      characters, core_event, antecedent_cause, consequent_effect,
                      source_pages, first_page, merge_count
               FROM events WHERE doc_id = %s
               ORDER BY topological_order, first_page, event_name""",
            (doc_id,),
        )
        return [dict(r) for r in cur.fetchall()]
