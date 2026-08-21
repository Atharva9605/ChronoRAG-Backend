import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from pgvector.psycopg import Vector

from . import llm
from .config import settings, TAXONOMY, stage_index
from .db import pg
from .ingest import parse_pages_from_bullet
from .schemas import BatchMergeResponse

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
PASS2_SYSTEM = f"""You are a structural data engineer managing a narrative timeline database.
You receive (a) a light index of events ALREADY in the database, and (b) a new set of raw
observations. Produce one instruction per observation describing how to integrate it.

STRICT TIMELINE ANCHOR TAXONOMY - timeline_anchor must be EXACTLY one of:
1. "{TAXONOMY[0]}"      (84 unlucky days, Manolin forced to leave, coffee, preparations on shore)
2. "{TAXONOMY[1]}"      (alone at sea, flying fish, bait, the great strike / hooking the marlin)
3. "{TAXONOMY[2]}"      (multi-day fight with the marlin, hand cuts, endurance, killing the fish)
4. "{TAXONOMY[3]}"      (lashing the fish alongside, shark attacks stripping the catch, skeleton left)
5. "{TAXONOMY[4]}"      (return to the village, Manolin weeps, tourists see the skeleton, sleep/lions)

Assign the anchor by WHERE THE EVENT SITS IN STORY-WORLD TIME, never by where it is printed.
A late scene printed anywhere that shows the skeleton on the beach or tourists belongs in "{TAXONOMY[4]}".
Shark attacks belong in "{TAXONOMY[3]}", which is AFTER the marlin is killed ("{TAXONOMY[2]}").

CATEGORY:
- "major": core structural milestones that drive the main plot - inciting incidents, turning
  points, climax, resolution, irreversible decisions. A novel has roughly 8-15 of these. Be strict.
- "minor": subplots, backstory reveals, secondary interactions, atmosphere, comic relief.

DUPLICATE DETECTION:
Compare each observation against the baseline index. If it describes the SAME event frame as an
existing entry (same participants, same action, adjacent pages), set is_duplicate=true and put the
EXACT baseline event_name in matched_event_name. Otherwise is_duplicate=false and
matched_event_name must be an empty string.

LOCATION FORMAT: specific-to-general, e.g. "skiff, Gulf Stream off Cuba". Never let a room name
replace the setting.

CAUSALITY: antecedent_cause = what triggered this event. consequent_effect = what it leads to.
Both must be grounded in the text, not invented.

SOURCE PAGES: copy the [PAGE N] numbers from the observation bullet verbatim."""


def _light_index(events: list[dict]) -> list[dict]:
    return [
        {"event_name": e["event_name"], "category": e["category"],
         "timeline_anchor": e["timeline_anchor"], "source_pages": e["source_pages"]}
        for e in events
    ]


def _native_merge(events: list[dict], resp: BatchMergeResponse) -> tuple[int, int]:
    """Apply LLM decisions in plain Python. Returns (added, merged)."""
    by_name = {e["event_name"]: e for e in events}
    added = merged = 0

    for ins in resp.instructions:
        inf = ins.inferred_event
        pages = sorted(set(int(p) for p in inf.source_pages if p and p > 0))

        if ins.is_duplicate and ins.matched_event_name:
            target = by_name.get(ins.matched_event_name)
            if target is not None:
                target["source_pages"] = sorted(set(target["source_pages"] + pages))
                if inf.antecedent_cause and inf.antecedent_cause not in target["antecedent_cause"]:
                    target["antecedent_cause"] = (
                        f"{target['antecedent_cause']} | {inf.antecedent_cause}".strip(" |")
                    )
                if inf.consequent_effect and inf.consequent_effect not in target["consequent_effect"]:
                    target["consequent_effect"] = (
                        f"{target['consequent_effect']} | {inf.consequent_effect}".strip(" |")
                    )
                for c in inf.characters_present:
                    if c not in target["characters"]:
                        target["characters"].append(c)
                target["merge_count"] += 1
                merged += 1
                continue
            # matched name not found -> fall through and add as new

        record = {
            "id": f"ev_{uuid.uuid4().hex[:12]}",
            "event_name": inf.event_name,
            "category": ins.category,
            "timeline_anchor": inf.timeline_anchor,
            "location": inf.location,
            "characters": list(inf.characters_present),
            "core_event": inf.core_event,
            "antecedent_cause": inf.antecedent_cause,
            "consequent_effect": inf.consequent_effect,
            "source_pages": pages,
            "merge_count": 1,
        }
        events.append(record)
        by_name[record["event_name"]] = record
        added += 1

    return added, merged


def run_pass2(doc_id: str, observations: list[dict], on_progress=None) -> list[dict]:
    """
    Parallel fetch, sequential commit.

    Several batches are sent to the model at once against a FROZEN snapshot of the
    timeline index, but their decisions are applied one at a time. That keeps the
    merge deterministic and free of race conditions.
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
    stats = {"added": 0, "merged": 0, "failed": 0}

    def fetch(idx: int, index_snapshot: list[dict], text: str):
        user = (
            f"BASELINE TIMELINE INDEX ({len(index_snapshot)} events):\n"
            f"{json.dumps(index_snapshot, indent=2, ensure_ascii=False)}\n\n"
            f"NEW OBSERVATIONS TO RECONCILE:\n{text}"
        )
        try:
            return idx, llm.chat_structured(PASS2_SYSTEM, user, BatchMergeResponse)
        except Exception:
            return idx, None

    processed = 0
    for start in range(0, total, settings.concurrency):
        group = list(enumerate(batches[start:start + settings.concurrency], start=start))
        snapshot = _light_index(events)      # frozen for this whole group

        results: dict[int, BatchMergeResponse | None] = {}
        with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
            futures = [pool.submit(fetch, i, snapshot, t) for i, t in group]
            for fut in as_completed(futures):
                i, parsed = fut.result()
                results[i] = parsed

        for i, _ in group:                   # commit in deterministic order
            parsed = results.get(i)
            if parsed is None:
                stats["failed"] += 1
            else:
                a, m = _native_merge(events, parsed)
                stats["added"] += a
                stats["merged"] += m
            processed += 1
            if on_progress:
                on_progress(processed / total,
                            f"pass 2: batch {processed}/{total} "
                            f"({len(events)} events, {stats['merged']} merges)")

        _save_cache(doc_id, "pass2", {"events": events, "stats": stats})

    return events


# ============================================================
# PASS 3 — deterministic chronology
# ============================================================
FRAME_HINTS = ("tourist", "tourists", "skeleton", "aftermath", "homecoming",
               "years later", "epilogue")


def run_pass3(events: list[dict]) -> list[dict]:
    """
    Two deterministic steps, no LLM:
      1. Assign a stage_order from the taxonomy, overriding obvious framing devices.
      2. Sort by (stage_order, first source page).
    """
    for e in events:
        e["source_pages"] = sorted(set(e.get("source_pages") or []))
        e["first_page"] = e["source_pages"][0] if e["source_pages"] else 0

        blob = f"{e['event_name']} {e['core_event']}".lower()
        if any(h in blob for h in FRAME_HINTS):
            e["timeline_anchor"] = TAXONOMY[-1]      # Future / Epilogue
        e["stage_order"] = stage_index(e["timeline_anchor"])

    events.sort(key=lambda e: (e["stage_order"], e["first_page"], e["event_name"]))
    return events


# ============================================================
# Persistence
# ============================================================
def persist_events(doc_id: str, events: list[dict], on_progress=None) -> None:
    texts = [
        f"{e['event_name']}. {e['core_event']} "
        f"Characters: {', '.join(e['characters'])}. Location: {e['location']}."
        for e in events
    ]
    vectors: list = []
    for i in range(0, len(texts), 64):
        vectors.extend(llm.embed(texts[i:i + 64]))
        if on_progress:
            on_progress(min(1.0, (i + 64) / max(1, len(texts))),
                        f"embedding events {min(i + 64, len(texts))}/{len(texts)}")

    with pg() as cur:
        cur.execute("DELETE FROM events WHERE doc_id = %s", (doc_id,))
        cur.executemany(
            """INSERT INTO events
               (id, doc_id, event_name, category, timeline_anchor, stage_order,
                location, characters, core_event, antecedent_cause, consequent_effect,
                source_pages, first_page, merge_count, embedding)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [
                (e["id"], doc_id, e["event_name"], e["category"], e["timeline_anchor"],
                 e["stage_order"], e["location"], e["characters"], e["core_event"],
                 e["antecedent_cause"], e["consequent_effect"], e["source_pages"],
                 e["first_page"], e["merge_count"], Vector(v))
                for e, v in zip(events, vectors)
            ],
        )


def load_events(doc_id: str) -> list[dict]:
    with pg() as cur:
        cur.execute(
            """SELECT id, event_name, category, timeline_anchor, stage_order, location,
                      characters, core_event, antecedent_cause, consequent_effect,
                      source_pages, first_page, merge_count
               FROM events WHERE doc_id = %s
               ORDER BY stage_order, first_page, event_name""",
            (doc_id,),
        )
        return [dict(r) for r in cur.fetchall()]
