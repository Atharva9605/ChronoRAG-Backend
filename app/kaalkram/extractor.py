import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Callable

from pgvector.psycopg import Vector

from .. import llm
from ..config import settings
from ..db import pg
from .schemas import (
    ExtractedEvent,
    RollingMemory,
    WindowExtractionResult,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a chronological intelligence agent specializing in deep narrative and textual extraction.
Your task is to analyze a window of text from a document/book and construct an exact, point-wise chronological timeline of events.

CRITICAL EXTRACTION RULES:
1. EXACT PAGE MAPPING & FACTUAL PRECISION:
   - Each page in the text slice is clearly marked with `=== [PAGE N] ===`.
   - For every single event, you MUST record the exact `page_start`, `page_end`, and all `page_numbers` where it occurs.
   - For every event, provide `point_wise_actions` broken down by specific page number (e.g. page_no=N, action="...").
   - NEVER lose specific entity names, dates, quotes, numbers, dialogue, or locations. Capture every specific character/entity, setting, and concrete action.

2. EXTRACT ALL PAST EVENTS, MEMORIES & HISTORICAL BACKSTORY:
   - You MUST extract EVERY referenced historical event, character memory, backstory, prior milestone, or past exposition mentioned in the text as its own distinct event.
   - Any event describing something that occurred before the primary real-time narrative (e.g. youth memories, historical origins, prior life events, past disputes/agreements) MUST be extracted as an event.
   - Classify historical/prior events as 'flashback' or 'backstory' and set their `story_era` to 'Deep Past' or 'Recent Past'.

3. TEMPORAL CLASSIFICATION:
   - 'story_progression': Normal real-time narrative moving forward in the main timeline.
   - 'flashback': A memory, reflection, or scene depicting events from the past.
   - 'flash_forward': Foreshadowing, predictions, or future scenes.
   - 'backstory': Historical exposition, background lore, or prior life milestones.

4. CONTINUITY WITH PREVIOUS WINDOWS:
   - Use the provided 'STORY CONTEXT & ROLLING MEMORY SO FAR' to maintain continuity.
   - If an event is a continuation of an ongoing action from previous pages, reference the preceding event in `preceding_event_reference`.
   - If this window is the beginning of the document, set `preceding_event_reference` to 'None'.
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


def build_sliding_windows(pages: list[str], window_size: int = 10, overlap: int = 2) -> list[dict]:
    """Generates 10-page windows with 2-page overlap, with explicit page tags."""
    windows = []
    total = len(pages)
    start = 0
    step = max(1, window_size - overlap)
    
    while start < total:
        end = min(start + window_size, total)
        slice_pages = pages[start:end]
        
        formatted_text = "\n\n".join(
            f"=== [PAGE {start + i + 1}] ===\n{slice_pages[i]}"
            for i in range(len(slice_pages))
        )
        
        windows.append({
            "window_index": len(windows) + 1,
            "page_start": start + 1,
            "page_end": end,
            "text": formatted_text,
        })
        
        if end == total:
            break
        start += step
        
    return windows


def extract_timeline_events(
    doc_id: str,
    pages: list[str],
    on_progress: Callable[[float, str], None] | None = None,
) -> list[dict]:
    """
    Extracts chronological events using sequential rolling memory across sliding windows.
    Checkpoints every window's response and state to data/cache/.
    """
    windows = build_sliding_windows(pages, window_size=10, overlap=2)
    total_windows = len(windows)
    
    memory = RollingMemory()
    all_events: list[dict] = []
    event_counter = 1
    
    for idx, win in enumerate(windows):
        progress_val = idx / total_windows
        status_msg = f"Reading pages {win['page_start']}-{win['page_end']} (Window {idx + 1}/{total_windows})"
        if on_progress:
            on_progress(progress_val * 0.75, status_msg)
            
        # Build prompt incorporating rolling memory
        memory_context = (
            f"STORY CONTEXT & ROLLING MEMORY SO FAR:\n"
            f"- Story Summary: {memory.story_summary_so_far or 'Start of the story.'}\n"
            f"- Known Active Characters: {', '.join(memory.active_characters) or 'None'}\n"
            f"- Recent Prior Events: {json.dumps(memory.recent_events_summary[-3:], indent=2) if memory.recent_events_summary else 'None'}\n"
            f"- Unresolved Threads: {', '.join(memory.unresolved_threads) or 'None'}\n\n"
        )
        
        user_prompt = (
            f"{memory_context}"
            f"CURRENT TEXT WINDOW (Pages {win['page_start']} to {win['page_end']}):\n\n"
            f"{win['text']}\n\n"
            f"Extract all distinct narrative events in this window with point-wise page breakdowns."
        )
        
        # Azure OpenAI Structured Output
        result: WindowExtractionResult = llm.chat_structured(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            model_cls=WindowExtractionResult,
            temperature=0.1,
            max_tokens=4000,
        )
        
        # Process and link extracted events
        window_event_dicts = []
        for ev in result.events:
            event_id = f"ev_{doc_id}_{event_counter:03d}"
            
            # Convert point_wise_actions to serializable dicts
            action_items = [
                {"page_no": act.page_no, "action": act.action}
                for act in ev.point_wise_actions
            ]
            
            ev_dict = {
                "id": event_id,
                "doc_id": doc_id,
                "event_name": ev.event_name,
                "classification": ev.classification,
                "story_era": ev.story_era or "Present",
                "summary": ev.summary,
                "page_start": ev.page_start,
                "page_end": ev.page_end,
                "page_numbers": ev.page_numbers or list(range(ev.page_start, ev.page_end + 1)),
                "point_wise_actions": action_items,
                "characters": ev.characters_involved,
                "temporal_anchor": ev.temporal_anchor or "",
                "preceding_event_reference": ev.preceding_event_reference or "None",
                "consequence_or_effect": ev.consequence_or_effect or "",
                "window_index": win["window_index"],
            }
            
            all_events.append(ev_dict)
            window_event_dicts.append({
                "id": event_id,
                "name": ev.event_name,
                "pages": f"{ev.page_start}-{ev.page_end}",
                "classification": ev.classification,
                "summary": ev.summary[:140],
            })
            event_counter += 1
            
        # Update rolling memory for next window
        summary_update_prompt = (
            f"Previous story summary: {memory.story_summary_so_far}\n"
            f"New developments in pages {win['page_start']}-{win['page_end']}: {result.window_summary}\n"
            f"Synthesize this into a concise 1-paragraph updated story summary so far."
        )
        updated_story_summary = llm.chat(
            system="You maintain a concise, ongoing story synopsis.",
            user=summary_update_prompt,
            max_tokens=350,
        )
        
        # Update memory state
        memory.story_summary_so_far = updated_story_summary
        memory.last_window_page_end = win["page_end"]
        memory.active_characters = list(set(memory.active_characters + result.active_characters))
        memory.recent_events_summary.extend(window_event_dicts)
        
        # Save inspection checkpoint for this window
        checkpoint_payload = {
            "window_index": win["window_index"],
            "page_range": f"{win['page_start']}-{win['page_end']}",
            "rolling_memory_prompt_sent": memory_context,
            "window_summary_extracted": result.window_summary,
            "active_characters_extracted": result.active_characters,
            "events_extracted_count": len(result.events),
            "events": [e.model_dump() for e in result.events],
            "rolling_memory_state_after_window": memory.model_dump(),
        }
        _save_json_checkpoint(
            f"{doc_id}_checkpoint_window_{win['window_index']:02d}_pages_{win['page_start']}_to_{win['page_end']}.json",
            checkpoint_payload,
        )
        
    # Save overall all-events checkpoint
    _save_json_checkpoint(f"{doc_id}_checkpoint_all_events.json", all_events)
    return all_events


def persist_events_to_postgres(doc_id: str, events: list[dict], on_progress=None) -> None:
    """Embeds events and saves them to PostgreSQL."""
    if not events:
        return

    if on_progress:
        on_progress(0.80, "Embedding events for semantic retrieval")

    # Generate embeddings
    texts_to_embed = [
        f"{e['event_name']}: {e['summary']} "
        f"Characters: {', '.join(e['characters'])}. "
        f"Actions: {' '.join(a['action'] for a in e['point_wise_actions'])}"
        for e in events
    ]
    embeddings = llm.embed(texts_to_embed)

    with pg() as cur:
        # Clear previous events for doc
        cur.execute("DELETE FROM events WHERE doc_id = %s", (doc_id,))
        
        records = []
        for e, emb in zip(events, embeddings):
            # category map: story_progression -> major, flashback/backstory -> minor
            category = "major" if e["classification"] == "story_progression" else "minor"
            actions_json = json.dumps(e["point_wise_actions"])
            
            records.append((
                e["id"],
                doc_id,
                e["event_name"],
                category,
                e["temporal_anchor"],
                e.get("topological_order", 0),
                e["classification"],  # stored in location or dedicated column
                e["characters"],
                e["summary"],
                e["preceding_event_reference"],
                actions_json,  # stored in consequent_effect field for rich actions
                e["page_numbers"],
                e["page_start"],
                1,
                Vector(emb),
            ))
            
        cur.executemany(
            """INSERT INTO events (
                   id, doc_id, event_name, category, chronological_clue,
                   topological_order, location, characters, core_event,
                   antecedent_cause, consequent_effect, source_pages,
                   first_page, merge_count, embedding
               ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            records,
        )
