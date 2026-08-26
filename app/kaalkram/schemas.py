from typing import Literal
from pydantic import BaseModel, Field

EventClassification = Literal["story_progression", "flashback", "flash_forward", "backstory"]


class PageActionItem(BaseModel):
    page_no: int = Field(description="The exact PDF page number where this action or dialogue happens.")
    action: str = Field(description="Specific, detailed narrative action or dialogue occurring on this page.")


class ExtractedEvent(BaseModel):
    event_name: str = Field(
        description="A concise 4-7 word title of the specific event, memory, or historical backstory."
    )
    classification: EventClassification = Field(
        description=(
            "Temporal classification: "
            "'story_progression' = real-time forward story beat; "
            "'flashback' = memory/scene from the past; "
            "'flash_forward' = foreshadowing/future scene; "
            "'backstory' = historical exposition, past expeditions, or character lore."
        )
    )
    story_era: str = Field(
        description=(
            "The story-world era/epoch when this event actually took place: "
            "e.g., 'Deep Past (Youth & Africa Voyages)', 'Recent Past (Early Days with Manolin & Past Streaks)', "
            "'Present Day 1 (Shore & Shack Preparation)', 'Present Day 2 (Rowing Out & Hooking Marlin)', "
            "'Present Day 3 (The Marlin Battle & Endurance)', 'Present Day 4 (Shark Attacks & Return)', "
            "'Present Day 5 (Aftermath on Shore)'."
        )
    )
    summary: str = Field(
        description="Comprehensive summary of what happened, retaining all specific names, numbers, objects, quotes, and outcomes."
    )
    page_start: int = Field(description="Starting PDF page number of this event.")
    page_end: int = Field(description="Ending PDF page number of this event.")
    page_numbers: list[int] = Field(
        description="List of all PDF page numbers spanning this event."
    )
    point_wise_actions: list[PageActionItem] = Field(
        description="Point-wise breakdown of exact actions and dialogue mapped to their exact page numbers."
    )
    characters_involved: list[str] = Field(
        description="List of all character names present or referenced."
    )
    temporal_anchor: str = Field(
        description="Primary explicit time marker from text (e.g., 'September 1950', 'Years ago in youth', 'Morning of Day 1', '84 days ago')."
    )
    exact_dates_and_timestamps: list[str] = Field(
        description="All explicit dates, calendar years, seasons, times of day (e.g. 'dawn', 'sunset', 'noon'), or durations (e.g. 'two hours later', 'for three days') mentioned."
    )
    relative_time_context: str = Field(
        description="Precise chronological placement relative to the surrounding story (e.g. 'Occurs the morning after the shack discussion', 'Occurred 20 years before the main story')."
    )
    preceding_event_reference: str = Field(
        description="Reference or name of the prior event that directly preceded or caused this one. 'None' if starting event."
    )
    consequence_or_effect: str = Field(
        description="Immediate causal fallout or next development resulting from this event."
    )


class WindowExtractionResult(BaseModel):
    window_summary: str = Field(
        description="Summary of all major developments across this 10-page window."
    )
    active_characters: list[str] = Field(
        description="Key characters active or discussed in this slice."
    )
    events: list[ExtractedEvent] = Field(
        description="Chronologically ordered list of distinct events, including both real-time actions and historical memories/backstories."
    )


class RollingMemory(BaseModel):
    story_summary_so_far: str = ""
    last_window_page_end: int = 0
    active_characters: list[str] = []
    recent_events_summary: list[dict] = []
    unresolved_threads: list[str] = []


# ============================================================
# Global LLM Timeline Ordering Models
# ============================================================
class OrderedTimelineEventItem(BaseModel):
    event_id: str = Field(description="The unique ID of the event being placed.")
    chronological_rank: int = Field(
        description="Strict sequential integer rank in true story time (1 = earliest in history/past, N = final resolution)."
    )
    story_era: str = Field(
        description="Era category (e.g. 'Deep Past', 'Recent Past', 'Day 1 Shore', 'Day 2 Hook', 'Day 3 Battle', 'Day 4 Sharks', 'Day 5 Return')."
    )
    story_time_period: str = Field(
        description="Estimated story time period (e.g. 'Decades ago in Casablanca', 'Day 1 Morning', 'Day 3 Night')."
    )
    chronological_rationale: str = Field(
        description="Brief 1-sentence reason explaining why this event occurs at this specific point in story chronology."
    )


class GlobalTimelineOrderingResponse(BaseModel):
    chronological_overview: str = Field(
        description="High-level breakdown of the story's true chronological progression across eras from past to present."
    )
    ordered_events: list[OrderedTimelineEventItem] = Field(
        description="All events ordered strictly in true story-world chronological sequence."
    )


class TimelineGraphNode(BaseModel):
    id: str
    doc_id: str
    event_name: str
    classification: str
    story_era: str = ""
    summary: str
    page_start: int
    page_end: int
    page_numbers: list[int]
    characters: list[str]
    temporal_anchor: str
    topological_order: int = 0
    point_wise_actions: list[dict] = []


class TimelineGraphEdge(BaseModel):
    source: str
    target: str
    relation: str  # 'HAPPENS_BEFORE', 'CAUSES', 'FLASHBACK_FROM'
