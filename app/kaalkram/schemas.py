from typing import Literal
from pydantic import BaseModel, Field

EventClassification = Literal["story_progression", "flashback", "flash_forward", "backstory"]


class PageActionItem(BaseModel):
    page_no: int = Field(description="The exact PDF page number where this action happens.")
    action: str = Field(description="Specific, detailed narrative action or dialogue occurring on this page.")


class ExtractedEvent(BaseModel):
    event_name: str = Field(
        description="A concise 4-7 word title of the specific event or scene."
    )
    classification: EventClassification = Field(
        description=(
            "Temporal classification: "
            "'story_progression' = real-time forward story beat; "
            "'flashback' = memory/scene from the past; "
            "'flash_forward' = foreshadowing/future scene; "
            "'backstory' = historical exposition or lore."
        )
    )
    summary: str = Field(
        description="Comprehensive summary of what happens in this event, retaining all specific names, objects, quotes, and outcomes."
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
        description="Explicit time marker from text (e.g., 'Morning of Day 1', '84 days ago', 'At dusk')."
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
        description="Key characters active in this slice."
    )
    events: list[ExtractedEvent] = Field(
        description="Chronologically ordered list of distinct events occurring within this window."
    )


class RollingMemory(BaseModel):
    story_summary_so_far: str = ""
    last_window_page_end: int = 0
    active_characters: list[str] = []
    recent_events_summary: list[dict] = []
    unresolved_threads: list[str] = []


class TimelineGraphNode(BaseModel):
    id: str
    doc_id: str
    event_name: str
    classification: str
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
