from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# LLM contracts (Part B) — these become strict JSON schemas
# ============================================================
class TimelineEvent(BaseModel):
    event_name: str = Field(description="Short descriptive name of the event")
    chronological_clue: str = Field(
        description="Any explicit time markers in the text (e.g., '1922', 'the next morning', 'after the war'). Leave empty if none."
    )
    category: Literal["major", "minor"] = Field(
        description="major = drives the main plot; minor = subplot, backstory, atmosphere"
    )
    location: str = Field(description="Setting, formatted specific-to-general e.g. 'Kumaon Hostel, IIT Delhi'")
    characters_present: list[str] = Field(description="Names of characters involved")
    core_event: str = Field(description="One or two sentence summary of what happens")
    antecedent_cause: str = Field(description="What triggered this event")
    consequent_effect: str = Field(description="What this event causes or leads to")
    source_pages: list[int] = Field(description="PDF page numbers where this event appears")


class BatchExtractResponse(BaseModel):
    events: list[TimelineEvent] = Field(description="List of extracted timeline events from the observations.")


class GraphAnswer(BaseModel):
    """Final natural-language rendering, grounded in supplied events only."""
    answer: str = Field(description="Direct answer in plain prose, citing pages as (p. N)")
    used_event_ids: list[str] = Field(description="ids of events actually used in the answer")


# ============================================================
# API models
# ============================================================
class DocumentOut(BaseModel):
    id: str
    title: str
    filename: str
    page_count: int
    naive_ready: bool = False
    kaalkram_ready: bool = False
    event_count: int = 0


class JobOut(BaseModel):
    id: str
    doc_id: str
    kind: str
    status: str
    stage: str
    progress: float
    detail: dict
    error: str | None = None


class EventOut(BaseModel):
    id: str
    event_name: str
    category: str
    chronological_clue: str
    topological_order: int
    location: str
    characters: list[str]
    core_event: str
    antecedent_cause: str
    consequent_effect: str
    source_pages: list[int]
    first_page: int
    merge_count: int


class Citation(BaseModel):
    label: str
    pages: list[int]


class PipelineAnswer(BaseModel):
    pipeline: str
    answer: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    citations: list[Citation] = []
    retrieved: list[dict] = []
    trace: list[str] = []


class CompareResponse(BaseModel):
    question: str
    naive: PipelineAnswer
    kaalkram: PipelineAnswer
