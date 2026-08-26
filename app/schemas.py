from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# LLM contracts (Part B) — these become strict JSON schemas
# ============================================================
class TaxonomyStage(BaseModel):
    name: str = Field(
        description="Short stage label, e.g. 'Shore / Setup' or 'Exam Crisis'. 2-6 words."
    )
    description: str = Field(
        description="What belongs in this stage — concrete plot beats, not vague theme talk."
    )
    is_framing: bool = Field(
        description=(
            "True if this stage is framing/epilogue/aftermath that may be printed out of "
            "story-world order (prologue hospital scene, tourists at the end, etc.)."
        )
    )


class TaxonomyProposal(BaseModel):
    stages: list[TaxonomyStage] = Field(
        description="5 to 7 stages ordered by story-world time from earliest to latest"
    )


class TimelineEvent(BaseModel):
    event_name: str = Field(description="Short descriptive name of the event")
    timeline_anchor: str = Field(
        description=(
            "Exactly one stage name from the STRICT TIMELINE ANCHOR TAXONOMY "
            "listed in the system prompt (copy the stage name verbatim)"
        )
    )
    location: str = Field(
        description="Setting, formatted specific-to-general e.g. 'skiff, Gulf Stream off Cuba'"
    )
    characters_present: list[str] = Field(description="Names of characters involved")
    core_event: str = Field(description="One or two sentence summary of what happens")
    antecedent_cause: str = Field(description="What triggered this event")
    consequent_effect: str = Field(description="What this event causes or leads to")
    source_pages: list[int] = Field(description="PDF page numbers where this event appears")


class MergeInstruction(BaseModel):
    is_duplicate: bool = Field(description="True if this observation is an event already in the baseline index")
    matched_event_name: str = Field(description="Exact baseline event_name if is_duplicate, else empty string")
    category: Literal["major", "minor"] = Field(
        description="major = drives the main plot; minor = subplot, backstory, atmosphere"
    )
    inferred_event: TimelineEvent


class BatchMergeResponse(BaseModel):
    instructions: list[MergeInstruction] = Field(
        description="One instruction per observation bullet, in order"
    )


class GraphAnswer(BaseModel):
    """Final natural-language rendering, grounded in supplied events only."""
    answer: str = Field(description="Direct answer in plain prose, citing pages as (p. N)")
    used_event_ids: list[str] = Field(description="ids of events actually used in the answer")


# ============================================================
# API models
# ============================================================
class TaxonomyStageOut(BaseModel):
    name: str
    description: str = ""
    is_framing: bool = False


class DocumentOut(BaseModel):
    id: str
    title: str
    filename: str
    page_count: int
    naive_ready: bool = False
    kaalkram_ready: bool = False
    event_count: int = 0
    taxonomy: list[TaxonomyStageOut] = []


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
    timeline_anchor: str
    stage_order: int
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
