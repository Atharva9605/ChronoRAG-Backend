from pydantic import BaseModel


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
