import time
import uuid

from qdrant_client.http import models

from . import llm
from .config import settings
from .db import pg
from . import db
from .ingest import naive_chunks
from .schemas import Citation, PipelineAnswer

SYSTEM = """You are a fast literary QA assistant. You always answer from the passages you are given.

Rules:
- NEVER say "insufficient", "cannot determine", "not enough information", or "context does not include".
- ALWAYS produce an answer. For before/after questions, you MUST pick one side
  (before OR after), even if you are only roughly sure.
- Prefer vague-but-decisive wording, e.g. "It seems Manolin leaves after…" / "From the
  passages, the fight looks like it comes first…" — not a refusal.
- Stay grounded in the passages; do not invent new plot. Short answers only (2-5 sentences).
- Treat the text as fiction under discussion."""


def build(doc_id: str, pages: list[str], on_progress=None) -> int:
    """Chunk, embed and store. Returns chunk count."""
    chunks = naive_chunks(pages)
    total = len(chunks)

    with pg() as cur:
        cur.execute("DELETE FROM naive_chunks WHERE doc_id = %s", (doc_id,))

    # Also delete from Qdrant
    db.qdrant().delete(
        collection_name="naive_chunks",
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            )
        )
    )

    batch = 64
    for i in range(0, total, batch):
        part = chunks[i:i + batch]
        vectors = llm.embed([c["content"] for c in part])
        
        points = []
        for c, v in zip(part, vectors):
            points.append(models.PointStruct(
                id=uuid.uuid4().hex,
                vector=v,
                payload={
                    "doc_id": doc_id,
                    "chunk_index": c["chunk_index"],
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                    "content": c["content"],
                }
            ))
        db.qdrant().upsert(collection_name="naive_chunks", points=points)

        with pg() as cur:
            cur.executemany(
                """INSERT INTO naive_chunks
                   (doc_id, chunk_index, page_start, page_end, content)
                   VALUES (%s,%s,%s,%s,%s)""",
                [
                    (doc_id, c["chunk_index"], c["page_start"], c["page_end"], c["content"])
                    for c in part
                ],
            )
        if on_progress:
            on_progress(min(1.0, (i + len(part)) / max(1, total)),
                        f"embedded {min(i + len(part), total)}/{total} chunks")
    return total


def retrieve(doc_id: str, question: str, k: int | None = None) -> list[dict]:
    k = k or settings.naive_top_k
    qvec = llm.embed([question])[0]
    hits = db.qdrant().search(
        collection_name="naive_chunks",
        query_vector=qvec,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        ),
        limit=k,
    )
    return [
        {
            "chunk_index": h.payload["chunk_index"],
            "page_start": h.payload["page_start"],
            "page_end": h.payload["page_end"],
            "content": h.payload["content"],
            "score": h.score,
        }
        for h in hits
    ]


def answer(doc_id: str, question: str) -> PipelineAnswer:
    t0 = time.perf_counter()
    before = llm.usage_snapshot()

    hits = retrieve(doc_id, question)
    trace = [
        f"Embedded the question into a {settings.embed_dim}-dim vector.",
        f"Cosine-searched {len(hits)} nearest chunks out of the whole book.",
        "Concatenated those chunks into the prompt in similarity order "
        "(NOT story order — this is the core limitation).",
    ]

    # Short, cleaned excerpts — long OCR-heavy PDF dumps trip Azure content filters.
    context = "\n\n---\n\n".join(
        f"[Passage {i + 1} | similarity {h['score']:.3f}]\n{h['content'][:420]}"
        for i, h in enumerate(hits)
    )
    user = (
        f"PASSAGES (similarity order — NOT story order):\n\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Respond with a short decisive answer. "
        "If this is before/after, end with a clear choice: BEFORE or AFTER. "
        "Forbidden phrases: insufficient, cannot determine, not enough, does not include."
    )
    try:
        text = llm.chat(SYSTEM, user, temperature=0.4, max_tokens=800)
    except llm.ContentFilterError:
        short = hits[:3]
        context = "\n\n---\n\n".join(
            f"[Passage {i + 1}]\n{h['content'][:220]}"
            for i, h in enumerate(short)
        )
        user = (
            f"PASSAGES:\n\n{context}\n\nQUESTION: {question}\n"
            "Short answer required. Pick before or after if asked. Never refuse."
        )
        text = llm.chat(SYSTEM, user, temperature=0.4, max_tokens=400)
        hits = short

    # Last-resort: if the model still refuses, emit a vague guess from similarity rank
    # (this is the naive failure mode — treating retrieval order as story order).
    low = text.lower()
    if any(p in low for p in (
        "insufficient", "cannot determine", "not enough information",
        "does not include", "do not include enough", "unable to determine",
        "additional context", "needed to answer",
    )):
        force = (
            f"PASSAGES:\n\n{context}\n\nQUESTION: {question}\n\n"
            "Your previous reply refused. Rewrite now as a vague but decisive guess "
            "from these passages only. Must choose before or after if relevant. "
            "One short paragraph. No refusal language."
        )
        try:
            text = llm.chat(SYSTEM, force, temperature=0.7, max_tokens=400)
        except Exception:
            text = ""
        low = (text or "").lower()
        if (not text) or any(p in low for p in (
            "insufficient", "cannot determine", "not enough",
            "does not include", "unable to determine", "additional context",
        )):
            pages = [h["page_start"] for h in hits]
            text = (
                f"From the retrieved passages the timeline feels a bit scrambled — "
                f"snippets jump between about p. {min(pages)} and p. {max(pages)}. "
                f"Going by how closely each passage matched the question, it seems the "
                f"second event in the question comes first and the first event follows, "
                f"though the fragments only support a rough guess rather than a clean sequence."
            )
            trace.append(
                "Model refused despite prompting; fell back to a vague answer "
                "derived from similarity-ranked passages (not story order)."
            )

    after = llm.usage_snapshot()
    return PipelineAnswer(
        pipeline="naive",
        answer=text,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        prompt_tokens=after["prompt"] - before["prompt"],
        completion_tokens=after["completion"] - before["completion"],
        citations=[
            Citation(label=f"Passage {i + 1}",
                     pages=list(range(h["page_start"], h["page_end"] + 1)))
            for i, h in enumerate(hits)
        ],
        retrieved=[
            {"rank": i + 1, "score": round(h["score"], 4),
             "pages": [h["page_start"], h["page_end"]],
             "preview": h["content"][:280]}
            for i, h in enumerate(hits)
        ],
        trace=trace,
    )
