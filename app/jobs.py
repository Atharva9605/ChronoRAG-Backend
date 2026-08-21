import json
import traceback
import uuid

from .db import pg


def create(doc_id: str, kind: str) -> str:
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    with pg() as cur:
        cur.execute(
            """INSERT INTO jobs (id, doc_id, kind, status, stage, progress)
               VALUES (%s,%s,%s,'queued','',0)""",
            (job_id, doc_id, kind),
        )
    return job_id


def update(job_id: str, *, status=None, stage=None, progress=None,
           detail=None, error=None) -> None:
    sets, args = [], []
    if status is not None:
        sets.append("status = %s"); args.append(status)
    if stage is not None:
        sets.append("stage = %s"); args.append(stage)
    if progress is not None:
        sets.append("progress = %s"); args.append(float(progress))
    if detail is not None:
        sets.append("detail = %s"); args.append(json.dumps(detail))
    if error is not None:
        sets.append("error = %s"); args.append(error)
    if status in ("done", "error"):
        sets.append("finished_at = now()")
    if not sets:
        return
    args.append(job_id)
    with pg() as cur:
        cur.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = %s", args)


def get(job_id: str) -> dict | None:
    with pg() as cur:
        cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def latest(doc_id: str, kind: str) -> dict | None:
    with pg() as cur:
        cur.execute(
            """SELECT * FROM jobs WHERE doc_id = %s AND kind = %s
               ORDER BY started_at DESC LIMIT 1""",
            (doc_id, kind),
        )
        row = cur.fetchone()
    return dict(row) if row else None


# ------------------------------------------------------------
# Runners (executed in a FastAPI BackgroundTask thread)
# ------------------------------------------------------------
def run_naive(job_id: str, doc_id: str) -> None:
    from . import naive_rag
    from .main import doc_pages
    try:
        update(job_id, status="running", stage="chunk+embed", progress=0.02)
        pages = doc_pages(doc_id)
        count = naive_rag.build(
            doc_id, pages,
            on_progress=lambda p, msg: update(job_id, progress=p * 0.98 + 0.02, stage=msg),
        )
        update(job_id, status="done", stage="complete", progress=1.0,
               detail={"chunks": count})
    except Exception as exc:
        traceback.print_exc()
        update(job_id, status="error", error=f"{type(exc).__name__}: {exc}")


def run_kaalkram(job_id: str, doc_id: str) -> None:
    from . import graph, passes
    from .ingest import sliding_windows
    from .main import doc_pages
    try:
        pages = doc_pages(doc_id)
        windows = sliding_windows(pages)

        # ---- Pass 1: 0.00 -> 0.45
        update(job_id, status="running", stage="pass 1: reading windows", progress=0.01)
        obs = passes.run_pass1(
            doc_id, windows,
            on_progress=lambda p, msg: update(job_id, progress=0.01 + p * 0.44, stage=msg),
        )

        # ---- Pass 2: 0.45 -> 0.80
        update(job_id, stage="pass 2: merging duplicates", progress=0.45)
        events = passes.run_pass2(
            doc_id, obs,
            on_progress=lambda p, msg: update(job_id, progress=0.45 + p * 0.35, stage=msg),
        )

        # ---- Pass 3 + persist: 0.80 -> 0.93
        update(job_id, stage="pass 3: ordering the story", progress=0.80)
        events = passes.run_pass3(events)
        passes.persist_events(
            doc_id, events,
            on_progress=lambda p, msg: update(job_id, progress=0.82 + p * 0.11, stage=msg),
        )

        # ---- Graph: 0.93 -> 1.00
        update(job_id, stage="building event graph", progress=0.94)
        edges, stats = graph.build_edges(events)
        graph.push(doc_id, events, edges)

        majors = sum(1 for e in events if e["category"] == "major")
        merges = sum(e["merge_count"] - 1 for e in events)
        update(job_id, status="done", stage="complete", progress=1.0, detail={
            "events": len(events),
            "major": majors,
            "minor": len(events) - majors,
            "merges": merges,
            "windows": len(windows),
            **stats,
        })
    except Exception as exc:
        traceback.print_exc()
        update(job_id, status="error", error=f"{type(exc).__name__}: {exc}")
