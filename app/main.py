import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import db, jobs, llm, naive_rag
from .config import settings
from .ingest import doc_id_for, extract_pages
from .schemas import DocumentOut, JobOut


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.pool()
    db.init_neo4j()
    yield
    db.close()


app = FastAPI(title="ChronoRAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
def doc_pages(doc_id: str) -> list[str]:
    with db.pg() as cur:
        cur.execute(
            "SELECT content FROM pages WHERE doc_id = %s ORDER BY page_no", (doc_id,)
        )
        rows = cur.fetchall()
    if not rows:
        raise HTTPException(404, "document not found")
    return [r["content"] for r in rows]


# ------------------------------------------------------------
# Documents
# ------------------------------------------------------------
@app.post("/api/documents", response_model=DocumentOut)
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf is supported")

    dest = settings.upload_path / file.filename
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    doc_id = doc_id_for(dest)
    pages = extract_pages(dest)
    if not any(p.strip() for p in pages):
        raise HTTPException(400, "no extractable text — this looks like a scanned PDF")

    title = Path(file.filename).stem.replace("_", " ").title()
    with db.pg() as cur:
        cur.execute(
            """INSERT INTO documents (id, title, filename, page_count)
               VALUES (%s,%s,%s,%s)
               ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title""",
            (doc_id, title, file.filename, len(pages)),
        )
        cur.execute("DELETE FROM pages WHERE doc_id = %s", (doc_id,))
        cur.executemany(
            "INSERT INTO pages (doc_id, page_no, content) VALUES (%s,%s,%s)",
            [(doc_id, i, txt) for i, txt in enumerate(pages, start=1)],
        )

    return DocumentOut(id=doc_id, title=title, filename=file.filename,
                       page_count=len(pages))


@app.get("/api/documents", response_model=list[DocumentOut])
async def list_documents():
    with db.pg() as cur:
        cur.execute(
            """SELECT d.id, d.title, d.filename, d.page_count,
                      (SELECT count(*) FROM naive_chunks n WHERE n.doc_id = d.id) AS chunks
               FROM documents d ORDER BY d.uploaded_at DESC"""
        )
        rows = cur.fetchall()
    return [
        DocumentOut(
            id=r["id"], title=r["title"], filename=r["filename"],
            page_count=r["page_count"],
            naive_ready=r["chunks"] > 0,
        )
        for r in rows
    ]


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    with db.pg() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    with db.neo4j().session() as sess:
        sess.run("MATCH (n) WHERE n.doc_id = $d DETACH DELETE n", d=doc_id)
    return {"deleted": doc_id}


# ------------------------------------------------------------
# Build jobs
# ------------------------------------------------------------
@app.post("/api/documents/{doc_id}/build/{kind}", response_model=JobOut)
async def build(doc_id: str, kind: str, bg: BackgroundTasks):
    if kind != "naive":
        raise HTTPException(400, "kind must be 'naive'")
    doc_pages(doc_id)                      # 404s if unknown
    job_id = jobs.create(doc_id, kind)
    bg.add_task(jobs.run_naive, job_id, doc_id)
    return JobOut(**{**jobs.get(job_id), "detail": {}})


@app.get("/api/jobs/{job_id}", response_model=JobOut)
async def job_status(job_id: str):
    row = jobs.get(job_id)
    if not row:
        raise HTTPException(404, "job not found")
    return JobOut(id=row["id"], doc_id=row["doc_id"], kind=row["kind"],
                  status=row["status"], stage=row["stage"],
                  progress=row["progress"], detail=row["detail"],
                  error=row["error"])


@app.get("/api/documents/{doc_id}/jobs")
async def doc_jobs(doc_id: str):
    return {"naive": jobs.latest(doc_id, "naive")}


# ------------------------------------------------------------
# Query
# ------------------------------------------------------------
class Ask(BaseModel):
    question: str


@app.post("/api/documents/{doc_id}/ask/naive")
async def ask_naive(doc_id: str, body: Ask):
    try:
        return naive_rag.answer(doc_id, body.question)
    except llm.ContentFilterError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc


@app.get("/api/health")
async def health():
    ok_pg = ok_neo = True
    try:
        with db.pg() as cur:
            cur.execute("SELECT 1")
    except Exception:
        ok_pg = False
    try:
        with db.neo4j().session() as s:
            s.run("RETURN 1").consume()
    except Exception:
        ok_neo = False
    return {"postgres": ok_pg, "neo4j": ok_neo,
            "chat_deployment": settings.azure_chat_deployment}
