CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- Shared: documents
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    filename      TEXT NOT NULL,
    page_count    INT  NOT NULL,
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pages (
    doc_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no       INT  NOT NULL,
    content       TEXT NOT NULL,
    PRIMARY KEY (doc_id, page_no)
);

-- ============================================================
-- PART A: naive RAG
-- ============================================================
CREATE TABLE IF NOT EXISTS naive_chunks (
    id            BIGSERIAL PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INT  NOT NULL,
    page_start    INT  NOT NULL,
    page_end      INT  NOT NULL,
    content       TEXT NOT NULL,
    embedding     vector(1536)
);

CREATE INDEX IF NOT EXISTS naive_chunks_doc_idx ON naive_chunks (doc_id);
CREATE INDEX IF NOT EXISTS naive_chunks_vec_idx
    ON naive_chunks USING hnsw (embedding vector_cosine_ops);

-- ============================================================
-- PART B: extracted timeline events
-- ============================================================
CREATE TABLE IF NOT EXISTS events (
    id                 TEXT PRIMARY KEY,
    doc_id             TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    event_name         TEXT NOT NULL,
    category           TEXT NOT NULL CHECK (category IN ('major','minor')),
    chronological_clue TEXT NOT NULL DEFAULT '',
    topological_order  INT  NOT NULL DEFAULT 0,
    location           TEXT NOT NULL DEFAULT '',
    characters         TEXT[] NOT NULL DEFAULT '{}',
    core_event         TEXT NOT NULL,
    antecedent_cause   TEXT NOT NULL DEFAULT '',
    consequent_effect  TEXT NOT NULL DEFAULT '',
    source_pages       INT[] NOT NULL DEFAULT '{}',
    first_page         INT  NOT NULL DEFAULT 0,
    merge_count        INT  NOT NULL DEFAULT 1,
    embedding          vector(1536)
);

CREATE INDEX IF NOT EXISTS events_doc_idx   ON events (doc_id);
CREATE INDEX IF NOT EXISTS events_order_idx ON events (doc_id, topological_order, first_page);
CREATE INDEX IF NOT EXISTS events_vec_idx
    ON events USING hnsw (embedding vector_cosine_ops);

-- Raw pass-1 observations, kept for auditability and the "show your work" panel
CREATE TABLE IF NOT EXISTS observations (
    id            BIGSERIAL PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    window_id     TEXT NOT NULL,
    page_start    INT  NOT NULL,
    page_end      INT  NOT NULL,
    raw_text      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS observations_doc_idx ON observations (doc_id);

-- ============================================================
-- Job tracking (drives the frontend progress UI)
-- ============================================================
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    kind          TEXT NOT NULL,            -- 'naive' | 'kaalkram'
    status        TEXT NOT NULL,            -- 'queued'|'running'|'done'|'error'
    stage         TEXT NOT NULL DEFAULT '',
    progress      REAL NOT NULL DEFAULT 0,  -- 0..1
    detail        JSONB NOT NULL DEFAULT '{}'::jsonb,
    error         TEXT,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS jobs_doc_idx ON jobs (doc_id, kind);

-- ============================================================
-- Saved comparison runs (for the metrics screen)
-- ============================================================
CREATE TABLE IF NOT EXISTS query_runs (
    id            BIGSERIAL PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    question      TEXT NOT NULL,
    naive_answer  TEXT,
    naive_ms      INT,
    naive_tokens  INT,
    kaal_answer   TEXT,
    kaal_ms       INT,
    kaal_tokens   INT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
