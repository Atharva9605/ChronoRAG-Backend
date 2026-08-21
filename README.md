# ChronoRAG / Kaalkram Backend

FastAPI API: naive RAG + Kaalkram timeline pipeline.

## Setup

```bash
cp .env.example .env   # fill Azure OpenAI + DB settings
docker compose up -d   # Postgres (pgvector) + Neo4j
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs
