# Document Intelligence RAG Platform

A production-oriented document question-answering application built around FastAPI, PostgreSQL/PGVector, Redis, local BGE embeddings, Ollama, and a React + TypeScript frontend.

## What it demonstrates

- JWT authentication and user-scoped document retrieval
- PDF upload, SHA-256 duplicate detection, extraction, chunking and indexing
- PGVector semantic retrieval with MMR and lightweight lexical reranking
- Context-bounded RAG prompts with source/page citations
- Persisted conversations and chat history
- Redis-backed request rate limiting
- Lazy singleton model/vector-store initialization
- Bounded Ollama concurrency and configurable local inference
- Health/readiness/metrics endpoints
- Focused backend tests with external AI dependencies mocked where appropriate
- Responsive React/Vite frontend integrated with the FastAPI API

## Architecture

```text
React + TypeScript
       |
       | JWT / JSON / multipart
       v
FastAPI
  |-- Auth --------------------> PostgreSQL
  |-- Conversations ------------> PostgreSQL
  |-- Documents -> PDF -> chunks -> BGE embeddings -> PGVector
  |-- Search -------------------> PGVector
  |-- QA -> history + retrieval -> bounded context -> Ollama
  |                                      |
  |                                      +--> source citations
  +-- Rate limiting ------------> Redis
```

## Backend setup

From `backend/`:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

pip install -r ../requirements.txt
docker compose up -d postgres redis
python -m app.init_db
uvicorn app.main:app --reload
```

Swagger is available at `http://localhost:8000/api/v1/docs`.

Start Ollama separately:

```bash
ollama serve
ollama pull qwen2.5:7b
```

Copy `.env.example` to `.env` and configure credentials/secrets.

## Frontend setup

From `frontend/`:

```bash
npm install
npm run dev
```

The frontend defaults to `http://localhost:8000/api/v1`. Override it with:

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

The UI supports login, conversations, PDF upload/delete, grounded chat with sources, and semantic search.

## Retrieval pipeline

1. Authenticate the user.
2. Retrieve only vectors owned by that user (unless the caller is a superuser).
3. Run MMR or similarity retrieval with a bounded candidate set.
4. Apply lightweight lexical reranking.
5. Remove duplicate chunks.
6. Bound the final context by configured document count and character budget.
7. Ask Ollama to answer only from the supplied context.
8. Return source document/page metadata with the answer.

## Performance considerations

The embedding model and PGVector client are cached per process and warmed during application startup. LLM requests (Ollama or Groq, depending on `LLM_PROVIDER`) are reused through one client and bounded to the configured concurrency. Retrieval and RAG timings are logged independently so local hardware/model performance can be measured rather than guessed.

The default local model is `qwen2.5:7b` via Ollama; generation speed depends strongly on available CPU/GPU memory and Ollama's actual offloading. The application therefore exposes model, context, token, thread and GPU-layer settings through environment variables. Switching `LLM_PROVIDER=groq` trades that local-hardware dependency for network-bound, typically sub-second responses.

## LLM provider: Ollama vs Groq

Which model answers questions is controlled by `LLM_PROVIDER` in `.env` — both code paths always exist in `app/services/llm_service.py`; this only picks which one gets built at runtime.

**`LLM_PROVIDER=ollama` (default)** — self-hosted local inference. Use this for local development, or if you're running this project open-source against your own GPU/machine. Requires Ollama running locally (see Backend setup above) and the `OLLAMA_*` variables in `.env`.

**`LLM_PROVIDER=groq`** — hosted inference via [Groq](https://console.groq.com/keys)'s free API (no credit card required, generous free tier, sub-second responses). This is what the publicly deployed demo runs, since a live demo needs to stay fast and always-on without a self-hosted GPU/VM running 24/7. To use it:
```bash
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```
No other code changes needed — `/ready` and title generation both work identically under either provider.

## LangSmith setup

Tracing is optional and off by default, and works the same way regardless of which `LLM_PROVIDER` is active. To enable it:

1. Create a LangSmith account and API key at https://smith.langchain.com.
2. In `.env`, set:
   ```bash
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=ls__your_key_here
   LANGSMITH_PROJECT=enterprise-rag-api
   ```
3. Restart the API. `app/core/config.py` populates the standard `LANGCHAIN_*`
   environment variables at process startup, which LangChain's runnables
   (embeddings, retriever, `ChatOllama`/`ChatGroq`) pick up automatically —
   no code changes are required elsewhere. Traces for ingestion, retrieval
   and QA calls will appear under the configured project in the LangSmith UI.

Leave `LANGSMITH_API_KEY` blank to keep tracing disabled (default).

## Tests

From `backend/`:

```bash
pytest
```

The test suite covers health endpoints, JWT/password helpers and retrieval ranking helpers. AI/vector infrastructure should be mocked for API integration tests so tests do not require a running Ollama instance.

## Docker

The included compose file provisions PostgreSQL with pgvector and Redis for local development. Ollama is intentionally kept as a host dependency because GPU passthrough is environment-specific.

## Security notes

- Never commit `.env` or real credentials.
- Normal users are restricted to their own document vectors.
- Uploaded filenames are normalized to their basename before storage.
- Upload size and PDF signature are validated.
- Duplicate uploads are rejected per user using SHA-256.
- API errors expose safe messages while server logs retain diagnostics.

## Production considerations

For deployment, use real database migrations rather than automatic table creation, TLS/reverse proxying, secret management, private networks for PostgreSQL/Redis/Ollama, centralized logs/metrics, durable document storage, backups, and an appropriate process/container manager.
