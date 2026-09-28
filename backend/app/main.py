from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from sqlmodel import SQLModel

from app.api.main import api_router
from app.core.config import logger, settings
from app.core.db import engine
from app.core.exception_handlers import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.services.llm_service import llm_service
from app.services.metrics_service import metrics_service
from app.services.redis_service import redis_service

# Import models so SQLModel metadata is complete before create_all.
from app.models.conversation_model import Conversation
from app.models.document_model import Document
from app.models.message_model import Message
from app.models.user_model import Item, User
from app.services.search_service import search_service
from app.services.embedding_service import embedding_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"Starting API | environment={settings.ENVIRONMENT} "
        f"| version={settings.API_VERSION}"
    )

    if not settings.TESTING and settings.AUTO_CREATE_TABLES:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(SQLModel.metadata.create_all)

        logger.info(
            "Database tables and pgvector extension verified/created."
        )

    if settings.TESTING:
        yield
        await engine.dispose()
        return

    if redis_service.ping():
        logger.info("Redis connectivity verified.")
    else:
        logger.warning(
            "Redis unavailable; rate limiting will fail open."
        )

    # Warm up retrieval infrastructure.
    try:
        embedding_service.get_embeddings()
        search_service.warmup()

        logger.success("RAG retrieval infrastructure warmed up.")
    except Exception as exc:
        logger.exception(
            f"RAG warmup failed | error={exc}"
        )

    yield

    await engine.dispose()
    logger.info("API stopped.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description=(
        "Enterprise document management, semantic search and "
        "retrieval-augmented question answering."
    ),
    lifespan=lifespan,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Compresses JSON/text responses over 1KB (chat answers, document
# lists, search results) -- meaningful bandwidth savings in production
# with negligible CPU cost.
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


@app.middleware("http")
async def request_metrics_middleware(request: Request, call_next):
    started = perf_counter()
    try:
        response = await call_next(request)
        metrics_service.increment("http_requests_total")
        if response.status_code >= 400:
            metrics_service.increment("http_errors_total")
        return response
    finally:
        elapsed = (perf_counter() - started) * 1000
        logger.debug(
            f"HTTP request | method={request.method} | path={request.url.path} "
            f"| duration={elapsed:.2f} ms"
        )


@app.get("/", tags=["Health"])
async def root():
    return {"status": "healthy", "service": settings.PROJECT_NAME}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "service": settings.PROJECT_NAME}


@app.get("/ready", tags=["Health"])
async def readiness():
    database = "unhealthy"
    redis = "healthy" if redis_service.ping() else "unhealthy"
    llm = "healthy" if await llm_service.health() else "unhealthy"

    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        database = "healthy"
    except Exception as exc:
        logger.error(f"Database readiness failed | error={exc}")

    ready = database == "healthy" and redis == "healthy" and llm == "healthy"
    payload = {
        "status": "ready" if ready else "not_ready",
        "service": settings.PROJECT_NAME,
        "dependencies": {
            "database": database,
            "redis": redis,
            f"llm ({settings.LLM_PROVIDER})": llm,
        },
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)


@app.get("/metrics", tags=["Health"])
async def metrics():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "counters": metrics_service.snapshot(),
    }


app.include_router(api_router, prefix=settings.API_V1_STR)
