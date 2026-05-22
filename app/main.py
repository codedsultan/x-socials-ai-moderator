"""
x-socials AI Moderation Service v2

Startup:
    cp .env.example .env        # add ANTHROPIC_API_KEY, MONGODB_URI, LARAVEL_DB_URL
    pip install -r requirements.txt
    uvicorn app.main:app --reload --port 8001

Endpoints:
    POST /moderate            — analyse one comment (on-demand, no DB write)
    POST /moderate/batch      — analyse up to 50 concurrently (on-demand, no DB write)
    POST /scan/trigger        — trigger background scan (reads MongoDB, writes Laravel DB)
    GET  /health              — liveness + integration status
    GET  /docs                — Swagger UI
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.models.settings import settings
from app.services.db_client import close_laravel_db, close_mongo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "Moderator starting  model=%s  remove_threshold=%.2f  review_threshold=%.2f",
        settings.moderator_model,
        settings.remove_threshold,
        settings.review_threshold,
    )
    configured = "configured"
    not_set = "NOT SET — /scan/trigger will fail"
    logger.info("MongoDB:    %s", configured if settings.mongodb_uri else not_set)
    logger.info("Laravel DB: %s", configured if settings.laravel_db_url else not_set)
    yield
    await close_mongo()
    await close_laravel_db()
    logger.info("Shutdown complete")


app = FastAPI(
    title="x-socials AI Moderation Service",
    description=(
        "Reads comments from MongoDB, analyses them with Claude, and writes "
        "verdicts to the Laravel admin database. Human review and all control "
        "actions remain in the Laravel admin panel."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", tags=["system"])
async def health() -> dict:
    return {
        "status": "ok",
        "model": settings.moderator_model,
        "thresholds": {
            "remove": settings.remove_threshold,
            "review": settings.review_threshold,
        },
        "integrations": {
            "mongodb": bool(settings.mongodb_uri),
            "laravel_db": bool(settings.laravel_db_url),
        },
    }
