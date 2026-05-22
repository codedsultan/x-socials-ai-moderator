"""
db_client.py

Manages async connections to:
  - MongoDB  (read-only)  — Node.js comment/post data
  - Laravel DB (write)    — moderation_records, moderation_queue, scan_runs

Both connections are lazily initialised on first use and reused
across requests for the lifetime of the process.

Credentials are scoped conservatively:
  MongoDB:    read role on x_socials.comments and x_socials.posts only
  Laravel DB: INSERT on moderation_records, INSERT/UPDATE on moderation_queue,
              UPDATE on scan_runs — no access to admin_users or admin_action_logs
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.models.settings import settings

logger = logging.getLogger(__name__)


# ── MongoDB ───────────────────────────────────────────────────────────────────

_mongo_client: AsyncIOMotorClient | None = None
_mongo_db: AsyncIOMotorDatabase | None = None


def get_mongo_db() -> AsyncIOMotorDatabase:
    global _mongo_client, _mongo_db
    if _mongo_db is None:
        if not settings.mongodb_uri:
            raise RuntimeError(
                "MONGODB_URI is not configured. "
                "Set it in .env to enable direct MongoDB access for scanning."
            )
        _mongo_client = AsyncIOMotorClient(
            settings.mongodb_uri,
            # [Fix 3] Fail fast on bad URI rather than blocking the event loop
            serverSelectionTimeoutMS=5_000,
        )
        _mongo_db = _mongo_client[settings.mongodb_db]
        logger.info("MongoDB connection established (db=%s)", settings.mongodb_db)
    return _mongo_db


async def close_mongo() -> None:
    global _mongo_client, _mongo_db
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
        _mongo_db = None


# ── Laravel DB (SQLAlchemy async) ─────────────────────────────────────────────

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None


def get_session_factory() -> async_sessionmaker:
    global _engine, _session_factory
    if _session_factory is None:
        if not settings.laravel_db_url:
            raise RuntimeError(
                "LARAVEL_DB_URL is not configured. "
                "Set it in .env to enable writing results to the Laravel database."
            )
        _engine = create_async_engine(
            settings.laravel_db_url,
            echo=False,
            # pool_pre_ping=True,
            pool_pre_ping=False,
            pool_recycle=3600,
            pool_size=5,
            max_overflow=10,
        )
        _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
        logger.info("Laravel DB connection pool ready (pool_pre_ping=True)")
    return _session_factory


async def close_laravel_db() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


# ── Laravel DB write helpers ───────────────────────────────────────────────────


async def insert_moderation_record(
    session: AsyncSession,
    comment_id: str | None,
    post_id: str,
    content_type: str,
    content_id: str,
    author_id: str,
    content: str,
    verdict: str,
    confidence_pct: int,
    categories: list[str],
    explanation: str,
    flagged_phrases: list[str],
    model: str,
    trigger: str,
) -> int | None:
    """
    Insert a row into moderation_records.

    [Fix 1] Uses INSERT IGNORE so duplicate rows (same content_id +
    content_type + calendar day) are silently skipped rather than raising
    an IntegrityError that would abort the entire scan batch.

    Returns the new row's auto-increment id, or None if the row was
    a duplicate and was ignored.
    """
    import json as _json

    result = await session.execute(
        text("""
            INSERT IGNORE INTO moderation_records
                (comment_id, post_id, content_type, content_id,
                 author_id, content, verdict,
                 confidence_pct, categories, explanation, flagged_phrases,
                 model, `trigger`, created_at, updated_at)
            VALUES
                (:comment_id, :post_id, :content_type, :content_id,
                 :author_id, :content, :verdict,
                 :confidence_pct, :categories, :explanation, :flagged_phrases,
                 :model, :trigger, :now, :now)
        """),
        {
            "comment_id": comment_id,
            "post_id": post_id,
            "content_type": content_type,
            "content_id": content_id,
            "author_id": author_id,
            "content": content,
            "verdict": verdict,
            "confidence_pct": confidence_pct,
            "categories": _json.dumps(categories),
            "explanation": explanation,
            "flagged_phrases": _json.dumps(flagged_phrases),
            "model": model,
            "trigger": trigger,
            "now": _now(),
        },
    )
    # lastrowid is 0 when INSERT IGNORE skips a duplicate
    row_id = result.lastrowid  # type: ignore[attr-defined]
    return row_id if row_id else None


async def upsert_moderation_queue(
    session: AsyncSession,
    comment_id: str | None,
    post_id: str,
    content_type: str,
    content_id: str,
    author_id: str,
    content: str,
    verdict: str,
    confidence_pct: int,
    explanation: str,
    flagged_phrases: list[str],
    moderation_record_id: int,
) -> None:
    """
    Insert or update a row in moderation_queue.
    Unique constraint is on (content_id, content_type).
    """
    import json as _json

    now = _now()

    dialect = _engine.url.get_dialect().name if _engine else "sqlite"

    if dialect == "sqlite":
        await session.execute(
            text("""
                INSERT INTO moderation_queue
                    (comment_id, post_id, content_type, content_id,
                     author_id, content, verdict,
                     confidence_pct, explanation, flagged_phrases, status,
                     resolved_by, resolved_at, resolution_note,
                     moderation_record_id, created_at, updated_at)
                VALUES
                    (:comment_id, :post_id, :content_type, :content_id,
                     :author_id, :content, :verdict,
                     :confidence_pct, :explanation, :flagged_phrases, 'pending',
                     NULL, NULL, NULL,
                     :record_id, :now, :now)
                ON CONFLICT(content_id, content_type) DO UPDATE SET
                    verdict              = excluded.verdict,
                    confidence_pct       = excluded.confidence_pct,
                    explanation          = excluded.explanation,
                    flagged_phrases      = excluded.flagged_phrases,
                    status               = 'pending',
                    resolved_by          = NULL,
                    resolved_at          = NULL,
                    moderation_record_id = excluded.moderation_record_id,
                    updated_at           = excluded.updated_at
            """),
            {
                "comment_id": comment_id,
                "post_id": post_id,
                "content_type": content_type,
                "content_id": content_id,
                "author_id": author_id,
                "content": content,
                "verdict": verdict,
                "confidence_pct": confidence_pct,
                "explanation": explanation,
                "flagged_phrases": _json.dumps(flagged_phrases),
                "record_id": moderation_record_id,
                "now": now,
            },
        )
    else:
        # MySQL
        await session.execute(
            text("""
                INSERT INTO moderation_queue
                    (comment_id, post_id, content_type, content_id,
                     author_id, content, verdict,
                     confidence_pct, explanation, flagged_phrases, status,
                     resolved_by, resolved_at, resolution_note,
                     moderation_record_id, created_at, updated_at)
                VALUES
                    (:comment_id, :post_id, :content_type, :content_id,
                     :author_id, :content, :verdict,
                     :confidence_pct, :explanation, :flagged_phrases, 'pending',
                     NULL, NULL, NULL,
                     :record_id, :now, :now)
                ON DUPLICATE KEY UPDATE
                    verdict              = VALUES(verdict),
                    confidence_pct       = VALUES(confidence_pct),
                    explanation          = VALUES(explanation),
                    flagged_phrases      = VALUES(flagged_phrases),
                    status               = 'pending',
                    resolved_by          = NULL,
                    resolved_at          = NULL,
                    moderation_record_id = VALUES(moderation_record_id),
                    updated_at           = VALUES(updated_at)
            """),
            {
                "comment_id": comment_id,
                "post_id": post_id,
                "content_type": content_type,
                "content_id": content_id,
                "author_id": author_id,
                "content": content,
                "verdict": verdict,
                "confidence_pct": confidence_pct,
                "explanation": explanation,
                "flagged_phrases": _json.dumps(flagged_phrases),
                "record_id": moderation_record_id,
                "now": now,
            },
        )


async def create_scan_run(session: AsyncSession) -> int:
    result = await session.execute(
        text(
            "INSERT INTO scan_runs (status, posts_scanned, comments_scanned,"
            " flagged, queued_for_review, safe, started_at)"
            " VALUES ('running', 0, 0, 0, 0, 0, :now)"
        ),
        {"now": _now()},
    )
    return result.lastrowid  # type: ignore[attr-defined]


async def complete_scan_run(session: AsyncSession, run_id: int, counts: dict[str, int]) -> None:
    await session.execute(
        text("""
            UPDATE scan_runs SET
                status            = 'completed',
                posts_scanned     = :posts,
                comments_scanned  = :comments,
                flagged           = :flagged,
                queued_for_review = :review,
                safe              = :safe,
                finished_at       = :now
            WHERE id = :id
        """),
        {
            "id": run_id,
            "posts": counts.get("posts_scanned", 0),
            "comments": counts.get("comments_scanned", 0),
            "flagged": counts.get("flagged", 0),
            "review": counts.get("queued_for_review", 0),
            "safe": counts.get("safe", 0),
            "now": _now(),
        },
    )


async def fail_scan_run(session: AsyncSession, run_id: int, error: str) -> None:
    await session.execute(
        text(
            "UPDATE scan_runs SET status='failed', error_message=:err, finished_at=:now"
            " WHERE id=:id"
        ),
        {"id": run_id, "err": error[:1000], "now": _now()},
    )


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
