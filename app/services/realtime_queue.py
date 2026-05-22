"""
realtime_queue.py  (updated — stats tracking)

Changes:
  Added QueueStats dataclass that tracks lifetime counters and a rolling
  window of the last 100 processing latencies. Stats are in-memory and
  reset on process restart — the permanent audit trail is moderation_records.

  New public property: realtime_queue.stats → QueueStats
  Exposed on GET /health as the 'realtime_stats' key.

  Counters:
    enqueued_total   — items received by enqueue() (includes duplicates)
    completed_total  — items that produced a new moderation_records row
    failed_total     — items that raised an unhandled exception in _process_one
    skipped_total    — items where INSERT IGNORE absorbed a duplicate
                       (content already analysed today — existing verdict stands)

  Latency window:
    recent_latencies — deque(maxlen=100) of wall-clock seconds from enqueue()
                       to DB commit. Used to compute avg and p95 on /health.
                       p95 is the most useful single number: it tells you what
                       a typical 'slow' analysis looks like, excluding outliers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

from app.models.schemas import ContentType
from app.models.settings import settings
from app.services.db_client import (
    get_session_factory,
    insert_moderation_record,
    upsert_moderation_queue,
)
from app.services.moderation_service import moderation_service

logger = logging.getLogger(__name__)


@dataclass
class QueueStats:
    """
    In-memory stats for the real-time moderation queue.

    All counters are lifetime values since last process start.
    recent_latencies is a rolling window of the last 100 wall-clock
    processing times (seconds from enqueue to DB commit).
    """

    enqueued_total: int = 0
    completed_total: int = 0
    failed_total: int = 0
    skipped_total: int = 0
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=100))

    def avg_latency_ms(self) -> int:
        """Average latency of recent items in milliseconds. 0 if no data."""
        lats = list(self.recent_latencies)
        return round(sum(lats) / len(lats) * 1000) if lats else 0

    def p95_latency_ms(self) -> int:
        """
        95th percentile latency in milliseconds.
        With maxlen=100 this is the 95th item in the sorted list — i.e. the
        slowest item excluding the top 5%. 0 if fewer than 20 samples.
        """
        lats = sorted(self.recent_latencies)
        if len(lats) < 20:
            return 0
        idx = int(len(lats) * 0.95)
        return round(lats[min(idx, len(lats) - 1)] * 1000)

    def health_pct(self) -> float:
        """
        Percentage of enqueued items that completed successfully (not failed,
        not skipped). Used to show a simple health indicator on the dashboard.
        Returns 100.0 if no items have been enqueued yet.
        """
        if self.enqueued_total == 0:
            return 100.0
        return round(self.completed_total / self.enqueued_total * 100, 1)

    def to_dict(self) -> dict:
        return {
            "enqueued_total": self.enqueued_total,
            "completed_total": self.completed_total,
            "failed_total": self.failed_total,
            "skipped_total": self.skipped_total,
            "avg_latency_ms": self.avg_latency_ms(),
            "p95_latency_ms": self.p95_latency_ms(),
            "health_pct": self.health_pct(),
        }


class RealtimeQueue:
    """
    Lightweight in-process async task queue for real-time moderation.

    Not a persistent queue — items in flight are lost on process restart.
    The periodic reconciliation scan (daily) is the durability guarantee.
    This queue is purely for latency reduction: fresh content is analysed
    within seconds of creation rather than waiting for the next scan.
    """

    def __init__(self) -> None:
        self._active_tasks: set[asyncio.Task] = set()
        self._semaphore = asyncio.Semaphore(settings.realtime_concurrency)
        self._stats = QueueStats()

    @property
    def depth(self) -> int:
        """Current number of tasks in-flight (waiting + running)."""
        return len(self._active_tasks)

    @property
    def stats(self) -> QueueStats:
        """Live stats snapshot. Read-only from callers."""
        return self._stats

    def enqueue(
        self,
        content_id: str,
        content: str,
        author_id: str,
        content_type: ContentType,
        post_id: str | None = None,
    ) -> None:
        """
        Non-blocking enqueue. Returns immediately after creating the asyncio Task.
        Called from the HTTP request handler — must never block.
        """
        # Record receipt time so we can measure end-to-end latency
        enqueued_at = time.monotonic()
        self._stats.enqueued_total += 1

        task = asyncio.create_task(
            self._process_one(content_id, content, author_id, content_type, post_id, enqueued_at),
            name=f"realtime_{content_type}_{content_id[:8]}",
        )
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        task.add_done_callback(self._log_exception)

        logger.debug(
            "RealtimeQueue: enqueued %s %s (depth=%d, total=%d)",
            content_type,
            content_id[:12],
            len(self._active_tasks),
            self._stats.enqueued_total,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _process_one(
        self,
        content_id: str,
        content: str,
        author_id: str,
        content_type: ContentType,
        post_id: str | None,
        enqueued_at: float,
    ) -> None:
        """Analyse one item and write results to MySQL. Records stats on every exit path."""
        async with self._semaphore:
            try:
                result = await moderation_service.moderate(
                    content_id=content_id,
                    content=content,
                    author_id=author_id,
                    content_type=content_type,
                )

                factory = get_session_factory()
                async with factory() as session:
                    record_id = await insert_moderation_record(
                        session=session,
                        comment_id=content_id if content_type == "comment" else None,
                        post_id=post_id or content_id,
                        content_type=content_type,
                        content_id=content_id,
                        author_id=author_id,
                        content=content,
                        verdict=result.verdict,
                        confidence_pct=int(round(result.confidence * 100)),
                        categories=result.categories,
                        explanation=result.explanation,
                        flagged_phrases=result.flaggedPhrases,
                        model=(
                            settings.openrouter_model
                            if settings.moderation_mode in ("openrouter", "hybrid")
                            else settings.moderator_model
                        ),
                        trigger="realtime",
                    )

                    if record_id and result.verdict in ("review", "remove"):
                        await upsert_moderation_queue(
                            session=session,
                            comment_id=(content_id if content_type == "comment" else None),
                            post_id=post_id or content_id,
                            content_type=content_type,
                            content_id=content_id,
                            author_id=author_id,
                            content=content,
                            verdict=result.verdict,
                            confidence_pct=int(round(result.confidence * 100)),
                            explanation=result.explanation,
                            flagged_phrases=result.flaggedPhrases,
                            moderation_record_id=record_id,
                        )
                        logger.info(
                            "RealtimeQueue: %s %s → %s (%.0f%%)",
                            content_type,
                            content_id[:12],
                            result.verdict,
                            result.confidence * 100,
                        )

                    await session.commit()

                # ── Stats recording ───────────────────────────────────────────
                elapsed = time.monotonic() - enqueued_at

                if record_id is None:
                    # INSERT IGNORE skipped — content already analysed today
                    self._stats.skipped_total += 1
                    logger.debug(
                        "RealtimeQueue: %s %s already analysed today — skipped",
                        content_type,
                        content_id[:12],
                    )
                else:
                    self._stats.completed_total += 1
                    self._stats.recent_latencies.append(elapsed)
                    logger.debug(
                        "RealtimeQueue: %s %s completed in %.0fms",
                        content_type,
                        content_id[:12],
                        elapsed * 1000,
                    )

            except Exception as exc:
                self._stats.failed_total += 1
                logger.error(
                    "RealtimeQueue: error processing %s %s: %s",
                    content_type,
                    content_id[:12],
                    exc,
                    exc_info=False,
                )

    @staticmethod
    def _log_exception(task: asyncio.Task) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("RealtimeQueue: unhandled task exception: %s", exc)

    async def shutdown(self) -> None:
        """Cancel all pending tasks. Call during application shutdown."""
        if not self._active_tasks:
            return
        count = len(self._active_tasks)
        logger.info("RealtimeQueue: cancelling %d pending tasks on shutdown", count)
        for task in self._active_tasks:
            task.cancel()
        await asyncio.gather(*self._active_tasks, return_exceptions=True)
        self._active_tasks.clear()
        logger.info("RealtimeQueue: shutdown complete")


# Singleton — shared across all requests in this process
realtime_queue = RealtimeQueue()
