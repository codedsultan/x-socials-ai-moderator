"""
scan_service.py  (updated — reconciliation mode)

Changes:
  trigger() now accepts a reconciliation: bool flag.
  When True, the scan uses settings.reconciliation_lookback_h instead of
  settings.scan_lookback_h, turning the scan into a gap-filling sweep
  rather than the primary analysis pipeline.

  With real-time webhooks in place:
    Standard scan   — runs on-demand or manually, scan_lookback_h=1h
    Reconciliation  — runs daily via Laravel scheduler, looks back 48h,
                      catches anything the webhook dropped
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from app.models.schemas import ContentType
from app.models.settings import settings
from app.services.db_client import (
    complete_scan_run,
    create_scan_run,
    fail_scan_run,
    get_mongo_db,
    get_session_factory,
    insert_moderation_record,
    upsert_moderation_queue,
)
from app.services.moderation_service import moderation_service

logger = logging.getLogger(__name__)


class ScanService:

    def __init__(self) -> None:
        self._active_scans: set[asyncio.Task] = set()

    async def trigger(
        self,
        post_id:        str | None          = None,
        force_model:    str | None          = None,
        content_type:   ContentType | None  = None,
        reconciliation: bool                = False,
    ) -> int:
        """
        Start a background scan and return the scan_run_id immediately.

        reconciliation=True uses the longer reconciliation_lookback_h window
        to sweep for items the real-time webhook may have missed.
        reconciliation=False (default) uses scan_lookback_h for a short
        targeted scan of recent content.
        """
        factory = get_session_factory()
        async with factory() as session:
            run_id = await create_scan_run(session)
            await session.commit()

        task = asyncio.create_task(
            self._run(run_id, post_id, force_model, content_type, reconciliation),
            name=f"scan_run_{run_id}",
        )

        self._active_scans.add(task)
        task.add_done_callback(self._active_scans.discard)

        def log_exception(task: asyncio.Task) -> None:
            try:
                task.exception()
            except asyncio.CancelledError:
                logger.info("Scan %d was cancelled", run_id)
            except Exception as e:
                logger.error("Unhandled exception in scan %d: %s", run_id, e, exc_info=True)

        task.add_done_callback(log_exception)

        return run_id

    # ── Internal pipeline ─────────────────────────────────────────────────────

    async def _run(
        self,
        run_id:         int,
        post_id:        str | None,
        force_model:    str | None,
        content_type:   ContentType | None,
        reconciliation: bool,
    ) -> None:
        counts = {
            "posts_scanned":     0,
            "comments_scanned":  0,
            "flagged":           0,
            "queued_for_review": 0,
            "safe":              0,
        }

        factory      = get_session_factory()
        current_task = asyncio.current_task()

        # Use the longer lookback for reconciliation sweeps
        lookback_h = (
            settings.reconciliation_lookback_h if reconciliation
            else settings.scan_lookback_h
        )

        try:
            if current_task and current_task.cancelled():
                return

            db = get_mongo_db()

            if post_id:
                post_ids = [post_id]
            else:
                cutoff = datetime.now(UTC) - timedelta(hours=lookback_h)
                unique_post_ids: set[str] = set()

                comment_cursor = db.comments.find(
                    {"createdAt": {"$gte": cutoff}},
                    {"postId": 1}
                ).limit(5000)

                async for doc in comment_cursor:
                    if current_task and current_task.cancelled():
                        return
                    unique_post_ids.add(str(doc["postId"]))

                post_cursor = db.posts.find(
                    {"createdAt": {"$gte": cutoff}, "deletedAt": None},
                    {"_id": 1}
                ).limit(2000)

                async for doc in post_cursor:
                    if current_task and current_task.cancelled():
                        return
                    unique_post_ids.add(str(doc["_id"]))

                post_ids = list(unique_post_ids)

            counts["posts_scanned"] = len(post_ids)
            logger.info(
                "Scan %d: %d posts, content_type=%s, mode=%s (lookback=%dh)",
                run_id, len(post_ids), content_type or "both",
                "reconciliation" if reconciliation else "standard",
                lookback_h,
            )

            processed = 0
            for idx, pid in enumerate(post_ids):
                if current_task and current_task.cancelled():
                    async with factory() as session:
                        await fail_scan_run(session, run_id, f"Cancelled after {idx} posts")
                        await session.commit()
                    return

                try:
                    stats = await self._scan_one_post(pid, factory, force_model, content_type)
                    counts["comments_scanned"]  += stats["comments_scanned"]
                    counts["flagged"]           += stats["flagged"]
                    counts["queued_for_review"] += stats["review"]
                    counts["safe"]              += stats["safe"]
                    processed += 1

                    if processed % 10 == 0 or processed == len(post_ids):
                        logger.info(
                            "Scan %d progress: %d/%d posts (flagged=%d, review=%d, safe=%d)",
                            run_id, processed, len(post_ids),
                            counts["flagged"], counts["queued_for_review"], counts["safe"],
                        )

                except Exception as post_error:
                    logger.error("Scan %d error on post %s: %s", run_id, pid, post_error)
                    continue

            async with factory() as session:
                await complete_scan_run(session, run_id, counts)
                await session.commit()

            logger.info("Scan %d completed: %s", run_id, counts)

        except asyncio.CancelledError:
            try:
                async with factory() as session:
                    await fail_scan_run(session, run_id, "Cancelled by system shutdown")
                    await session.commit()
            except Exception:
                pass
            raise

        except Exception as exc:
            logger.exception("Scan %d failed: %s", run_id, exc)
            try:
                async with factory() as session:
                    await fail_scan_run(session, run_id, str(exc)[:1000])
                    await session.commit()
            except Exception:
                pass

    async def _scan_one_post(
        self,
        post_id:      str,
        factory:      Any,
        force_model:  str | None,
        content_type: ContentType | None,
    ) -> dict[str, int]:
        counts = {"comments_scanned": 0, "flagged": 0, "review": 0, "safe": 0}

        if content_type != "comment":
            s = await self._scan_post_content(post_id, factory, force_model)
            counts["flagged"] += s["flagged"]
            counts["review"]  += s["review"]
            counts["safe"]    += s["safe"]

        if content_type != "post":
            s = await self._scan_comments(post_id, factory, force_model)
            counts["comments_scanned"] += s["scanned"]
            counts["flagged"]          += s["flagged"]
            counts["review"]           += s["review"]
            counts["safe"]             += s["safe"]

        return counts

    async def _scan_post_content(
        self,
        post_id:     str,
        factory:     Any,
        force_model: str | None,
    ) -> dict[str, int]:
        counts = {"flagged": 0, "review": 0, "safe": 0}
        db = get_mongo_db()

        post_doc = await db.posts.find_one({"_id": post_id, "deletedAt": None})
        if not post_doc:
            return counts

        already = await self._already_analysed([post_id], "post", factory)
        if post_id in already:
            return counts

        title   = post_doc.get("title", "")
        body    = post_doc.get("content", "")
        content = f"Title: {title}\n\nBody:\n{body}".strip()
        if not content:
            return counts

        author_id = str(post_doc.get("authorId", ""))

        result = await moderation_service.moderate(
            content_id=post_id, content=content,
            author_id=author_id, model=force_model, content_type="post",
        )

        async with factory() as session:
            record_id = await insert_moderation_record(
                session=session, comment_id=None, post_id=post_id,
                content_type="post", content_id=post_id, author_id=author_id,
                content=content, verdict=result.verdict,
                confidence_pct=int(round(result.confidence * 100)),
                categories=result.categories, explanation=result.explanation,
                flagged_phrases=result.flaggedPhrases,
                model=force_model or settings.moderator_model, trigger="auto",
            )

            if record_id and result.verdict in ("review", "remove"):
                await upsert_moderation_queue(
                    session=session, comment_id=None, post_id=post_id,
                    content_type="post", content_id=post_id, author_id=author_id,
                    content=content, verdict=result.verdict,
                    confidence_pct=int(round(result.confidence * 100)),
                    explanation=result.explanation, flagged_phrases=result.flaggedPhrases,
                    moderation_record_id=record_id,
                )

            if result.verdict == "remove":
                counts["flagged"] += 1
            elif result.verdict == "review":
                counts["review"] += 1
            else:
                counts["safe"] += 1

            await session.commit()

        return counts

    async def _scan_comments(
        self,
        post_id:     str,
        factory:     Any,
        force_model: str | None,
    ) -> dict[str, int]:
        counts = {"scanned": 0, "flagged": 0, "review": 0, "safe": 0}
        db = get_mongo_db()

        cursor   = db.comments.find({"postId": post_id, "deletedAt": None})
        comments: list[dict] = []
        async for doc in cursor:
            comments.append({
                "id":       str(doc["_id"]),
                "content":  doc.get("content", ""),
                "authorId": str(doc.get("authorId", "")),
                "postId":   post_id,
            })

        if not comments:
            return counts

        already_done = await self._already_analysed(
            [c["id"] for c in comments], "comment", factory
        )
        new_comments = [c for c in comments if c["id"] not in already_done]

        if not new_comments:
            return counts

        for i in range(0, len(new_comments), settings.scan_batch_size):
            batch = new_comments[i : i + settings.scan_batch_size]
            stats = await self._analyse_and_store_comments(batch, post_id, factory, force_model)
            counts["scanned"]  += stats["scanned"]
            counts["flagged"]  += stats["flagged"]
            counts["review"]   += stats["review"]
            counts["safe"]     += stats["safe"]

        return counts

    async def _already_analysed(
        self,
        ids:          list[str],
        content_type: str,
        factory:      Any,
    ) -> set[str]:
        if not ids:
            return set()

        today_start = datetime.now(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).strftime("%Y-%m-%d %H:%M:%S")

        placeholders = ",".join([f":id{i}" for i in range(len(ids))])
        params       = {f"id{i}": v for i, v in enumerate(ids)}
        params["today"]        = today_start
        params["content_type"] = content_type

        async with factory() as session:
            result = await session.execute(
                text(f"""
                    SELECT content_id FROM moderation_records
                    WHERE content_id IN ({placeholders})
                      AND content_type = :content_type
                      AND created_at >= :today
                """),
                params,
            )
            return {row[0] for row in result.fetchall()}

    async def _analyse_and_store_comments(
        self,
        comments:    list[dict],
        post_id:     str,
        factory:     Any,
        force_model: str | None,
    ) -> dict[str, int]:
        stats = {"scanned": len(comments), "flagged": 0, "review": 0, "safe": 0}

        results = await moderation_service.moderate_batch(
            items=comments, model=force_model, content_type="comment",
        )

        comment_map = {c["id"]: c for c in comments}

        async with factory() as session:
            for result in results:
                comment = comment_map.get(result.id)
                if not comment:
                    continue

                record_id = await insert_moderation_record(
                    session=session, comment_id=result.id, post_id=post_id,
                    content_type="comment", content_id=result.id,
                    author_id=comment.get("authorId", ""),
                    content=comment.get("content", ""),
                    verdict=result.verdict,
                    confidence_pct=int(round(result.confidence * 100)),
                    categories=result.categories, explanation=result.explanation,
                    flagged_phrases=result.flaggedPhrases,
                    model=force_model or settings.moderator_model, trigger="auto",
                )

                if record_id and result.verdict in ("review", "remove"):
                    await upsert_moderation_queue(
                        session=session, comment_id=result.id, post_id=post_id,
                        content_type="comment", content_id=result.id,
                        author_id=comment.get("authorId", ""),
                        content=comment.get("content", ""),
                        verdict=result.verdict,
                        confidence_pct=int(round(result.confidence * 100)),
                        explanation=result.explanation,
                        flagged_phrases=result.flaggedPhrases,
                        moderation_record_id=record_id,
                    )

                if result.verdict == "remove":
                    stats["flagged"] += 1
                elif result.verdict == "review":
                    stats["review"] += 1
                else:
                    stats["safe"] += 1

            await session.commit()

        return stats

    async def shutdown(self) -> None:
        if not self._active_scans:
            return
        logger.info("Cancelling %d active scan tasks", len(self._active_scans))
        for task in self._active_scans:
            task.cancel()
        await asyncio.gather(*self._active_scans, return_exceptions=True)
        self._active_scans.clear()


scan_service = ScanService()
