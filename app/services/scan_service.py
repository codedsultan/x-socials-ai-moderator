"""
scan_service.py

The background scan pipeline. Reads both posts and comments from MongoDB,
analyses them with Claude, and writes results to the Laravel database.

Content coverage:
  Posts    — title + body analysed as one text block with the POST_PROMPT.
  Comments — content analysed individually with the COMMENT_PROMPT.

Both result types write to moderation_records and moderation_queue.
The content_type column distinguishes them. AutoRemoveCommand branches
on content_type to call the right Node.js admin endpoint.

Call paths:
  - Automatic: Laravel scheduler fires POST /scan/trigger every 30 min
  - Manual full:        POST /scan/trigger {}
  - Manual single-post: POST /scan/trigger { "post_id": "..." }
  - Content filter:     POST /scan/trigger { "content_type": "post" | "comment" }
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.models.schemas import ContentType
from app.models.settings import settings
from app.services.moderation_service import moderation_service
from app.services.db_client import (
    get_mongo_db,
    get_session_factory,
    insert_moderation_record,
    upsert_moderation_queue,
    create_scan_run,
    complete_scan_run,
    fail_scan_run,
)
from sqlalchemy import text

logger = logging.getLogger(__name__)


class ScanService:

    async def trigger(
        self,
        post_id:      str | None          = None,
        force_model:  str | None          = None,
        content_type: ContentType | None  = None,  # None = scan both posts and comments
    ) -> int:
        """
        Start a background scan and return the scan_run_id immediately.
        content_type=None means scan both posts and comments.
        content_type='post' scans only post content (not comments).
        content_type='comment' scans only comments (original behaviour).
        """
        factory = get_session_factory()
        async with factory() as session:
            run_id = await create_scan_run(session)
            await session.commit()
        
        # Create and track the background task
        task = asyncio.create_task(
            self._run(run_id, post_id, force_model, content_type),
            name=f"scan_run_{run_id}"
        )
        
        # Track the task for cleanup on shutdown
        if not hasattr(self, '_active_scans'):
            self._active_scans = set()
        self._active_scans.add(task)
        task.add_done_callback(self._active_scans.discard)
        
        # Add error logging for unhandled exceptions
        def log_exception(task: asyncio.Task) -> None:
            try:
                task.exception()  # This will raise if there was an exception
            except asyncio.CancelledError:
                logger.info("Scan %d was cancelled", run_id)
            except Exception as e:
                logger.error("Unhandled exception in scan %d: %s", run_id, e, exc_info=True)
        
        task.add_done_callback(log_exception)
        
        return run_id

    # ── Internal pipeline ─────────────────────────────────────────────────────

    async def _run(
        self,
        run_id:       int,
        post_id:      str | None,
        force_model:  str | None,
        content_type: ContentType | None,
    ) -> None:
        """
        Background scan execution with cancellation support and proper error handling.
        """
        counts = {
            "posts_scanned":     0,
            "comments_scanned":  0,
            "flagged":           0,
            "queued_for_review": 0,
            "safe":              0,
        }
        
        factory = get_session_factory()
        current_task = asyncio.current_task()
        
        try:
            # Check for cancellation before starting any work
            if current_task and current_task.cancelled():
                logger.info("Scan %d cancelled before starting", run_id)
                return
            
            db = get_mongo_db()
            
            # Determine which posts to scan
            if post_id:
                post_ids = [post_id]
            else:
                # Full scan — find posts with recent activity
                cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.scan_lookback_h)
                
                # Collect post IDs from recent comments
                unique_post_ids: set[str] = set()
                
                # Process comments with a timeout and cancellation check
                comment_cursor = db.comments.find(
                    {"createdAt": {"$gte": cutoff}},
                    {"postId": 1}
                ).limit(5000)
                
                async for doc in comment_cursor:
                    # Check cancellation periodically
                    if current_task and current_task.cancelled():
                        logger.info("Scan %d cancelled while collecting comments", run_id)
                        return
                    unique_post_ids.add(str(doc["postId"]))
                
                # Also include posts created recently (may have no comments yet)
                post_cursor = db.posts.find(
                    {"createdAt": {"$gte": cutoff}, "deletedAt": None},
                    {"_id": 1}
                ).limit(2000)
                
                async for doc in post_cursor:
                    # Check cancellation periodically
                    if current_task and current_task.cancelled():
                        logger.info("Scan %d cancelled while collecting posts", run_id)
                        return
                    unique_post_ids.add(str(doc["_id"]))
                
                post_ids = list(unique_post_ids)
            
            counts["posts_scanned"] = len(post_ids)
            logger.info("Scan %d: %d posts, content_type=%s", run_id, len(post_ids), content_type or "both")
            
            # Process posts with batch progress tracking
            processed = 0
            for idx, pid in enumerate(post_ids):
                # Check for cancellation every iteration
                if current_task and current_task.cancelled():
                    logger.info("Scan %d cancelled mid-execution after %d posts", run_id, idx)
                    # Update database to show cancellation
                    async with factory() as session:
                        await fail_scan_run(session, run_id, f"Cancelled after processing {idx} of {len(post_ids)} posts")
                        await session.commit()
                    return
                
                try:
                    stats = await self._scan_one_post(pid, factory, force_model, content_type)
                    counts["comments_scanned"]  += stats["comments_scanned"]
                    counts["flagged"]           += stats["flagged"]
                    counts["queued_for_review"] += stats["review"]
                    counts["safe"]              += stats["safe"]
                    processed += 1
                    
                    # Log progress every 10 posts or at completion
                    if processed % 10 == 0 or processed == len(post_ids):
                        logger.info("Scan %d progress: %d/%d posts processed (flagged=%d, review=%d)", 
                                run_id, processed, len(post_ids), counts["flagged"], counts["queued_for_review"])
                    
                except Exception as post_error:
                    # Log but continue with other posts
                    logger.error("Scan %d error processing post %s: %s", run_id, pid, post_error, exc_info=False)
                    # Still count the post as scanned
                    counts["posts_scanned"] = max(counts["posts_scanned"], len(post_ids))
                    continue
            
            # Complete the scan run
            async with factory() as session:
                await complete_scan_run(session, run_id, counts)
                await session.commit()
            
            logger.info("Scan %d completed successfully: %s", run_id, counts)
            
        except asyncio.CancelledError:
            logger.info("Scan %d was cancelled", run_id)
            # Update database to show cancellation
            try:
                async with factory() as session:
                    await fail_scan_run(session, run_id, "Scan was cancelled by system shutdown")
                    await session.commit()
            except Exception as db_error:
                logger.error("Scan %d failed to log cancellation to DB: %s", run_id, db_error)
            raise  # Re-raise to properly signal cancellation
            
        except Exception as exc:
            logger.exception("Scan %d failed with error: %s", run_id, exc)
            try:
                async with factory() as session:
                    error_msg = str(exc)[:1000]  # Truncate to fit database column
                    await fail_scan_run(session, run_id, error_msg)
                    await session.commit()
            except Exception as db_error:
                logger.error("Scan %d failed to log error to DB: %s (original error: %s)", 
                            run_id, db_error, exc)
            
        finally:
            # Cleanup if needed
            pass

    async def _scan_one_post(
        self,
        post_id:      str,
        factory:      Any,
        force_model:  str | None,
        content_type: ContentType | None,
    ) -> dict[str, int]:
        """Scan a post's own content and/or its comments, depending on content_type."""
        counts = {"comments_scanned": 0, "flagged": 0, "review": 0, "safe": 0}

        # Scan the post itself (unless caller only wants comments)
        if content_type != "comment":
            post_stats = await self._scan_post_content(post_id, factory, force_model)
            counts["flagged"] += post_stats["flagged"]
            counts["review"]  += post_stats["review"]
            counts["safe"]    += post_stats["safe"]

        # Scan comments on the post (unless caller only wants posts)
        if content_type != "post":
            comment_stats = await self._scan_comments(post_id, factory, force_model)
            counts["comments_scanned"] += comment_stats["scanned"]
            counts["flagged"]          += comment_stats["flagged"]
            counts["review"]           += comment_stats["review"]
            counts["safe"]             += comment_stats["safe"]

        return counts

    # ── Post content scan ─────────────────────────────────────────────────────

    async def _scan_post_content(
        self,
        post_id:     str,
        factory:     Any,
        force_model: str | None,
    ) -> dict[str, int]:
        """
        Fetch the post document and analyse title + content as a single text block.
        Uses the POST_PROMPT so Claude knows it's reviewing primary authored content.
        """
        counts = {"flagged": 0, "review": 0, "safe": 0}
        db = get_mongo_db()

        # Fetch the post — skip if already deleted or already analysed today
        post_doc = await db.posts.find_one(
            {"_id": post_id, "deletedAt": None},
        )
        if not post_doc:
            return counts

        # Check if this post was already analysed today
        already = await self._already_analysed([post_id], "post", factory)
        if post_id in already:
            return counts

        # Combine title and body for analysis
        title   = post_doc.get("title", "")
        body    = post_doc.get("content", "")
        text    = f"Title: {title}\n\nBody:\n{body}".strip()
        if not text:
            return counts

        author_id = str(post_doc.get("authorId", ""))

        result = await moderation_service.moderate(
            content_id   = post_id,
            content      = text,
            author_id    = author_id,
            model        = force_model,
            content_type = "post",
        )

        # Write to database
        async with factory() as session:
            record_id = await insert_moderation_record(
                session        = session,
                comment_id     = None,       # no comment — this is a post
                post_id        = post_id,
                content_type   = "post",
                content_id     = post_id,
                author_id      = author_id,
                content        = text,
                verdict        = result.verdict,
                confidence_pct = int(round(result.confidence * 100)),
                categories     = result.categories,
                explanation    = result.explanation,
                flagged_phrases= result.flaggedPhrases,
                model          = force_model or settings.moderator_model,
                trigger        = "auto",
            )

            if result.verdict in ("review", "remove"):
                await upsert_moderation_queue(
                    session              = session,
                    comment_id           = None,
                    post_id              = post_id,
                    content_type         = "post",
                    content_id           = post_id,
                    author_id            = author_id,
                    content              = text,
                    verdict              = result.verdict,
                    confidence_pct       = int(round(result.confidence * 100)),
                    explanation          = result.explanation,
                    flagged_phrases      = result.flaggedPhrases,
                    moderation_record_id = record_id,
                )
                if result.verdict == "remove":
                    counts["flagged"] += 1
                else:
                    counts["review"] += 1
            else:
                counts["safe"] += 1

            await session.commit()

        return counts

    # ── Comment scan ──────────────────────────────────────────────────────────

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

        # Filter already-analysed today
        already_done = await self._already_analysed(
            [c["id"] for c in comments], "comment", factory
        )
        new_comments = [c for c in comments if c["id"] not in already_done]

        if not new_comments:
            return counts

        # Analyse in batches
        for i in range(0, len(new_comments), settings.scan_batch_size):
            batch = new_comments[i : i + settings.scan_batch_size]
            stats = await self._analyse_and_store_comments(batch, post_id, factory, force_model)
            counts["scanned"]  += stats["scanned"]
            counts["flagged"]  += stats["flagged"]
            counts["review"]   += stats["review"]
            counts["safe"]     += stats["safe"]

        return counts

    # ── Shared helpers ────────────────────────────────────────────────────────

    async def _already_analysed(
        self,
        ids:          list[str],
        content_type: str,
        factory:      Any,
    ) -> set[str]:
        """
        Return the set of content IDs that already have a moderation_record today.
        Filters by content_type so post and comment IDs don't collide.
        """
        if not ids:
            return set()

        today_start = datetime.now(timezone.utc).replace(
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
            items        = comments,
            model        = force_model,
            content_type = "comment",
        )

        comment_map = {c["id"]: c for c in comments}

        async with factory() as session:
            for result in results:
                comment = comment_map.get(result.id)
                if not comment:
                    continue

                record_id = await insert_moderation_record(
                    session        = session,
                    comment_id     = result.id,
                    post_id        = post_id,
                    content_type   = "comment",
                    content_id     = result.id,
                    author_id      = comment.get("authorId", ""),
                    content        = comment.get("content", ""),
                    verdict        = result.verdict,
                    confidence_pct = int(round(result.confidence * 100)),
                    categories     = result.categories,
                    explanation    = result.explanation,
                    flagged_phrases= result.flaggedPhrases,
                    model          = force_model or settings.moderator_model,
                    trigger        = "auto",
                )

                if result.verdict in ("review", "remove"):
                    await upsert_moderation_queue(
                        session              = session,
                        comment_id           = result.id,
                        post_id              = post_id,
                        content_type         = "comment",
                        content_id           = result.id,
                        author_id            = comment.get("authorId", ""),
                        content              = comment.get("content", ""),
                        verdict              = result.verdict,
                        confidence_pct       = int(round(result.confidence * 100)),
                        explanation          = result.explanation,
                        flagged_phrases      = result.flaggedPhrases,
                        moderation_record_id = record_id,
                    )
                    if result.verdict == "remove":
                        stats["flagged"] += 1
                    else:
                        stats["review"] += 1
                else:
                    stats["safe"] += 1

            await session.commit()

        return stats
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown all active scans.
        Call this during application shutdown.
        """
        if not hasattr(self, '_active_scans') or not self._active_scans:
            return
        
        logger.info("Cancelling %d active scan tasks", len(self._active_scans))
        
        # Cancel all active scans
        for task in self._active_scans:
            task.cancel()
        
        # Wait for all tasks to complete cancellation with timeout
        try:
            await asyncio.gather(*self._active_scans, return_exceptions=True)
        except Exception as e:
            logger.error("Error during scan shutdown: %s", e)
        
        self._active_scans.clear()
        logger.info("All scan tasks cancelled")


scan_service = ScanService()
