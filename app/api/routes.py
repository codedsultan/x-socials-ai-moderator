"""
routes.py  (updated — health endpoint with realtime_stats)

Changes:
  GET /health now returns a 'realtime_stats' block from realtime_queue.stats.
  All existing endpoints unchanged.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.models.schemas import (
    CommentRequest,
    BatchModerationRequest,
    BatchModerationResponse,
    EnqueueRequest,
    EnqueueResponse,
    ModerationResult,
    ScanTriggerRequest,
    ScanTriggerResponse,
)
from app.models.settings import settings
from app.services.moderation_service import moderation_service
from app.services.scan_service import scan_service
from app.services.realtime_queue import realtime_queue

router = APIRouter(tags=["moderation"])

# ── API key guard ─────────────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)


async def verify_api_key(key: str | None = Depends(api_key_header)) -> None:
    if settings.api_key and key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )


# ── Health ────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    summary="Service health and queue stats",
    description=(
        "Returns live queue depth, in-memory stats for the real-time webhook "
        "queue, and the count of active background scan tasks. "
        "Stats reset on process restart — use moderation_records.trigger for "
        "permanent historical data."
    ),
)
async def health() -> dict:
    return {
        "status":               "ok",
        "realtime_queue_depth": realtime_queue.depth,
        "realtime_stats":       realtime_queue.stats.to_dict(),
        "active_scans":         len(scan_service._active_scans),
    }


# ── Real-time enqueue (webhook from Node.js) ──────────────────────────────────

@router.post(
    "/moderate/enqueue",
    response_model=EnqueueResponse,
    status_code=202,
    summary="Enqueue a post or comment for real-time moderation",
    description=(
        "Called by the Node.js service immediately after a post or comment is "
        "created or updated. Acknowledges with 202 Accepted and analyses "
        "asynchronously. Results appear in moderation_records within seconds."
    ),
)
async def enqueue_content(
    body: EnqueueRequest,
    _:    None = Depends(verify_api_key),
) -> EnqueueResponse:
    realtime_queue.enqueue(
        content_id   = body.id,
        content      = body.content,
        author_id    = body.author_id,
        content_type = body.content_type,
        post_id      = body.post_id,
    )
    return EnqueueResponse(
        accepted     = True,
        content_id   = body.id,
        content_type = body.content_type,
        queue_depth  = realtime_queue.depth,
    )


# ── On-demand endpoints ───────────────────────────────────────────────────────

@router.post(
    "/moderate",
    response_model=ModerationResult,
    summary="Analyse a single item on demand",
    description=(
        "Results returned to caller only — not written to the database. "
        "Use for the admin modal preview or admin re-analysis."
    ),
)
async def moderate_single(
    body:        CommentRequest,
    force_model: str | None = None,
    _:           None = Depends(verify_api_key),
) -> ModerationResult:
    return await moderation_service.moderate(
        content_id=body.id,
        content=body.content,
        author_id=body.authorId,
        model=force_model,
    )


@router.post(
    "/moderate/batch",
    response_model=BatchModerationResponse,
    summary="Analyse multiple items concurrently",
    description="On-demand batch — up to 50 items. Results not written to database.",
)
async def moderate_batch(
    body: BatchModerationRequest,
    _:    None = Depends(verify_api_key),
) -> BatchModerationResponse:
    comments = [c.model_dump(by_alias=True) for c in body.comments]
    results  = await moderation_service.moderate_batch(comments)

    return BatchModerationResponse(
        results=results,
        total=len(results),
        flagged=sum(1 for r in results if r.verdict == "remove"),
        review=sum(1 for r in results if r.verdict == "review"),
    )


# ── Background scan ───────────────────────────────────────────────────────────

@router.post(
    "/scan/trigger",
    response_model=ScanTriggerResponse,
    summary="Trigger a background moderation scan",
    description=(
        "mode='standard'       — short lookback, on-demand use.\n"
        "mode='reconciliation' — long lookback, daily scheduler.\n"
        "Scans content not yet in moderation_records and fills the gap."
    ),
)
async def trigger_scan(
    body: ScanTriggerRequest,
    _:    None = Depends(verify_api_key),
) -> ScanTriggerResponse:
    try:
        run_id = await scan_service.trigger(
            post_id        = body.post_id,
            force_model    = body.force_model,
            content_type   = body.content_type,
            reconciliation = body.mode == "reconciliation",
        )
        scope = f"post {body.post_id}" if body.post_id else "full platform"
        ct    = f" [{body.content_type}]" if body.content_type else " [posts + comments]"
        return ScanTriggerResponse(
            started     = True,
            scan_run_id = run_id,
            message     = f"Scan started for {scope}{ct} mode={body.mode} (run_id={run_id})",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
