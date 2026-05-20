from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.models.schemas import (
    CommentRequest,
    BatchModerationRequest,
    BatchModerationResponse,
    ModerationResult,
    ScanTriggerRequest,
    ScanTriggerResponse,
)
from app.models.settings import settings
from app.services.moderation_service import moderation_service
from app.services.scan_service import scan_service

router = APIRouter(tags=["moderation"])

# ── Optional API key guard ────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)


async def verify_api_key(key: str | None = Depends(api_key_header)) -> None:
    """If MODERATOR_API_KEY is set, enforce it. Otherwise allow all requests."""
    if settings.api_key and key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )


# ── On-demand endpoints (no DB writes — results returned to caller) ───────────

@router.post(
    "/moderate",
    response_model=ModerationResult,
    summary="Analyse a single comment",
    description=(
        "On-demand single-comment analysis. Results are returned to the caller "
        "but NOT written to the database. Use this for the admin modal preview. "
        "Pass force_model in the query string to use a higher-quality model."
    ),
)
async def moderate_single(
    body:         CommentRequest,
    force_model:  str | None = None,
    _:            None = Depends(verify_api_key),
) -> ModerationResult:
    return await moderation_service.moderate(
        comment_id=body.id,
        content=body.content,
        author_id=body.authorId,
        model=force_model,
    )


@router.post(
    "/moderate/batch",
    response_model=BatchModerationResponse,
    summary="Analyse multiple comments concurrently",
    description=(
        "On-demand batch analysis — up to 50 comments in parallel. "
        "Results are returned to the caller but NOT written to the database. "
        "Use this for the Moderation/Index page preview."
    ),
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


# ── Background scan endpoint (reads MongoDB, writes Laravel DB) ───────────────

@router.post(
    "/scan/trigger",
    response_model=ScanTriggerResponse,
    summary="Trigger a background moderation scan",
    description=(
        "Starts a background scan immediately and returns the scan_run_id. "
        "The scan runs after the response is returned — poll the Laravel "
        "scan_runs table or dashboard to see progress.\n\n"
        "Without post_id: full scan over all posts with recent comments.\n"
        "With post_id: scans only comments on that specific post.\n"
        "force_model: override the Claude model for this run (e.g. claude-sonnet-4-20250514)."
    ),
)
async def trigger_scan(
    body: ScanTriggerRequest,
    _:    None = Depends(verify_api_key),
) -> ScanTriggerResponse:
    try:
        run_id = await scan_service.trigger(
            post_id      = body.post_id,
            force_model  = body.force_model,
            content_type = body.content_type,
        )
        scope = f"post {body.post_id}" if body.post_id else "full platform"
        ct    = f" [{body.content_type}]" if body.content_type else " [posts + comments]"
        return ScanTriggerResponse(
            started     = True,
            scan_run_id = run_id,
            message     = f"Scan started for {scope}{ct} (run_id={run_id})",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
