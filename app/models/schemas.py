from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

ContentType = Literal["comment", "post"]

# ── Request models ────────────────────────────────────────────────────────────

class CommentRequest(BaseModel):
    """On-demand single-item analysis request (kept for backward compatibility)."""
    id:       str = Field(..., description="Content ID")
    content:  str = Field(..., min_length=1, max_length=10_000)
    authorId: str = Field(default="", description="Author user ID")


class BatchModerationRequest(BaseModel):
    comments: list[CommentRequest] = Field(..., min_length=1, max_length=50)


class ScanTriggerRequest(BaseModel):
    """
    Body for POST /scan/trigger.

    post_id:      Scope scan to one post's content + comments. Omit for full scan.
    content_type: Filter what to scan — 'comment', 'post', or null (both).
                  Useful for targeted re-scans: scan only posts, or only comments.
    force_model:  Override the default Claude model for this scan run.
    """
    post_id:      str | None         = Field(default=None, description="Scope to one post")
    content_type: ContentType | None = Field(default=None, description="Scan 'comment', 'post', or both (null)")
    force_model:  str | None         = Field(default=None, description="Override Claude model")


# ── Response models ───────────────────────────────────────────────────────────

ModerationVerdict = Literal["safe", "review", "remove"]


class ModerationResult(BaseModel):
    id:             str
    verdict:        ModerationVerdict
    confidence:     float = Field(..., ge=0.0, le=1.0)
    categories:     list[str] = []
    explanation:    str
    flaggedPhrases: list[str] = []
    error:          bool = False


class BatchModerationResponse(BaseModel):
    results: list[ModerationResult]
    total:   int
    flagged: int
    review:  int


class ScanTriggerResponse(BaseModel):
    started:     bool
    scan_run_id: int | None = None
    message:     str
