"""
schemas.py  (updated)

New types:
  EnqueueRequest   — body for POST /moderate/enqueue (from Node.js webhook)
  EnqueueResponse  — 202 acknowledgement returned to Node.js
  ScanMode         — 'standard' | 'reconciliation' for POST /scan/trigger

ScanTriggerRequest extended with mode field.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ContentType = Literal["comment", "post"]
ScanMode    = Literal["standard", "reconciliation"]

# ── Request models ────────────────────────────────────────────────────────────

class CommentRequest(BaseModel):
    """On-demand single-item analysis request."""
    id:       str = Field(..., description="Content ID")
    content:  str = Field(..., min_length=1, max_length=10_000)
    authorId: str = Field(default="", description="Author user ID")


class BatchModerationRequest(BaseModel):
    comments: list[CommentRequest] = Field(..., min_length=1, max_length=50)


class EnqueueRequest(BaseModel):
    """
    Real-time enqueue request from the Node.js webhook.

    Fields mirror the on-demand CommentRequest but use snake_case to match
    the Python convention and add content_type + post_id for routing.
    """
    id:           str         = Field(..., description="Content ID (MongoDB _id)")
    content:      str         = Field(..., min_length=1, max_length=50_000)
    author_id:    str         = Field(default="", description="Author user ID")
    content_type: ContentType = Field(..., description="'post' or 'comment'")
    post_id:      str | None  = Field(
        default=None,
        description="Parent post ID — required when content_type='comment'",
    )


class ScanTriggerRequest(BaseModel):
    """
    Body for POST /scan/trigger.

    post_id:      Scope scan to one post's content + comments. Omit for full scan.
    content_type: Filter what to scan — 'comment', 'post', or null (both).
    force_model:  Override the default model for this scan run.
    mode:         'standard' uses scan_lookback_h (short window, frequent runs).
                  'reconciliation' uses reconciliation_lookback_h (long window,
                  run daily to catch items missed by the real-time webhook).
    """
    post_id:      str | None         = Field(default=None)
    content_type: ContentType | None = Field(default=None)
    force_model:  str | None         = Field(default=None)
    mode:         ScanMode           = Field(default="standard")


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


class EnqueueResponse(BaseModel):
    """
    202 acknowledgement returned to the Node.js webhook caller.
    The caller ignores this — it's purely for observability/debugging.
    """
    accepted:     bool
    content_id:   str
    content_type: ContentType
    queue_depth:  int = Field(description="Current number of items being processed")


class ScanTriggerResponse(BaseModel):
    started:     bool
    scan_run_id: int | None = None
    message:     str
