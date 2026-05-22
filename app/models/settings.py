"""
settings.py  (updated)

New settings:
  realtime_concurrency      — semaphore size for RealtimeQueue (default 5).
                              Separate from batch_concurrency (scan pipeline).
                              Lower default because real-time items arrive
                              unpredictably and we don't want a burst to starve
                              the scan pipeline's semaphore budget.

  reconciliation_lookback_h — how far back the reconciliation scan looks (default 48h).
                              The standard scan uses scan_lookback_h (2h) and runs
                              frequently. The reconciliation scan uses this longer
                              window and runs daily to catch webhook-missed items.
"""
from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Moderation mode ───────────────────────────────────────────────────────
    moderation_mode: Literal[
        "rule", "anthropic", "openai", "ollama", "openrouter", "hybrid"
    ] = "hybrid"

    # ── Anthropic ─────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    moderator_model:   str = "claude-haiku-3-5-20251001"

    # ── OpenAI-compatible ─────────────────────────────────────────────────────
    openai_api_key:  str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model:    str = "gpt-4o-mini"

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model:    str = "llama3.2"

    # ── OpenRouter ────────────────────────────────────────────────────────────
    openrouter_api_key:  str = ""
    openrouter_model:    str = "anthropic/claude-haiku-3-5"
    openrouter_site_url: str = "https://github.com/codesultan/x-socials"
    openrouter_app_title:str = "X-Socials AI Moderator"

    # ── Hybrid mode ───────────────────────────────────────────────────────────
    hybrid_safe_ceiling:  float = 0.15
    hybrid_flag_floor:    float = 0.80
    hybrid_smart_backend: Literal["anthropic", "openai", "ollama", "openrouter"] = "openrouter"

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8001

    # ── Verdict thresholds ────────────────────────────────────────────────────
    remove_threshold: float = 0.85
    review_threshold: float = 0.50

    # ── MongoDB ───────────────────────────────────────────────────────────────
    mongodb_uri: str = ""
    mongodb_db:  str = "x_socials"

    # ── Laravel DB ────────────────────────────────────────────────────────────
    laravel_db_url: str = ""

    # ── Scan config ───────────────────────────────────────────────────────────
    scan_batch_size: int = 20

    # Standard scan lookback — short window, used by frequent scheduled scans
    # or manual on-demand scans. With real-time webhooks in place, this can be
    # reduced to 1h since fresh content is handled immediately.
    scan_lookback_h: int = 1

    # Reconciliation scan lookback — long window, used by the daily scheduler
    # run to catch items the real-time webhook dropped (FastAPI down, timeout,
    # edit after creation, historical backfill). 48h gives two full days of
    # coverage so even a prolonged outage doesn't leave gaps.
    reconciliation_lookback_h: int = 48

    # ── Concurrency ───────────────────────────────────────────────────────────
    # Scan pipeline: max concurrent LLM calls in moderate_batch()
    batch_concurrency: int = 10

    # Real-time queue: max concurrent LLM calls in RealtimeQueue._process_one()
    # Lower than batch_concurrency — real-time bursts shouldn't starve the scan.
    realtime_concurrency: int = 5

    # ── API key guard ─────────────────────────────────────────────────────────
    api_key: str = ""


settings = Settings()
