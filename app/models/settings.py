from __future__ import annotations
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Moderation mode ───────────────────────────────────────────────────────
    # rule       — local detoxify only (free, offline, no API key needed)
    # anthropic  — Claude via Anthropic API (original behaviour)
    # openai     — any OpenAI-compatible provider (OpenAI, Groq, Together, Mistral…)
    # ollama     — local Ollama server (self-hosted, free)
    # hybrid     — rule-based pre-filter → LLM only for ambiguous cases (recommended)
    moderation_mode: Literal["rule", "anthropic", "openai", "ollama", "hybrid"] = "hybrid"

    # ── Anthropic ─────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    moderator_model:   str = "claude-haiku-3-5-20251001"

    # ── OpenAI-compatible (covers OpenAI, Groq, Together, Mistral, LM Studio) ─
    openai_api_key:  str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model:    str = "gpt-4o-mini"

    # ── Ollama (self-hosted) ──────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model:    str = "llama3.2"

    # ── Hybrid mode thresholds ────────────────────────────────────────────────
    # Fast score <= safe_ceiling  → return rule-based result (clearly safe)
    # Fast score >= flag_floor    → return rule-based result (clearly toxic)
    # Anything between            → escalate to the configured LLM backend
    hybrid_safe_ceiling: float = 0.15
    hybrid_flag_floor:   float = 0.80
    # Which LLM backend to use for the smart stage of hybrid mode
    # anthropic | openai | ollama
    hybrid_smart_backend: Literal["anthropic", "openai", "ollama"] = "anthropic"

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8001

    # ── Verdict thresholds ────────────────────────────────────────────────────
    remove_threshold: float = 0.85
    review_threshold: float = 0.50

    # ── MongoDB (read-only — Node.js owns writes) ─────────────────────────────
    mongodb_uri: str = ""
    mongodb_db:  str = "x_socials"

    # ── Laravel admin DB (insert/upsert only) ─────────────────────────────────
    laravel_db_url: str = ""

    # ── Scan config ───────────────────────────────────────────────────────────
    scan_batch_size: int = 20
    scan_lookback_h: int = 2

    # ── API key guard (optional) ──────────────────────────────────────────────
    api_key: str = ""


settings = Settings()
