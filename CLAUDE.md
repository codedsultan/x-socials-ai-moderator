# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the service locally (dev with live reload):**
```bash
uvicorn app.main:app --reload --port 8001
```

**Run via Docker (preferred):**
```bash
docker compose up -d moderator                    # dev server with live reload
docker compose --profile ollama up -d             # with local Ollama sidecar
```

**Run tests:**
```bash
# Locally
pytest app/tests/ -v

# Single test class
pytest app/tests/test_moderation.py -v -k "Hybrid"

# Via Docker (no need for real DB or API keys)
docker compose run --rm test
docker compose run --rm test pytest app/tests/test_moderation.py -v -k "Hybrid"
```

**Lint and type-check:**
```bash
ruff check app/ tests/
mypy app/
# or via Docker:
docker compose run --rm test ruff check app/ tests/
docker compose run --rm test mypy app/
```

**Environment setup:**
```bash
cp .env.example .env   # then add ANTHROPIC_API_KEY, MONGODB_URI, LARAVEL_DB_URL
```

## Architecture

This is a FastAPI AI content moderation service. It sits between a Node.js social platform (MongoDB) and a Laravel admin panel (MySQL). It reads content from MongoDB and writes verdicts to the Laravel DB.

### Data flow

1. **Real-time path**: Node.js webhook → `POST /moderate/enqueue` → `RealtimeQueue` → LLM backend → writes to `moderation_records` and `moderation_queue` in MySQL
2. **On-demand path**: `POST /moderate` / `POST /moderate/batch` → backend → returns result to caller (no DB write)
3. **Scan path**: `POST /scan/trigger` → `ScanService` reads MongoDB for unmoderated content → backend → writes to MySQL

### Backend system (`app/services/backends/`)

The `ModerationBackend` ABC defines a single method: `analyse(content_id, content, content_type, author_id) → ModerationResult`. All backends satisfy this contract.

Available backends, selected by `MODERATION_MODE`:
- **`rule`** — `RuleBasedBackend`: local detoxify model, free, no API keys
- **`anthropic`** — direct Anthropic Claude SDK
- **`openai`** — `OpenAICompatBackend`: works for OpenAI, Groq, Together, Mistral
- **`ollama`** — `OpenAICompatBackend` pointed at local Ollama
- **`openrouter`** — `OpenRouterBackend`: 200+ models, unified billing
- **`hybrid`** (default) — `HybridBackend`: runs rule-based first; escalates to LLM only for ambiguous scores (confidence between `HYBRID_FAST_SAFE_CEILING=0.15` and `HYBRID_FAST_FLAG_FLOOR=0.80`). Cuts LLM calls by ~60%.

`ModerationService` is the singleton orchestrator. It wraps any backend and adds semaphore-bounded concurrency for batch calls (`batch_concurrency=10`). `RealtimeQueue` has its own separate semaphore (`realtime_concurrency=5`).

### Verdict logic

Every backend inherits `_verdict(is_problematic, confidence)` from the base class:
- `confidence >= remove_threshold (0.85)` → `"remove"`
- `confidence >= review_threshold (0.50)` → `"review"`
- otherwise → `"safe"`

Errors always return `verdict="review"` so they surface for human inspection.

### Database (`app/services/db_client.py`)

Two async connections, lazily initialised:
- **MongoDB** (motor): read-only, reads `x_socials.comments` and `x_socials.posts`
- **Laravel DB** (SQLAlchemy + aiomysql/aiosqlite): writes to `moderation_records`, `moderation_queue`, `scan_runs`

Tests use `LARAVEL_DB_URL=sqlite+aiosqlite:///./test.db` — no MySQL needed in CI.

### Settings (`app/models/settings.py`)

All config via environment variables / `.env` (pydantic-settings). Key variables:
- `MODERATION_MODE` — selects backend
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`
- `MONGODB_URI`, `LARAVEL_DB_URL`
- `REMOVE_THRESHOLD` (0.85), `REVIEW_THRESHOLD` (0.50)
- `HYBRID_SMART_BACKEND` — which LLM the hybrid backend escalates to (`openrouter` by default)

### Testing

Tests live in `app/tests/`. `asyncio_mode = "auto"` is set in `pyproject.toml` so `async def test_*` methods run without `@pytest.mark.asyncio`. All backend tests mock the LLM calls — no real API keys needed. `RuleBasedBackend` tests bypass `__init__` to avoid triggering detoxify model downloads.

Model override routing: model strings containing `/` (e.g. `anthropic/claude-haiku-3-5`) route to `OpenRouterBackend`; strings without `/` route to `AnthropicBackend`.
