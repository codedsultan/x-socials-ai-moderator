# x-socials AI Moderation Service

FastAPI microservice that moderates posts and comments for policy violations.
Supports multiple backends — local ML, Anthropic Claude, any OpenAI-compatible
provider, or a hybrid mode that cuts LLM API costs by ~60%.

## Architecture

```
Laravel scheduler / admin panel
         │
         ▼
  FastAPI  (this service)
         │
         ▼
  ModerationService
         │
         ├─ rule      →  RuleBasedBackend   (detoxify, local, free)
         ├─ anthropic →  AnthropicBackend   (Claude)
         ├─ openai    →  OpenAICompatBackend (OpenAI / Groq / Together / Mistral)
         ├─ ollama    →  OpenAICompatBackend (local Ollama)
         └─ hybrid    →  HybridBackend
                           │
                           ├─ fast:  RuleBasedBackend
                           └─ smart: any LLM backend (ambiguous cases only)
```

### Hybrid mode — how it works

1. **Fast stage** — detoxify scores the content locally (no API call, microseconds).
2. **If unambiguous** — clearly safe (`< HYBRID_SAFE_CEILING`) or clearly toxic
   (`>= HYBRID_FLAG_FLOOR`) — return the rule-based result immediately.
3. **If ambiguous** — escalate to the configured LLM for a richer verdict and
   phrase-level attribution.

On typical social content this eliminates ~60% of LLM API calls while keeping
full quality on the items that matter.

## Stack

| Layer | Library |
|---|---|
| HTTP framework | FastAPI + uvicorn |
| Validation | Pydantic v2 |
| Local classifier | detoxify (HuggingFace transformers + PyTorch) |
| Anthropic | `anthropic` SDK |
| Any OpenAI-compat | `openai` SDK (one client, any base URL) |
| MongoDB (read) | motor (async) |
| Laravel DB (write) | SQLAlchemy async + aiomysql / aiosqlite |

## Setup

```bash
# 1. Copy and configure environment
cp .env.example .env
# Edit .env — set MODERATION_MODE and the matching API keys

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the service
uvicorn app.main:app --reload --port 8001

# 4. Run tests
pytest app/tests/ -v
```

## Docker

### Quick start (hybrid mode)

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY (or switch HYBRID_SMART_BACKEND)

docker compose up -d moderator
```

### With local Ollama (zero API cost)

```bash
# In .env:
#   MODERATION_MODE=hybrid
#   HYBRID_SMART_BACKEND=ollama
#   OLLAMA_MODEL=llama3.2

docker compose --profile ollama up -d
```

### Run tests in Docker

```bash
docker compose run --rm test
```

### Build only

```bash
docker build -t x-socials-moderator .
docker run --env-file .env -p 8001:8001 x-socials-moderator
```

### Model weight caching

The `hf-cache` named volume persists detoxify's PyTorch weights (~200 MB)
across container restarts. On a cold start (empty volume) the weights are
downloaded automatically on the first request.

To pre-bake weights into the image (larger image, instant cold start):

```dockerfile
# Uncomment in Dockerfile:
RUN python -c "from detoxify import Detoxify; Detoxify('original')"
```

## Configuration

All options are set via environment variables (or `.env`).

### Moderation mode

```env
# Options: rule | anthropic | openai | ollama | hybrid
MODERATION_MODE=hybrid
```

### Anthropic (Claude)

```env
ANTHROPIC_API_KEY=sk-ant-...
MODERATOR_MODEL=claude-haiku-3-5-20251001
```

### OpenAI-compatible providers

Point `OPENAI_BASE_URL` at any compatible endpoint:

| Provider | Base URL | Example model |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.1-8b-instant` |
| Together AI | `https://api.together.xyz/v1` | `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` |
| Mistral | `https://api.mistral.ai/v1` | `mistral-small-latest` |
| Ollama (local) | `http://localhost:11434/v1` | `llama3.2` |
| LM Studio | `http://localhost:1234/v1` | *(loaded model)* |

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.1-8b-instant
```

### Hybrid tuning

```env
HYBRID_SAFE_CEILING=0.15     # fast score <= this → skip LLM (safe)
HYBRID_FLAG_FLOOR=0.80       # fast score >= this → skip LLM (flagged)
HYBRID_SMART_BACKEND=anthropic  # anthropic | openai | ollama
```

### Verdict thresholds

```env
REMOVE_THRESHOLD=0.85   # confidence >= this → 'remove'
REVIEW_THRESHOLD=0.50   # confidence >= this (but < remove) → 'review'
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness — returns mode, model, thresholds |
| `POST` | `/moderate` | Analyse a single comment or post (no DB write) |
| `POST` | `/moderate/batch` | Analyse up to 50 items concurrently (no DB write) |
| `POST` | `/scan/trigger` | Background scan — reads MongoDB, writes Laravel DB |
| `GET` | `/docs` | Swagger UI |

### Single item request

```json
POST /moderate
{
  "id": "6849f2a1c3d4e5f6a7b8c901",
  "content": "This is the text to analyse",
  "authorId": "019e29a8-7b9f-7209-bf7c-9fa84a4f3a74"
}
```

### Response

```json
{
  "id": "6849f2a1c3d4e5f6a7b8c901",
  "verdict": "safe",
  "confidence": 0.03,
  "categories": [],
  "explanation": "No policy violations detected.",
  "flaggedPhrases": [],
  "error": false
}
```

### Batch request

```json
POST /moderate/batch
{
  "comments": [
    { "id": "c1", "content": "Great post!" },
    { "id": "c2", "content": "...", "authorId": "user-123" }
  ]
}
```

### Scan trigger

```json
POST /scan/trigger
{
  "post_id": null,
  "content_type": null,
  "force_model": null
}
```

`post_id` scopes the scan to one post. `content_type` can be `"post"`,
`"comment"`, or `null` (both). `force_model` overrides the backend model for
that run (Anthropic model strings only, e.g. `"claude-sonnet-4-20250514"`).

## Verdicts

| Verdict | Meaning | Suggested action |
|---|---|---|
| `safe` | No violations detected | No action |
| `review` | Potentially problematic | Human review |
| `remove` | Clear violation, high confidence | Remove immediately |

## Backend comparison

| Mode | Cost | Latency | Offline | Phrase attribution | Setup |
|---|---|---|---|---|---|
| `rule` | Free | ~20 ms | ✅ | ❌ | `pip install detoxify` |
| `anthropic` | Pay-per-token | ~800 ms | ❌ | ✅ | API key |
| `openai` | Pay-per-token | ~400 ms | ❌ | ✅ | API key + base URL |
| `ollama` | Free | ~2–10 s | ✅ | ✅ | Ollama installed |
| `hybrid` | ~40% of LLM cost | ~20–800 ms | Partial | ✅ for escalated | Detoxify + LLM key |

## Adding a new backend

1. Create `app/services/backends/my_backend.py` with a class that extends `ModerationBackend`.
2. Implement `async def analyse(self, content_id, content, content_type, author_id) -> ModerationResult`.
3. Add a branch in `moderation_service._build_backend()`.
4. Add the new option to `Settings.moderation_mode` in `settings.py`.
