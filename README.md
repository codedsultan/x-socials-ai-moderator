# X-Socials AI Moderation Service

FastAPI microservice that analyses posts and comments for policy violations.
Part of the x-socials three-service moderation system alongside the Node.js
social platform and the Laravel admin panel.

## How it fits in the system

```
Node.js (x-socials)
  │  POST /moderate/enqueue  ← fires after every post/comment create or update
  │  fire-and-forget, 200ms timeout
  ▼
FastAPI (this service)
  │
  ├─ RealtimeQueue          real-time path — analyses content within seconds
  │    └─ asyncio Tasks     bounded by realtime_concurrency semaphore (default 5)
  │
  ├─ ScanService            reconciliation path — daily sweep for missed items
  │    └─ asyncio Tasks     bounded by batch_concurrency semaphore (default 10)
  │
  └─ ModerationService      shared backend dispatcher
       │
       ├─ rule       → RuleBasedBackend    (detoxify, local, free)
       ├─ anthropic  → AnthropicBackend    (Claude SDK — supports prompt caching)
       ├─ openai     → OpenAICompatBackend (OpenAI / Groq / Together / Mistral)
       ├─ ollama     → OpenAICompatBackend (local Ollama)
       ├─ openrouter → OpenRouterBackend   (200+ models, unified billing, auto-failover)
       └─ hybrid     → HybridBackend       (rule pre-filter + LLM for ambiguous only)
            │
            ├─ fast:  RuleBasedBackend
            └─ smart: any LLM backend (default: openrouter)

FastAPI writes results to MySQL (shared with Laravel):
  moderation_records  — append-only audit log (trigger: realtime | auto | manual)
  moderation_queue    — human review inbox (status: pending → reviewed/removed)
  scan_runs           — operational log for each reconciliation sweep

Laravel admin panel reads the same MySQL to power the queue, dashboard, and
moderation review pages. No message queue between them — shared DB is the
integration point.
```

## Two moderation pipelines

### Real-time (primary path)
Node.js fires `POST /moderate/enqueue` after every content creation or edit.
FastAPI acknowledges with `202 Accepted` in microseconds and processes
asynchronously. Content is analysed and results are in the database within
seconds. High-confidence violations surface to `AutoRemoveCommand` within
5 minutes of creation.

### Reconciliation (safety net)
Laravel fires `POST /scan/trigger` with `mode=reconciliation` daily at 03:00
UTC. The scan sweeps the last 48 hours for content with no `moderation_records`
row — items the real-time webhook dropped because FastAPI was temporarily
down, timed out, or the content was created before this system existed.
The `_already_analysed()` check skips already-processed items in microseconds
so only true gaps incur AI API calls.

The two pipelines write to the same tables. `INSERT IGNORE` on
`moderation_records` and `ON DUPLICATE KEY UPDATE` on `moderation_queue`
make concurrent writes safe.

## Hybrid mode — cost optimisation

```
content
   │
   ▼
detoxify (local, free, ~20ms)
   │
   ├─ score < 0.15  →  safe     (no LLM call)
   ├─ score > 0.80  →  flagged  (no LLM call)
   └─ 0.15–0.80    →  escalate to LLM (ambiguous cases only)
                          │
                          ▼
                    OpenRouter / Claude / OpenAI
                    (phrase-level attribution, explanation)
```

On typical social content ~60% of items are resolved by detoxify alone.
Only the ambiguous middle band reaches the paid API. Combined with the
real-time webhook skipping already-analysed content, LLM costs scale
sub-linearly with platform growth.

## Stack

| Layer | Library |
|---|---|
| HTTP framework | FastAPI + uvicorn |
| Validation | Pydantic v2 |
| Local classifier | detoxify (HuggingFace transformers + PyTorch) |
| Anthropic | `anthropic` SDK (native — supports prompt caching) |
| OpenRouter / OpenAI-compat | `openai` SDK (one client, any base URL) |
| MongoDB read | motor (async) |
| MySQL write | SQLAlchemy async + aiomysql |
| Concurrency | asyncio semaphores — no Celery, no Redis required |

## Quick start

```bash
cp .env.example .env
# Set MODERATION_MODE and the matching API keys (see Configuration below)

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

### Docker

```bash
# Hybrid mode (recommended)
cp .env.example .env
# Edit .env — set OPENROUTER_API_KEY or ANTHROPIC_API_KEY

docker compose up -d moderator

# With local Ollama (zero API cost)
# In .env: MODERATION_MODE=hybrid, HYBRID_SMART_BACKEND=ollama
docker compose --profile ollama up -d

# Tests
docker compose run --rm test
```

### Model weight caching

The `hf-cache` named volume persists detoxify's PyTorch weights (~200 MB)
across restarts. Pre-bake into the image for instant cold starts:

```dockerfile
# Uncomment in Dockerfile:
RUN python -c "from detoxify import Detoxify; Detoxify('original')"
```

## Configuration

All options via environment variables or `.env`.

### Moderation mode

```env
# Options: rule | anthropic | openai | ollama | openrouter | hybrid (recommended)
MODERATION_MODE=hybrid
```

### OpenRouter (recommended smart backend)

Single API key for 200+ models. Provides automatic failover across providers
if one is down, unified billing, and usage dashboards.

```env
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=anthropic/claude-haiku-3-5   # same model as direct Anthropic
OPENROUTER_SITE_URL=https://github.com/yourname/x-socials
OPENROUTER_APP_TITLE=X-Socials AI Moderator
```

Recommended models for moderation:

| Model | Cost | Quality | Notes |
|---|---|---|---|
| `anthropic/claude-haiku-3-5` | Low | High | Default — best quality/cost |
| `mistralai/mistral-nemo` | Very low | Good | Fast bulk screening |
| `meta-llama/llama-3.1-8b-instruct:free` | Free | Moderate | Low-stakes screening |
| `google/gemini-flash-1.5` | Very low | Good | Very fast |

### Anthropic direct (for prompt caching)

Keep `AnthropicBackend` wired when you implement prompt caching
(`anthropic-beta: prompt-caching-2024-07-31`). The static system prompt
cached for 5 minutes cuts per-call token cost by ~80%. OpenRouter cannot
surface this feature.

```env
ANTHROPIC_API_KEY=sk-ant-...
MODERATOR_MODEL=claude-haiku-3-5-20251001
```

### OpenAI-compatible providers

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.1-8b-instant
```

Supported providers: OpenAI · Groq · Together AI · Mistral · LM Studio · Ollama

### Hybrid tuning

```env
HYBRID_SAFE_CEILING=0.15        # detoxify score <= this → safe, skip LLM
HYBRID_FLAG_FLOOR=0.80          # detoxify score >= this → flagged, skip LLM
HYBRID_SMART_BACKEND=openrouter # openrouter | anthropic | openai | ollama
```

### Scan settings

```env
SCAN_LOOKBACK_H=1               # standard scan window (on-demand)
RECONCILIATION_LOOKBACK_H=48    # reconciliation sweep window (daily)
SCAN_BATCH_SIZE=20              # comments per batch
```

### Concurrency

```env
# Max concurrent LLM calls in the real-time queue
# Lower = less API burst, more queue depth during spikes
REALTIME_CONCURRENCY=5

# Max concurrent LLM calls in the reconciliation scan batch
BATCH_CONCURRENCY=10
```

### Verdict thresholds

```env
REMOVE_THRESHOLD=0.85   # confidence >= this → 'remove'
REVIEW_THRESHOLD=0.50   # confidence >= this (below remove) → 'review'
```

### Database connections

```env
MONGODB_URI=mongodb://localhost:27017    # Node.js data — read only
MONGODB_DB=x_socials
LARAVEL_DB_URL=mysql+aiomysql://user:pass@host:3306/x_socials_admin
```

### API key guard (optional)

```env
API_KEY=your-secret-key
# Enforced via X-Api-Key header on all endpoints
# Set to empty string to disable (internal network only)
```

## API reference

### Health and stats

```
GET /health
```

Returns live queue depth and in-memory stats for the real-time webhook queue.
Stats reset on process restart — use `moderation_records.trigger` for permanent
historical data.

```json
{
  "status": "ok",
  "realtime_queue_depth": 3,
  "realtime_stats": {
    "enqueued_total": 1482,
    "completed_total": 1459,
    "failed_total": 2,
    "skipped_total": 21,
    "avg_latency_ms": 340,
    "p95_latency_ms": 890,
    "health_pct": 98.4
  },
  "active_scans": 0
}
```

**Reading the stats:**
- `skipped_total` — content already analysed today. Expected and healthy.
- `failed_total` — unhandled errors. Should be near zero; reconciliation covers these.
- `health_pct` — `completed / enqueued`. Below 95% warrants investigation.
- `p95_latency_ms` — slowest typical analysis. Returns 0 until 20 samples collected.

### Real-time enqueue (webhook from Node.js)

```
POST /moderate/enqueue
X-Api-Key: <optional>

{
  "id": "6849f2a1c3d4e5f6a7b8c901",
  "content": "Comment or post text",
  "author_id": "user-123",
  "content_type": "comment",
  "post_id": "6849f2a1c3d4e5f6a7b8c900"
}
```

Returns `202 Accepted` immediately. Analysis happens in the background.

```json
{
  "accepted": true,
  "content_id": "6849f2a1c3d4e5f6a7b8c901",
  "content_type": "comment",
  "queue_depth": 4
}
```

`content_type` is `"post"` or `"comment"`. `post_id` is required when
`content_type` is `"comment"`. For posts, omit `post_id` or set it to the
same value as `id`.

### On-demand analysis (no DB write)

```
POST /moderate

{
  "id": "6849f2a1c3d4e5f6a7b8c901",
  "content": "Text to analyse",
  "authorId": "user-123"
}
```

```json
{
  "id": "6849f2a1c3d4e5f6a7b8c901",
  "verdict": "review",
  "confidence": 0.72,
  "categories": ["harassment"],
  "explanation": "The comment contains language that could be perceived as targeted harassment.",
  "flaggedPhrases": ["you're worthless"],
  "error": false
}
```

Pass `?force_model=anthropic/claude-sonnet-4-5` to use a higher-quality model
for borderline cases. Model strings with `/` route to OpenRouter; strings
without route to the Anthropic SDK.

### Batch analysis (no DB write)

```
POST /moderate/batch

{
  "comments": [
    { "id": "c1", "content": "Great post!" },
    { "id": "c2", "content": "...", "authorId": "user-123" }
  ]
}
```

Up to 50 items. Concurrent with the `batch_concurrency` semaphore.

### Reconciliation scan

```
POST /scan/trigger

{
  "post_id": null,
  "content_type": null,
  "force_model": null,
  "mode": "reconciliation"
}
```

`mode` is `"standard"` (default, `scan_lookback_h` window) or
`"reconciliation"` (`reconciliation_lookback_h` window, run daily).
Returns the `scan_run_id` immediately; scan runs in the background.

```json
{
  "started": true,
  "scan_run_id": 47,
  "message": "Scan started for full platform [posts + comments] mode=reconciliation (run_id=47)"
}
```

## Verdicts

| Verdict | Confidence | Action |
|---|---|---|
| `safe` | Below review threshold | No action — recorded in `moderation_records` |
| `review` | ≥ 0.50 | Queued for human review in admin panel |
| `remove` | ≥ 0.85 | Queued for removal — auto-removed if ≥ 95% within 5 min |

## Observability

**Live:** `GET /health` — queue depth, in-memory stats, active scan count.

**Historical — pipeline split (last 7 days):**
```sql
SELECT `trigger`, COUNT(*) as total,
       SUM(verdict = 'remove') as removed,
       SUM(verdict = 'safe') as safe
FROM moderation_records
WHERE created_at >= NOW() - INTERVAL 7 DAY
GROUP BY `trigger`;
```

A healthy system shows `realtime` at 90%+ of records. If `auto`
(reconciliation) spikes, the webhook is dropping items — check FastAPI logs
and the `failed_total` stat.

**Scan history:**
```sql
SELECT id, status, posts_scanned, comments_scanned, flagged, started_at, finished_at
FROM scan_runs
ORDER BY started_at DESC
LIMIT 20;
```

## Backend comparison

| Mode | Cost | Latency | Offline | Phrase attribution | Notes |
|---|---|---|---|---|---|
| `rule` | Free | ~20ms | ✅ | ❌ | Detoxify only |
| `anthropic` | Pay-per-token | ~800ms | ❌ | ✅ | Prompt caching ready |
| `openai` | Pay-per-token | ~400ms | ❌ | ✅ | Any OpenAI-compat URL |
| `ollama` | Free | ~2–10s | ✅ | ✅ | Self-hosted |
| `openrouter` | Pay-per-token | ~300ms | ❌ | ✅ | Failover + unified billing |
| `hybrid` | ~40% of LLM cost | ~20–800ms | Partial | ✅ escalated | **Recommended** |

## Adding a new backend

1. Create `app/services/backends/my_backend.py` extending `ModerationBackend`.
2. Implement `async def analyse(self, content_id, content, content_type, author_id) -> ModerationResult`.
3. Add a branch in `moderation_service._build_backend()`.
4. Add the new literal to `Settings.moderation_mode` in `settings.py`.
5. Add it to `HYBRID_SMART_BACKEND` options if it should be usable as the hybrid smart stage.

## Related services

| Service | Role |
|---|---|
| [x-socials](https://github.com/codedsultan/x-socials) | Node.js social platform backend API — content source, fires webhook |
| [x-socials-web](https://github.com/codedsultan/x-socials-web) | Next.js 16 frontend for the x-socials API  |
| [x-socials-admin](https://github.com/codedsultan/x-socials-admin) | Laravel admin panel — review queue, dashboard, auto-remove |

---

## License

MIT License — see the [LICENSE](LICENSE) file for details.

---

## Author

**Olusegun Ibraheem**
- Website: [codesultan.xurl.fyi](https://codesultan.xurl.fyi)
- Email: codesultan369@gmail.com

