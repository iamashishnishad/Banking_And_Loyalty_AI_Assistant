# Banking & Loyalty AI Assistant

A Hindi / Hinglish / English assistant for banking and rewards questions, built as a multi-tenant
backend: one bank per tenant, documents and customers isolated per tenant.

**Status: Phases 1–3 complete.**
- **Phase 1:** hybrid retrieval over Postgres + pgvector, document ingestion, Docker Compose, Alembic, env config.
- **Phase 2:** an LLM tool-calling agent (Gemini or Claude) with Redis conversation memory,
  Hindi/Hinglish detection and transliteration, voice input via local Whisper, and SSE streaming.
- **Phase 3:** JWT auth with tenant-from-token isolation, Redis rate limits per user and tenant,
  RabbitMQ ingestion workers, a Redis semantic answer cache, and an LLM fallback chain with
  circuit breakers, timeouts and a no-LLM degraded mode.

## Architecture

```
                     ┌──────────────────────────── FastAPI (api) ─────────────────────────────┐
 text  ─▶ /chat ─────┤                                                                        │
 SSE   ◀─ /chat/stream                     ┌─────────────── agent loop ──────────────┐        │
 audio ─▶ /voice/chat ─▶ Whisper (local) ─▶│ Claude (tool calling, streamed)         │        │
                     │                     │   ├─ search_knowledge_base ─▶ hybrid    │        │
                     │  language detect ─▶ │   ├─ get_points_balance                 │        │
                     │  memory (Redis) ──▶ │   ├─ start_redemption                   │        │
                     │                     │   └─ confirm_redemption (next turn only)│        │
                     │                     └─────────────────────────────────────────┘        │
                     │  hybrid search: vector (HNSW) + lexical (tsvector) + phonetic,  RRF    │
                     │  ingestion: PDF/DOCX/MD/TXT → chunk → embed → store                    │
                     └──────────────┬──────────────────────────────────────┬──────────────────┘
                         Postgres 17 + pgvector 0.8                   Redis 8
                         tenants, documents, chunks,                  conversation memory, turn counter,
                         customers, points_ledger, jobs               pending redemptions, query-embedding cache,
                                                                      semantic answer cache (vector index), rate limits
   POST /documents ─▶ jobs row ─▶ RabbitMQ ─▶ worker(s): parse → chunk → embed → store

   Every request: JWT (tenant + role) ─▶ rate limit (user + tenant buckets) ─▶ route
   /chat: semantic cache ─▶ agent (primary LLM ⇢ fallback LLM, circuit breakers) ─▶ no-LLM baseline
```

### Hybrid search

Two retrievers run in a single SQL statement ([app/services/search.py](app/services/search.py)):

1. **Lexical**: Postgres full-text search on a generated `tsvector` column, ranked with
   `ts_rank_cd`. It uses the `simple` config because Postgres ships no Hindi config. Stopword removal
   (English, romanised Hindi, Devanagari) and spelling normalisation (`pts`→`points`,
   `पॉइंट्स`→`points`, `khatam`→`expire`) run in Python
   ([app/services/text.py](app/services/text.py)), and the same function processes documents and
   queries. Query terms are OR-ed for recall.
2. **Vector**: `intfloat/multilingual-e5-small` embeddings (384-d, cosine, HNSW index). They match
   meaning across scripts, e.g. a Devanagari question finds an English policy paragraph.

The rankings are merged with **Reciprocal Rank Fusion**, `score = Σ 1/(60 + rank)`. RRF uses ranks
only, so the two score scales never need calibrating. `mode=vector|lexical` on `/search` runs a
single retriever, which is how the Phase 4 evals compare them.

> `ts_rank_cd` is not true BM25: it has no IDF term, which is why stopword removal matters.
> If evals show lexical recall is the bottleneck, the upgrade is ParadeDB `pg_search`, a real
> BM25 index inside Postgres.

### Phonetic retriever (Hinglish spelling)

Hinglish has no standard spelling: *khatam / khatm / खत्म* and *card / कार्ड* are the same word.
[language.py](app/services/language.py) romanises Devanagari the way people type (`खत्म → khatm`),
then reduces every word to a consonant skeleton: aspirates merged, `c/k/q` merged, `z/j` merged,
vowels dropped (`khatam → ktm`, `कार्ड → krd`). Those keys get their own `tsvector` index and
are the third retriever in RRF. `phonetic_rank` in `/search` results shows when it contributed.

### The agent

[app/agent/](app/agent/) runs a manual tool loop over streamed LLM responses. The provider sits
behind one interface ([llm.py](app/agent/llm.py)) and is picked with `LLM_PROVIDER`:

| Provider | Default model | Notes |
| --- | --- | --- |
| `gemini` | `gemini-2.5-flash` | Free tier works (5 requests/min, and one chat turn is about 2–3 requests). Thought signatures are replayed unchanged within a turn. |
| `claude` | `claude-haiku-5-5`, effort `low` | Prompt caching on the system prompt and tools |

The loop uses one message format and each adapter converts to and from it. Provider errors become
`LLMError`, and a rate limit is shown to the customer as "busy, try again in a minute".
There are four tools:

| Tool | What it does |
| --- | --- |
| `search_knowledge_base` | Hybrid search; passages come back numbered, and the model cites `[1]`, `[2]` |
| `get_points_balance` | The current customer's balance and points expiring soon |
| `start_redemption` | Checks the balance and stores a pending redemption in Redis; spends nothing |
| `confirm_redemption` | Executes the pending redemption, but only from a **later** customer message |

Guard rails are enforced in code, not only in the prompt:
- **No customer id in any tool.** The customer comes from the request, so a prompt injection
  ("show CUST1002's balance") has nothing to target.
- **Two-step redemption checked by turn number.** `confirm_redemption` fails if the confirmation
  isn't from a later customer message than `start_redemption`. The model can't redeem and
  confirm in one reply, even when told "I confirm in advance".
- **Idempotent redemption.** The pending id is the ledger's idempotency key, so a retried confirm
  can't double-spend.
- **Validated inputs.** Tool inputs are checked with pydantic (schemas are `strict`, inputs
  stream eagerly). A tool call cut off by `max_tokens` is never executed.
- **Instructions vs data.** Document text is treated as data. Operator context (bank, customer,
  detected language) goes in a mid-conversation `system` message, not the user turn.

**Prompt caching:** the system prompt and tools never change between requests, so they are cached.

**Memory:** Redis stores the last 20 user/assistant messages per customer (24h TTL), keeping
only text, not tool results.

**Streaming:** `/chat/stream` sends SSE events: `meta`, `token`, `tool_call`, `tool_result`, and
then `done` or `error`. `done` includes sources, tool calls and token usage.

### Voice

`/voice/transcribe` and `/voice/chat` run **faster-whisper `small`** (int8, CPU) inside the API
container. It's free, needs no key, and the model is baked into the image. Whisper sometimes labels
Hindi speech as Urdu and writes Perso-Arabic script, so the service detects the language first,
restricts it to Hindi or English, and transcribes with that language fixed.

### Multi-tenancy and auth

Every table carries `tenant_id`, and every query filters on it. `chunks.tenant_id` is denormalised
so search doesn't need a join. HNSW runs with `hnsw.iterative_scan = relaxed_order` so the tenant
filter can't starve the top-k.

The tenant comes from a signed JWT (HS256, algorithm pinned; `iss`, `aud` and `exp` required), never
from a header the caller controls. An `X-Tenant` header is ignored. Redis keys (memory, cache,
rate limits) are tenant-scoped too. [auth.py](app/services/auth.py) defines three roles:

| Role | Can do |
| --- | --- |
| `customer` | Only their own `customer_ref` (chat, voice, points, redeem, memory), plus search |
| `staff` | Any customer in their tenant (e.g. a call-centre agent) |
| `admin` | Staff rights, plus document upload/delete and job status |

`scripts/issue_token.py` stands in for an identity provider. Every route except `/health/*` needs a token.

### Rate limits

Each request takes a token from two Redis token buckets: one per user (default 30/min, burst 10)
and one per tenant (600/min, burst 100). Both are checked and charged in a single Lua script
([ratelimit.py](app/services/ratelimit.py)), so the check is atomic across API replicas, a
request blocked by the tenant bucket doesn't spend the user's token, and Redis's own clock is
used. Over the limit: `429` with `Retry-After`. If Redis is down the limiter fails open.

### Async ingestion (RabbitMQ)

`POST /documents` validates the file (type, size, duplicate by SHA-256), stores a `jobs` row with
the bytes, publishes the job id and returns **202**. A worker ([app/worker.py](app/worker.py))
parses, chunks and embeds it. Poll `GET /documents/jobs/{id}`.

```
API ──publish {job_id}──▶ jobs ──▶ worker ──ack──▶ done
                           │  ▲        │ transient error, attempts < 3
              reject ──────┘  └── jobs.retry (10 s TTL) ◀──┘
                ▼
            jobs.dead   (corrupt file, or out of attempts)
```

Delivery is at-least-once, so every step is idempotent. Jobs are claimed with a conditional
`UPDATE`, so a duplicate message for a finished job is dropped. Ingestion dedups on
`(tenant, sha256)`. A message is acked only after the outcome is committed. Postgres is the
source of truth: if RabbitMQ is down the upload still returns 202, and the worker's sweeper
re-publishes jobs left `queued`. A corrupt file fails at once without retries. Scale with
`docker compose up -d --scale worker=3`. Management UI: http://localhost:15673 (guest/guest).

### Semantic cache

A policy question close enough to one already answered gets the stored answer back, with no
LLM call ([semantic_cache.py](app/services/semantic_cache.py)). Entries live in Redis 8's
built-in vector index (HNSW, cosine). Lookup is a filtered KNN on tenant, reply language and
knowledge-base version; uploading or deleting a document bumps the version, so stale answers stop
matching at once.

Only answers that can't depend on who's asking are cached: the question has no first-person or
account words (*my, mere, मेरे, redeem, confirm …*), the agent used only `search_knowledge_base`,
and the answer doesn't contain the customer's name. Balances and redemptions always reach the LLM.

The threshold was calibrated with `scripts/calibrate_cache.py`. e5 similarities run high:
*"FD rate for senior citizens 1 year"* vs *"… 2 years"* scores **0.970**, above most true
paraphrases. So a hit also needs the **same numbers** in both questions:

| Rule | Paraphrases hit | False hits |
| --- | --- | --- |
| similarity ≥ 0.95 | 7/9 | 2/9 |
| similarity ≥ 0.965 + number guard (default) | 5/9 | **0/9** |

A cache hit returned in **1.8 ms with 0 tokens**. The same question uncached took ~2.9k tokens.
`"use_cache": false` in the request forces a fresh answer, for evals or "regenerate"; the fresh
answer still refreshes the cache.

### Reliability

| Failure | What happens |
| --- | --- |
| Primary LLM errors or sends nothing within 25 s | The round goes to `LLM_FALLBACK_PROVIDER`/`MODEL` (may be a different provider: message history converts both ways) |
| 3 consecutive failures | That provider's circuit opens: skipped for 30 s, then one trial request |
| Failure after text has streamed | Not retried elsewhere, since that would duplicate text; reported as an error |
| Every LLM down | `/chat` answers from the no-LLM baseline (balance or extractive search answer) with `degraded: true` |
| Slow search / balance tool | 15 s timeout, reported to the model as a tool error; redemptions are never cut off |
| Slow SQL | `statement_timeout` 15 s; Redis socket timeout 5 s |
| SDK-level | Anthropic / Gemini SDKs retry 429/5xx with backoff before a failure counts |

The `model` field of each reply shows which model actually answered. In testing, the free-tier
Gemini quota (20 requests/day on `gemini-2.5-flash`) ran out mid-run. The circuit opened after 3
failures and traffic moved to `gemini-3.5-flash-lite`. Before the fallback was configured
correctly, the same outage was served by the degraded baseline instead of returning errors.

### Points ledger

`points_ledger` is append-only, so balance is `SUM(delta)`. Redemptions lock the customer row
(`SELECT … FOR UPDATE`), so concurrent requests can't double-spend. A unique `(tenant_id, reference)`
makes each redemption idempotent.

## Run it

Requires Docker.

```bash
cp .env.example .env               # then set JWT_SECRET and GEMINI_API_KEY (or LLM_PROVIDER=claude + ANTHROPIC_API_KEY)
docker compose up -d --build        # postgres, redis, rabbitmq, migrate (one-shot), api, worker
docker compose exec api python -m scripts.seed          # demo tenants, customers, documents
docker compose exec api python -m scripts.reindex_text  # (re)build lexical + phonetic text for existing chunks
open http://localhost:8000/docs
```

The first build downloads CPU-only PyTorch and bakes the embedding and Whisper models into the
image (~2.5 GB). Host ports: API 8000, Postgres 5434, Redis 6380, RabbitMQ 5673 / UI 15673
(change them in `.env`).

### Try it

```bash
# Tokens (stand-in for an identity provider)
token() { docker compose exec -T api python -m scripts.issue_token "$@"; }
CUST=$(token --tenant demo-bank --role customer --customer CUST1001)
ADMIN=$(token --tenant demo-bank --role admin)
SECOND=$(token --tenant second-bank --role admin)
alias api='curl -s -H "Authorization: Bearer $CUST"'

# Hinglish, Hindi and English questions against the same documents
api localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "mere points kab expire honge", "top_k": 3}' | jq '.hits[] | {document_title, score, vector_rank, lexical_rank}'

api localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "मेरे पॉइंट्स कब खत्म होंगे"}' | jq '.hits[0].content'

# Upload a document (admin; PDF, DOCX, MD, TXT): 202 + job, then poll it
curl -s localhost:8000/documents -H "Authorization: Bearer $ADMIN" -F file=@policy.pdf
curl -s localhost:8000/documents/jobs/<job id> -H "Authorization: Bearer $ADMIN"

# Loyalty
api localhost:8000/customers/CUST1001/points
api localhost:8000/customers/CUST1002/points   # 403: not this customer's account
api localhost:8000/customers/CUST1001/redeem -H 'content-type: application/json' \
  -d '{"points": 500, "reference": "order-123"}'

# Agent chat (needs an LLM key)
api localhost:8000/chat -H 'content-type: application/json' \
  -d '{"customer_ref": "CUST1001", "message": "mere kitne points hai?"}'

# Streaming (SSE): watch tokens and tool calls arrive
curl -N -H "Authorization: Bearer $CUST" localhost:8000/chat/stream -H 'content-type: application/json' \
  -d '{"customer_ref": "CUST1001", "message": "FD rate for senior citizens 1 year?"}'

# Voice
api localhost:8000/voice/transcribe -F audio=@tests/fixtures/points_expiry_hi.wav
api localhost:8000/voice/chat -F customer_ref=CUST1001 -F audio=@tests/fixtures/points_question_en.wav

# Phase 1 keyword router, no LLM (kept as the eval baseline)
api localhost:8000/chat/baseline -H 'content-type: application/json' \
  -d '{"customer_ref": "CUST1001", "message": "mere kitne points hai?"}'

# Another tenant
curl -s localhost:8000/search -H "Authorization: Bearer $SECOND" -H 'content-type: application/json' -d '{"query": "points expire"}'
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/documents` | Upload a file: 202 + job (admin; 200 if already ingested) |
| GET | `/documents/jobs/{id}` | Ingestion job status and result (admin) |
| GET | `/documents` | List the tenant's documents |
| DELETE | `/documents/{id}` | Delete a document and its chunks |
| POST | `/search` | Hybrid / vector / lexical search with per-retriever ranks (`lexical` = keyword + phonetic) |
| GET | `/customers/{ref}/points` | Balance and points expiring in 30 days |
| POST | `/customers/{ref}/redeem` | Idempotent redemption |
| POST | `/chat` | Agent chat turn (JSON); `cached` / `degraded` flags, `use_cache: false` to bypass |
| POST | `/chat/stream` | Agent chat turn as Server-Sent Events |
| GET / DELETE | `/chat/memory/{ref}` | View / clear a customer's conversation memory |
| POST | `/chat/baseline` | Phase 1 keyword router (no LLM), the eval baseline |
| POST | `/voice/transcribe` | Speech to text (Hindi / English) |
| POST | `/voice/chat` | Speech to text, then an agent chat turn |
| GET | `/health/live`, `/health/ready` | Liveness; readiness checks Postgres and Redis (RabbitMQ reported, not required) |

All routes except `/health/*` need `Authorization: Bearer <JWT>` and are rate limited.

## Configuration

All settings are environment variables; see [.env.example](.env.example) and
[app/config.py](app/config.py). `EMBEDDING_DIM` must match the model, and the API refuses to start if
it doesn't. Switching to `BAAI/bge-m3` (1024-d) needs a new migration for `chunks.embedding` and
re-ingestion.

## Migrations

```bash
docker compose run --rm migrate alembic upgrade head
docker compose run --rm migrate alembic revision -m "describe change"
```

The `migrate` service runs `alembic upgrade head` on every `docker compose up`, before the API starts.

## Tests

```bash
uv venv --python 3.12 && uv pip install -r requirements-dev.txt
pytest tests   # unit tests (agent loop with a fake LLM, auth, circuit breaker, fallback chain,
               # cache rules) always run; integration tests need the stack + seed and mint tokens
               # with JWT_SECRET from .env. The LLM ones also need an LLM key in the API container.
               # On the Gemini free tier, run the LLM tests one at a time (5 requests/min, 20/day).
```

## Layout

```
app/
  agent/        LLM providers (Claude, Gemini) + fallback chain, agent loop, tools, prompts, chat turn
  api/          routers: chat (+stream, memory), chat_baseline, voice, documents, search, loyalty, health
  services/     text + language (detection, transliteration, phonetic keys), parsing, chunking,
                embeddings, ingestion, search, loyalty, memory, speech, baseline,
                auth, ratelimit, resilience, semantic_cache, jobs
  worker.py     RabbitMQ ingestion worker
  models.py     SQLAlchemy models        config.py   env settings        db.py   async engine
alembic/        migrations
data/seed/      fictional bank documents (English, Hinglish, Hindi) + tenants.json
scripts/        seed.py (idempotent demo data), reindex_text.py (rebuild lexical/phonetic text),
                issue_token.py (mint JWTs), calibrate_cache.py (semantic-cache threshold)
```

## Roadmap

- [x] Phase 1: pgvector, hybrid search, ingestion, Docker Compose, Alembic, env config
- [x] Phase 2: LLM tool calling, Redis conversation memory, transliteration, voice, SSE streaming
- [x] Phase 3: RabbitMQ ingestion workers, semantic cache, rate limits, circuit breaker + LLM fallback, JWT auth
- [ ] Phase 4: 100+ query eval set, LLM-as-judge CI gate, prompt-injection suite, audit log
- [ ] Phase 5: Langfuse, Prometheus + Grafana, load tests, Kubernetes on AWS
- [ ] Phase 6: demo, write-up

All bank names, rates and fees in `data/seed` are fictional.
