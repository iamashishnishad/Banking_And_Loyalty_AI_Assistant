# Banking & Loyalty AI Assistant

A Hindi / Hinglish / English assistant for banking and rewards questions, built as a multi-tenant
backend: one bank per tenant, documents and customers isolated per tenant.

**Status: Phases 1–2 complete.**
- **Phase 1:** hybrid retrieval over Postgres + pgvector, document ingestion, Docker Compose, Alembic, env config.
- **Phase 2:** an LLM tool-calling agent (Gemini or Claude) with Redis conversation memory,
  Hindi/Hinglish detection and transliteration, voice input via local Whisper, and SSE streaming.

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
                         Postgres 17 + pgvector 0.8                   Redis 7
                         tenants, documents, chunks,                  conversation memory, turn counter,
                         customers, points_ledger                     pending redemptions, query-embedding cache
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

### Multi-tenancy

Every table carries `tenant_id`, and every query filters on it. `chunks.tenant_id` is denormalised
so search doesn't need a join. HNSW runs with `hnsw.iterative_scan = relaxed_order` so the tenant
filter can't starve the top-k. In Phase 1 the tenant comes from the `X-Tenant` header; Phase 3 takes
it from the JWT.

### Points ledger

`points_ledger` is append-only, so balance is `SUM(delta)`. Redemptions lock the customer row
(`SELECT … FOR UPDATE`), so concurrent requests can't double-spend. A unique `(tenant_id, reference)`
makes each redemption idempotent.

## Run it

Requires Docker.

```bash
cp .env.example .env               # then set GEMINI_API_KEY (or LLM_PROVIDER=claude + ANTHROPIC_API_KEY)
docker compose up -d --build        # postgres, redis, migrate (one-shot), api
docker compose exec api python -m scripts.seed          # demo tenants, customers, documents
docker compose exec api python -m scripts.reindex_text  # (re)build lexical + phonetic text for existing chunks
open http://localhost:8000/docs
```

The first build downloads CPU-only PyTorch and bakes the embedding and Whisper models into the
image (~2.5 GB). Host ports: API 8000, Postgres 5434, Redis 6380 (change them in `.env`).

### Try it

```bash
# Hinglish, Hindi and English questions against the same documents
curl -s localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "mere points kab expire honge", "top_k": 3}' | jq '.hits[] | {document_title, score, vector_rank, lexical_rank}'

curl -s localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "मेरे पॉइंट्स कब खत्म होंगे"}' | jq '.hits[0].content'

# Upload a document (PDF, DOCX, MD, TXT)
curl -s localhost:8000/documents -F file=@policy.pdf

# Loyalty
curl -s localhost:8000/customers/CUST1001/points
curl -s localhost:8000/customers/CUST1001/redeem -H 'content-type: application/json' \
  -d '{"points": 500, "reference": "order-123"}'

# Agent chat (needs an LLM key)
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"customer_ref": "CUST1001", "message": "mere kitne points hai?"}'

# Streaming (SSE): watch tokens and tool calls arrive
curl -N localhost:8000/chat/stream -H 'content-type: application/json' \
  -d '{"customer_ref": "CUST1001", "message": "FD rate for senior citizens 1 year?"}'

# Voice
curl -s localhost:8000/voice/transcribe -F audio=@tests/fixtures/points_expiry_hi.wav
curl -s localhost:8000/voice/chat -F customer_ref=CUST1001 -F audio=@tests/fixtures/points_question_en.wav

# Phase 1 keyword router, no LLM (kept as the eval baseline)
curl -s localhost:8000/chat/baseline -H 'content-type: application/json' \
  -d '{"customer_ref": "CUST1001", "message": "mere kitne points hai?"}'

# Another tenant
curl -s localhost:8000/search -H 'X-Tenant: second-bank' -H 'content-type: application/json' -d '{"query": "points expire"}'
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/documents` | Upload and ingest a file (deduplicated by SHA-256 per tenant) |
| GET | `/documents` | List the tenant's documents |
| DELETE | `/documents/{id}` | Delete a document and its chunks |
| POST | `/search` | Hybrid / vector / lexical search with per-retriever ranks (`lexical` = keyword + phonetic) |
| GET | `/customers/{ref}/points` | Balance and points expiring in 30 days |
| POST | `/customers/{ref}/redeem` | Idempotent redemption |
| POST | `/chat` | Agent chat turn (JSON) |
| POST | `/chat/stream` | Agent chat turn as Server-Sent Events |
| GET / DELETE | `/chat/memory/{ref}` | View / clear a customer's conversation memory |
| POST | `/chat/baseline` | Phase 1 keyword router (no LLM), the eval baseline |
| POST | `/voice/transcribe` | Speech to text (Hindi / English) |
| POST | `/voice/chat` | Speech to text, then an agent chat turn |
| GET | `/health/live`, `/health/ready` | Liveness; readiness checks Postgres and Redis |

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
pytest tests   # unit tests (incl. agent loop with a fake LLM) always run; integration tests need
               # the stack + seed, and the LLM ones also need an LLM key in the API container.
               # On the Gemini free tier, run the LLM tests one at a time (5 requests/min).
```

## Layout

```
app/
  agent/        LLM providers (Claude, Gemini), agent loop, tools, prompts, chat-turn orchestration
  api/          routers: chat (+stream, memory), chat_baseline, voice, documents, search, loyalty, health
  services/     text + language (detection, transliteration, phonetic keys), parsing, chunking,
                embeddings, ingestion, search, loyalty, memory, speech
  models.py     SQLAlchemy models        config.py   env settings        db.py   async engine
alembic/        migrations
data/seed/      fictional bank documents (English, Hinglish, Hindi) + tenants.json
scripts/        seed.py (idempotent demo data), reindex_text.py (rebuild lexical/phonetic text)
```

## Roadmap

- [x] Phase 1: pgvector, hybrid search, ingestion, Docker Compose, Alembic, env config
- [x] Phase 2: LLM tool calling, Redis conversation memory, transliteration, voice, SSE streaming
- [ ] Phase 3: RabbitMQ ingestion workers, semantic cache, rate limits, circuit breaker, JWT auth
- [ ] Phase 4: 100+ query eval set, LLM-as-judge CI gate, prompt-injection suite, audit log
- [ ] Phase 5: Langfuse, Prometheus + Grafana, load tests, Kubernetes on AWS
- [ ] Phase 6: demo, write-up

All bank names, rates and fees in `data/seed` are fictional.
