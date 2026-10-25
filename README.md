# Banking & Loyalty AI Assistant

A Hindi / Hinglish / English assistant for banking and rewards questions, built as a multi-tenant
backend: one bank per tenant, documents and customers isolated per tenant.

**Status: Phase 1 (foundation) complete.** Hybrid retrieval over Postgres + pgvector, document
ingestion, Docker Compose, Alembic migrations, env-based config. Chat is still the keyword-routed,
extractive baseline; Phase 2 replaces it with an LLM tool-calling agent.

## Architecture

```
            ┌──────────────────────── FastAPI (api) ────────────────────────┐
 client ──▶ │ /chat  /search  /documents  /customers/{ref}/points|redeem    │
            │                                                               │
            │  ingestion: parse (PDF/DOCX/MD/TXT) → chunk → embed → store   │
            │  search:    query → normalise ─┬─▶ full-text (tsvector, GIN)  │
            │                                └─▶ embedding ─▶ pgvector HNSW │
            │                          RRF fusion of both rankings ◀─┘      │
            └───────────────┬────────────────────────────────┬──────────────┘
                            │                                │
                 Postgres 17 + pgvector 0.8             Redis 7
                 tenants, documents, chunks,            query-embedding cache
                 customers, points_ledger
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
cp .env.example .env
docker compose up -d --build        # postgres, redis, migrate (one-shot), api
docker compose exec api python -m scripts.seed   # demo tenants, customers, documents
open http://localhost:8000/docs
```

The first build downloads CPU-only PyTorch and bakes the embedding model into the image (~2 GB).

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

# Baseline chat
curl -s localhost:8000/chat -H 'content-type: application/json' \
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
| POST | `/search` | Hybrid / vector / lexical search with per-retriever ranks |
| GET | `/customers/{ref}/points` | Balance and points expiring in 30 days |
| POST | `/customers/{ref}/redeem` | Idempotent redemption |
| POST | `/chat` | Baseline keyword-routed chat |
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
pytest tests                       # unit tests always run; integration tests need the stack + seed
```

## Layout

```
app/
  api/          routers: chat, documents, search, loyalty, health; deps (tenant, session, redis)
  services/     text normalisation, parsing, chunking, embeddings, ingestion, search, loyalty
  models.py     SQLAlchemy models        config.py   env settings        db.py   async engine
alembic/        migrations
data/seed/      fictional bank documents (English, Hinglish, Hindi) + tenants.json
scripts/seed.py idempotent demo data loader
```

## Roadmap

- [x] Phase 1: pgvector, hybrid search, ingestion, Docker Compose, Alembic, env config
- [ ] Phase 2: LLM tool calling, Redis conversation memory, transliteration, voice, SSE streaming
- [ ] Phase 3: RabbitMQ ingestion workers, semantic cache, rate limits, circuit breaker, JWT auth
- [ ] Phase 4: 100+ query eval set, LLM-as-judge CI gate, prompt-injection suite, audit log
- [ ] Phase 5: Langfuse, Prometheus + Grafana, load tests, Kubernetes on AWS
- [ ] Phase 6: demo, write-up

All bank names, rates and fees in `data/seed` are fictional.
