# ADR 0008: Postgres schema, migrations and row-level security

- Status: Accepted
- Date: 2026-10-14

## Context

Ingestion (P2) writes articles and bookkeeping to Postgres. The API (P3) reads them, including pgvector ANN search. The same SQL must run on the local Docker stack (`pgvector/pgvector:0.8.7-pg17`) and on **Supabase** (P5). Supabase differs from local Postgres in ways that matter:
- `postgres` is not a superuser;
- extensions live in the `extensions` schema;
- the `anon` / `authenticated` roles can reach any `public` table that lacks row-level security, through the Data API.

We also want article embeddings stored compactly: 384 floats per article.

## Decision

- **Plain SQL migrations** in `infra/db/migrations/NNNN_name.sql`, applied in order by `firstread_core.db.apply_migrations`:
  - each file runs in its own transaction, under an advisory lock;
  - each applied file's SHA256 is recorded in `schema_migrations`, and the runner **refuses if an applied file was edited**: migrations are append-only;
  - commands: `make db-migrate` / `firstread-ingest migrate`.
- **pgvector in the `extensions` schema**, locally as on Supabase. Every reference is qualified: `extensions.halfvec(384)`, `extensions.halfvec_ip_ops`, `OPERATOR(extensions.<#>)`. The local `search_path` doesn't include `extensions` (CLAUDE.md gotcha).
- **Embeddings as `halfvec(384)`:** half the size of `vector`, and float16 precision is ample for L2-normalized vectors. An **HNSW index with `halfvec_ip_ops`**: inner product equals cosine on normalized vectors (m = 16, ef_construction = 64).
- **Tables:**
  - `articles` (SPEC §9, plus `category_source`, `geo`, `text_source`, `title_fingerprint`; `source_country` is ISO 3166-1 alpha-2; `summary` is always NULL in v1);
  - `ingestion_runs`;
  - `feed_state` (RSS conditional-GET state survives between Actions runs);
  - `story_coverage` (which outlets carried which story, for trending);
  - `source_cursor` (the GKG catch-up cursor);
  - `schema_migrations`.
- **RLS on every table.** Only `articles` has a policy (`for select using (true)`, public read, no role names, so it works on Supabase and locally). Every other table has RLS and no policy: invisible to `anon`/`authenticated`, reachable only by the owner or service role.
  - An integration test fails if any `public` table has RLS off.
  - A CI job applies the migrations to the real **`supabase/postgres:17.11.0.004`** image as non-superuser `postgres`, and checks that anon/authenticated can read articles, see no rows in the service tables, and write nothing.
- **Idempotent writes:** `insert … select from unnest(…) on conflict (url_canonical) do nothing returning url_canonical` is one statement per batch, and it reports which rows were actually new.
- **asyncpg with `statement_cache_size=0`** everywhere (`firstread_core.db.connect`), because Supabase's transaction-mode pooler breaks prepared-statement caching.
- **Dedicated database roles are deferred to P5.** Creating them needs passwords, which are secrets and don't belong in migrations.

## Alternatives considered

- **A migration framework (Alembic, sqitch, Supabase CLI).** These add a dependency, and on Supabase a second source of truth. Plain SQL with a 60-line runner is enough for one service writing a handful of tables, and the SHA check gives the safety that matters.
- **`vector(384)` (float32).** Twice the storage and index size for no ranking benefit at float16 precision.
- **IVFFlat index.** Needs training on existing data and tuning of `lists`; HNSW works from an empty table and has better recall/latency at our size.
- **RLS only on user tables.** Unsafe on Supabase: any table without RLS is open to the anon key.

## Consequences

- Measured footprint: **about 3.2 KB per article**, half heap and half indexes (HNSW about 1.2 KB). 14-day retention at about 5,000 articles a day stays well under the 400 MB target (ADR 0007).
- P3 queries must qualify pgvector operators with `extensions`.
- Editing a shipped migration is a hard error. Schema changes are always a new numbered file.
- P5 adds a service role and grants in a migration that reads no secrets (passwords are set outside SQL).
