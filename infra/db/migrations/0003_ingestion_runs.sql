-- One row per ingestion run (SPEC section 9 + Phase 2 counters). Service-only.
create table if not exists public.ingestion_runs (
  id               uuid primary key default gen_random_uuid(),
  started_at       timestamptz not null default now(),
  finished_at      timestamptz,
  source           text not null default 'all',
  window_start     timestamptz,
  window_end       timestamptz,
  fetched          integer not null default 0,
  inserted         integer not null default 0,
  duplicates       integer not null default 0,
  near_duplicates  integer not null default 0,
  clustered        integer not null default 0,
  errors           integer not null default 0,
  gdelt_calls      integer not null default 0,
  redis_commands   integer not null default 0,
  status           text not null default 'running'
                     check (status in ('running', 'ok', 'partial', 'failed')),
  details          jsonb not null default '{}'::jsonb
);
create index if not exists ingestion_runs_started_idx on public.ingestion_runs (started_at desc);
alter table public.ingestion_runs enable row level security;
