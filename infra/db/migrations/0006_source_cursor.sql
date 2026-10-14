-- Where a file-based source left off (GDELT GKG: the last 15-minute slot processed),
-- so a late or skipped scheduled run catches up instead of losing files. Service-only.
create table if not exists public.source_cursor (
  source      text primary key,
  cursor      text not null,
  updated_at  timestamptz not null default now()
);
alter table public.source_cursor enable row level security;
