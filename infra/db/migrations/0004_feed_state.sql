-- Conditional-GET state per RSS feed; must survive between GitHub Actions runs.
create table if not exists public.feed_state (
  feed_id              text primary key,
  etag                 text,
  last_modified        text,
  last_status          integer,
  last_fetched_at      timestamptz,
  consecutive_failures integer not null default 0
);
alter table public.feed_state enable row level security;
