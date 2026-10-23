-- Client events (SPEC section 9), idempotent by the client's event_id (ADR 0016).
-- Impressions are kept 7 days, everything else 30 (ingestion retention + size guard).
create table if not exists public.events (
  id            uuid primary key,
  user_id       uuid not null references auth.users (id) on delete cascade,
  article_id    uuid,  -- no FK: articles expire first
  type          text not null check (type in ('impression', 'click', 'like', 'hide', 'dwell')),
  scope         text check (scope in ('article', 'source')),  -- hide only
  position      integer check (position between 0 and 999),
  dwell_ms      integer check (dwell_ms between 0 and 3600000),
  request_id    text,
  model_version text,
  client_ts     timestamptz,
  created_at    timestamptz not null default now()
);
create index if not exists events_user_created_idx on public.events (user_id, created_at desc);
create index if not exists events_type_created_idx on public.events (type, created_at);
alter table public.events enable row level security;
drop policy if exists events_owner_read on public.events;
create policy events_owner_read on public.events for select using (user_id = auth.uid());
-- Data API access (Supabase only; the roles don't exist locally): the owner may read
-- their own rows, nobody may write. New public tables are no longer exposed to the
-- Data API by default on Supabase (2026), so access is explicit either way.
do $$
begin
  if exists (select from pg_roles where rolname = 'anon') then
    revoke all on table public.events from anon, authenticated;
    grant select on table public.events to authenticated;
  end if;
end
$$;
