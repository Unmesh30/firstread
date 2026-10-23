-- Every served feed page (ADR 0016): ids, positions and the blend's components, so
-- events can be joined to what was shown. Kept 30 days, like events.
create table if not exists public.served_slates (
  request_id    text primary key,
  slate_id      uuid not null,
  user_id       uuid not null references auth.users (id) on delete cascade,
  page          integer not null check (page >= 0),
  model_version text not null,
  article_ids   uuid[] not null,
  -- Four per item, in order: model, fresh, trend, final.
  components    real[] not null,
  created_at    timestamptz not null default now(),
  check (cardinality(components) = 4 * cardinality(article_ids))
);
create index if not exists served_slates_user_created_idx
  on public.served_slates (user_id, created_at desc);
create index if not exists served_slates_slate_idx on public.served_slates (slate_id);
create index if not exists served_slates_created_idx on public.served_slates (created_at);
alter table public.served_slates enable row level security;
drop policy if exists served_slates_owner_read on public.served_slates;
create policy served_slates_owner_read on public.served_slates
  for select using (user_id = auth.uid());
-- Data API access (Supabase only; the roles don't exist locally): the owner may read
-- their own rows, nobody may write. New public tables are no longer exposed to the
-- Data API by default on Supabase (2026), so access is explicit either way.
do $$
begin
  if exists (select from pg_roles where rolname = 'anon') then
    revoke all on table public.served_slates from anon, authenticated;
    grant select on table public.served_slates to authenticated;
  end if;
end
$$;
