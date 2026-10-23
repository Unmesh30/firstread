-- One row per reader (SPEC section 9, ADR 0016). Cascades from auth.users.
create table if not exists public.profiles (
  user_id        uuid primary key references auth.users (id) on delete cascade,
  topics         text[] not null default '{}' check (topics <@ array[
                   'us', 'world', 'politics', 'business', 'tech_science', 'health', 'sports',
                   'entertainment', 'lifestyle', 'food_drink', 'travel', 'autos', 'weather']),
  -- A preference, like topics: survives Redis loss and the 7-day Redis TTL.
  hidden_sources text[] not null default '{}',
  is_anonymous   boolean not null default true,
  active_persona text,
  -- Bumped once per history row written; every Redis history entry carries the
  -- version it was given, so a lost or out-of-order update shows up as a gap.
  hist_version   bigint not null default 0,
  created_at     timestamptz not null default now(),
  updated_at     timestamptz not null default now(),
  last_seen_at   timestamptz not null default now()
);
create index if not exists profiles_last_seen_idx on public.profiles (last_seen_at);
alter table public.profiles enable row level security;
drop policy if exists profiles_owner_read on public.profiles;
create policy profiles_owner_read on public.profiles for select using (user_id = auth.uid());
-- Data API access (Supabase only; the roles don't exist locally): the owner may read
-- their own rows, nobody may write. New public tables are no longer exposed to the
-- Data API by default on Supabase (2026), so access is explicit either way.
do $$
begin
  if exists (select from pg_roles where rolname = 'anon') then
    revoke all on table public.profiles from anon, authenticated;
    grant select on table public.profiles to authenticated;
  end if;
end
$$;
