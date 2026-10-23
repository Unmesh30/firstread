-- The durable copy of each reader's history (ADR 0016): the clicked article's vector,
-- title and topic are copied at click time, so a history outlives the 14-day article
-- retention (ADR 0008 note). At most 50 rows per reader (trimmed on write).
create table if not exists public.user_history (
  user_id          uuid not null references auth.users (id) on delete cascade,
  article_id       uuid not null,  -- no FK: articles expire after 14 days
  version          bigint not null,  -- profiles.hist_version when written
  kind             text not null check (kind in ('click', 'like', 'persona')),
  clicked_at       timestamptz not null,
  title            text not null,
  source_name      text not null,
  source_domain    text not null,
  url              text not null,
  category         text not null,
  story_cluster_id uuid not null,
  text_source      text not null check (text_source in ('title', 'title_summary')),
  embedding        extensions.halfvec(384) not null,
  primary key (user_id, article_id)
);
create index if not exists user_history_user_version_idx
  on public.user_history (user_id, version desc);
alter table public.user_history enable row level security;
drop policy if exists user_history_owner_read on public.user_history;
create policy user_history_owner_read on public.user_history
  for select using (user_id = auth.uid());
-- Data API access (Supabase only; the roles don't exist locally): the owner may read
-- their own rows, nobody may write. New public tables are no longer exposed to the
-- Data API by default on Supabase (2026), so access is explicit either way.
do $$
begin
  if exists (select from pg_roles where rolname = 'anon') then
    revoke all on table public.user_history from anon, authenticated;
    grant select on table public.user_history to authenticated;
  end if;
end
$$;
