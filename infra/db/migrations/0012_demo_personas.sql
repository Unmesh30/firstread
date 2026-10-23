-- Demo readers built from live articles (SPEC section 8, ADR 0018). Ingestion refreshes
-- them once a day from the runner-side window; the API copies a persona's articles
-- into a reader's history on activation. Service-only: RLS on, no policy, no grants.
create table if not exists public.demo_personas (
  id           text primary key check (id ~ '^[a-z0-9-]{1,40}$'),
  name         text not null,
  description  text not null,
  topics       text[] not null check (topics <@ array[
                 'us', 'world', 'politics', 'business', 'tech_science', 'health', 'sports',
                 'entertainment', 'lifestyle', 'food_drink', 'travel', 'autos', 'weather']),
  refreshed_at timestamptz not null
);

-- A copy of each selected article (vector included, copied server-side), so a persona
-- outlives the 14-day article retention between refreshes, like user_history does.
create table if not exists public.persona_articles (
  persona_id       text not null references public.demo_personas (id) on delete cascade,
  position         integer not null check (position >= 0),
  article_id       uuid not null,  -- no FK: articles expire first
  title            text not null,
  source_name      text not null,
  source_domain    text not null,
  url              text not null,
  category         text not null,
  story_cluster_id uuid not null,
  text_source      text not null check (text_source in ('title', 'title_summary')),
  embedding        extensions.halfvec(384) not null,
  primary key (persona_id, position),
  unique (persona_id, article_id)
);

alter table public.demo_personas enable row level security;
alter table public.persona_articles enable row level security;
-- Data API access (Supabase only; the roles don't exist locally): none.
do $$
begin
  if exists (select from pg_roles where rolname = 'anon') then
    revoke all on table public.demo_personas, public.persona_articles from anon, authenticated;
  end if;
end
$$;
