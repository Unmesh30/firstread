-- Live articles (SPEC section 9, ADR 0006/0008). One row per canonical URL.
create table if not exists public.articles (
  id                  uuid primary key default gen_random_uuid(),
  url                 text not null,
  url_canonical       text not null unique,
  title               text not null,
  -- Always NULL in v1: publisher text beyond the headline is never stored (SPEC section 4).
  summary             text,
  source_name         text not null,
  source_domain       text not null,
  image_url           text,
  language            text not null default 'en',
  source_country      char(2),            -- ISO 3166-1 alpha-2
  geo                 text not null check (geo in ('us', 'intl')),
  category            text not null check (category in (
                        'us', 'world', 'politics', 'business', 'tech_science', 'health', 'sports',
                        'entertainment', 'lifestyle', 'food_drink', 'travel', 'autos', 'weather')),
  category_confidence real not null check (category_confidence between 0 and 1),
  category_source     text not null check (category_source in ('model', 'feed_hint', 'geo_rule')),
  published_at        timestamptz,
  first_seen_at       timestamptz not null,
  embedding           extensions.halfvec(384) not null,
  embedding_model     text not null,
  text_source         text not null check (text_source in ('title', 'title_summary')),
  title_fingerprint   text not null,
  story_cluster_id    uuid not null,
  origin              text not null check (origin in ('gdelt', 'rss')),
  created_at          timestamptz not null default now()
);

-- Embeddings are L2-normalized, so inner product = cosine (CLAUDE.md gotcha).
create index if not exists articles_embedding_hnsw
  on public.articles using hnsw (embedding extensions.halfvec_ip_ops)
  with (m = 16, ef_construction = 64);
create index if not exists articles_first_seen_idx on public.articles (first_seen_at desc);
create index if not exists articles_category_first_seen_idx
  on public.articles (category, first_seen_at desc);
create index if not exists articles_fingerprint_idx
  on public.articles (title_fingerprint, first_seen_at);
create index if not exists articles_cluster_idx on public.articles (story_cluster_id);

-- Public read; writes only by roles that bypass RLS (the service role / owner).
alter table public.articles enable row level security;
drop policy if exists articles_public_read on public.articles;
create policy articles_public_read on public.articles for select using (true);
