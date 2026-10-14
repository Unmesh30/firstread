-- Which domains covered which story, including dropped duplicates, so trending
-- counts a wire story run by 30 outlets as 30 domains (ADR 0010). First sighting wins.
create table if not exists public.story_coverage (
  story_cluster_id uuid not null,
  source_domain    text not null,
  first_seen_at    timestamptz not null,
  primary key (story_cluster_id, source_domain)
);
create index if not exists story_coverage_first_seen_idx on public.story_coverage (first_seen_at);
alter table public.story_coverage enable row level security;
