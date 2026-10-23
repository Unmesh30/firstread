-- LOCAL ONLY: a stand-in for Supabase's auth schema (ADR 0016). Supabase owns `auth`
-- and, since 2025-04-21, allows no new tables or functions in it (foreign keys to it
-- are fine). So this block does nothing where `auth` already exists (hosted Supabase,
-- the supabase/postgres CI image) and only creates what our foreign keys and
-- owner-only policies reference on the plain pgvector image.
do $$
begin
  if to_regnamespace('auth') is null then
    create schema auth;
    create table auth.users (
      id           uuid primary key,
      is_anonymous boolean not null default false,
      created_at   timestamptz not null default now()
    );
    -- Same claim lookup as Supabase: the legacy setting, then the JSON claims.
    create function auth.uid() returns uuid language sql stable as $f$
      select nullif(coalesce(
        current_setting('request.jwt.claim.sub', true),
        nullif(current_setting('request.jwt.claims', true), '')::jsonb ->> 'sub'
      ), '')::uuid
    $f$;
  end if;
end
$$;
