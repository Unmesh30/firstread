-- pgvector in the `extensions` schema: Supabase's convention, reproduced locally so
-- the same SQL runs in both places. Everything references it as extensions.<name>.
create schema if not exists extensions;
create extension if not exists vector with schema extensions;
