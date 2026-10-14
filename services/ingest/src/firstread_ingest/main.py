"""``firstread-ingest``: apply migrations, or run one ingestion pass.

uv run firstread-ingest migrate
uv run firstread-ingest run [--sources rss] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence

from firstread_core.db import apply_migrations, connect, default_migrations_dir
from firstread_ingest.config import Settings
from firstread_ingest.models import PINNED, TopicClassifier, download_release, verify_release

SOURCES = ("rss",)  # GDELT joins once its adapter is chosen (Phase 2 probe results)


def describe() -> str:
    """One line describing the pipeline."""
    return "fetch -> normalize -> dedupe -> embed -> classify -> cluster -> upsert"


async def _migrate(settings: Settings) -> list[str]:
    conn = await connect(settings.database_url.get_secret_value())
    try:
        return await apply_migrations(conn, default_migrations_dir())
    finally:
        await conn.close()


def load_classifier(settings: Settings) -> TopicClassifier:
    tag = settings.ingest_model_tag
    if settings.ingest_model_dir is not None:
        verify_release(settings.ingest_model_dir, PINNED[tag]["manifest_sha256"])
        return TopicClassifier.load(settings.ingest_model_dir)
    return TopicClassifier.load(download_release(tag))


async def _run(settings: Settings, sources: Sequence[str], dry_run: bool) -> dict[str, object]:
    from firstread_ingest.pipeline import run

    summary = await run(settings, load_classifier(settings), sources=sources, dry_run=dry_run)
    return summary.as_dict()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="firstread-ingest", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate", help="apply infra/db/migrations to DATABASE_URL")
    p = sub.add_parser("run", help="run one ingestion pass")
    p.add_argument("--sources", default=",".join(SOURCES), help="comma-separated: rss")
    p.add_argument("--dry-run", action="store_true", help="fetch, embed, classify; write nothing")
    args = parser.parse_args(argv)
    settings = Settings()
    if args.command == "migrate":
        applied = asyncio.run(_migrate(settings))
        print(f"applied {len(applied)} migration(s): {', '.join(applied) or 'none pending'}")
        return
    sources = [s for s in args.sources.split(",") if s]
    unknown = sorted(set(sources) - set(SOURCES))
    if unknown:
        parser.error(f"unknown sources {unknown}; available: {list(SOURCES)}")
    print(json.dumps(asyncio.run(_run(settings, sources, args.dry_run)), indent=1, default=str))


if __name__ == "__main__":
    main()
