"""``firstread-ingest``: apply migrations, run one ingestion pass, or replay GKG archives.

    uv run firstread-ingest migrate
    uv run firstread-ingest run [--sources rss,gdelt] [--dry-run]
    uv run firstread-ingest replay [--days 3] [--fresh]

``run`` prints the summary as JSON and, when ``$GITHUB_STEP_SUMMARY`` is set (GitHub
Actions), appends it as Markdown. Exit codes: 0 for ok/partial (partial also prints a
warning annotation), 1 for failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from firstread_core.db import apply_migrations, check_migration_dsn, connect, default_migrations_dir
from firstread_ingest.config import Settings
from firstread_ingest.models import PINNED, TopicClassifier, download_release, verify_release

SOURCES = ("rss", "gdelt")


def describe() -> str:
    """One line describing the pipeline."""
    return "fetch -> normalize -> dedupe -> embed -> classify -> cluster -> upsert"


async def _migrate(settings: Settings) -> list[str]:
    dsn = settings.database_url.get_secret_value()
    check_migration_dsn(dsn)
    conn = await connect(dsn)
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
    p.add_argument("--sources", default=",".join(SOURCES), help="comma-separated: rss,gdelt")
    p.add_argument("--dry-run", action="store_true", help="fetch, embed, classify; write nothing")
    p.add_argument(
        "--require-db",
        action="store_true",
        help="fail fast if DATABASE_URL/REDIS_URL aren't set (GitHub Actions)",
    )
    p = sub.add_parser("replay", help="replay archived GDELT GKG files into a separate DB")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--end", default=None, help="last slot, YYYYMMDDHHMMSS (default: 45 min ago)")
    p.add_argument("--db", default="firstread_replay")
    p.add_argument("--redis-db", type=int, default=1)
    p.add_argument("--fresh", action="store_true", help="drop the replay DB and start over")
    args = parser.parse_args(argv)
    if args.command == "replay":
        from firstread_ingest import replay
        from firstread_ingest.sources.gkg import parse_slot

        end = parse_slot(args.end) if args.end else None
        asyncio.run(replay.main(args.days, end, args.db, args.redis_db, args.fresh))
        return
    settings = Settings()
    if args.command == "migrate":
        applied = asyncio.run(_migrate(settings))
        print(f"applied {len(applied)} migration(s): {', '.join(applied) or 'none pending'}")
        return
    sources = [s for s in args.sources.split(",") if s]
    unknown = sorted(set(sources) - set(SOURCES))
    if unknown:
        parser.error(f"unknown sources {unknown}; available: {list(SOURCES)}")
    if not args.dry_run and args.require_db:
        missing = [v for v in ("DATABASE_URL", "REDIS_URL") if not os.environ.get(v)]
        if missing:
            print(f"::error::{', '.join(missing)} not set; use --dry-run until P5 adds the "
                  "cloud database and Redis secrets")  # fmt: skip
            raise SystemExit(1)
    summary = asyncio.run(_run(settings, sources, args.dry_run))
    print(json.dumps(summary, indent=1, default=str))
    raise SystemExit(finish(summary, args.dry_run))


def finish(summary: dict[str, object], dry_run: bool) -> int:
    """Write the job summary (in GitHub Actions) and choose the exit code."""
    from firstread_ingest.report import markdown

    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with Path(path).open("a") as f:
            f.write(markdown(summary, dry_run))
    status = summary.get("status")
    if status == "partial":
        print("::warning::ingestion finished as partial; see the job summary", file=sys.stderr)
    return 1 if status == "failed" else 0


if __name__ == "__main__":
    main()
