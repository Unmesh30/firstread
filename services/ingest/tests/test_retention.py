"""Retention against a throwaway Postgres: what goes, what stays, batching."""

import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from firstread_core.db import apply_migrations, connect
from firstread_core.dbtest import MIGRATIONS, skip_or_fail_if_unreachable_async, temp_database
from firstread_ingest import store

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
pytestmark = pytest.mark.integration


def article(age: timedelta, n: int) -> store.ArticleRow:
    aid = uuid.uuid4()
    v = np.zeros(384, np.float32)
    v[n % 384] = 1.0
    return store.ArticleRow(
        id=aid, url=f"https://example.com/{n}", url_canonical=f"https://example.com/{n}",
        title=f"t{n}", source_name="x", source_domain="example.com", image_url=None,
        language="en", source_country="US", geo="us", category="us", category_confidence=0.9,
        category_source="model", published_at=None, first_seen_at=NOW - age, embedding=v,
        embedding_model="m", text_source="title", title_fingerprint=f"fp{n}",
        story_cluster_id=aid, origin="gdelt",
    )  # fmt: skip


async def test_retention_deletes_expired_rows_in_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    await skip_or_fail_if_unreachable_async()
    monkeypatch.setattr(store, "RETENTION_BATCH", 3)  # force several batches
    async with temp_database() as dsn:
        conn = await connect(dsn)
        try:
            await apply_migrations(conn, MIGRATIONS)
            await store.register_codecs(conn)
            old = [article(timedelta(days=15), i) for i in range(7)]
            new = [article(timedelta(days=13), 100 + i) for i in range(2)]
            await store.insert_articles(conn, old + new)
            cov = [(uuid.uuid4(), f"d{i}.com", NOW - timedelta(hours=49)) for i in range(5)]
            cov += [(uuid.uuid4(), "fresh.com", NOW - timedelta(hours=47))]
            await store.insert_coverage(conn, cov)
            await conn.execute(
                "insert into public.ingestion_runs (started_at) values ($1), ($2)",
                NOW - timedelta(days=31), NOW - timedelta(days=1),
            )  # fmt: skip
            await conn.execute(
                "insert into public.feed_state (feed_id) values ('kept'), ('removed-feed')"
            )
            out = await store.apply_retention(
                conn, NOW, articles=timedelta(days=14), coverage=timedelta(hours=48),
                runs=timedelta(days=30), feed_ids=["kept"],
            )  # fmt: skip
            assert out == {"articles": 7, "story_coverage": 5, "ingestion_runs": 1, "feed_state": 1}
            assert await conn.fetchval("select count(*) from public.articles") == 2
            assert await conn.fetchval("select count(*) from public.story_coverage") == 1
            assert await conn.fetchval("select feed_id from public.feed_state") == "kept"
            # Coverage retention is independent of article retention: 48 h vs 14 d.
            again = await store.apply_retention(
                conn, NOW, articles=timedelta(days=14), coverage=timedelta(hours=48),
                runs=timedelta(days=30),
            )  # fmt: skip
            assert again == {"articles": 0, "story_coverage": 0, "ingestion_runs": 0}
            size = await store.database_size(conn)
            assert size["db"] > size["articles"] > 0
        finally:
            await conn.close()


# ---- user data and the size guard (ADR 0016) ----

MB = 1_000_000


@pytest.mark.parametrize(
    ("size_mb", "step", "impressions", "events", "slates", "articles"),
    [
        (200, 0, 7, 30, 30, 14),
        (351, 1, 2, 30, 30, 14),
        (376, 2, 2, 14, 7, 14),
        (401, 3, 2, 14, 7, 10),
    ],
)
def test_the_size_guard_tightens_in_steps(
    size_mb: int, step: int, impressions: float, events: float, slates: float, articles: float
) -> None:
    r, got = store.guard(size_mb * MB, store.UserRetention())
    assert got == step
    assert (r.impressions, r.events, r.slates, r.articles) == (
        impressions,
        events,
        slates,
        articles,
    )
    assert r.inactive_anonymous == 30


def test_the_guard_never_lengthens_shorter_settings() -> None:
    short = store.UserRetention(impressions=1, events=5, slates=3, articles=7)
    r, _ = store.guard(450 * MB, short)
    assert (r.impressions, r.events, r.slates, r.articles) == (1, 5, 3, 7)


async def test_user_retention_and_inactive_anonymous_readers() -> None:
    await skip_or_fail_if_unreachable_async()
    async with temp_database() as dsn:
        conn = await connect(dsn)
        try:
            await apply_migrations(conn, MIGRATIONS)
            active, gone, signed_in = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            for uid, anon, seen in ((active, True, 1), (gone, True, 40), (signed_in, False, 40)):
                await conn.execute("insert into auth.users (id) values ($1)", uid)
                await conn.execute(
                    "insert into public.profiles (user_id, is_anonymous, last_seen_at) "
                    "values ($1, $2, $3)",
                    uid, anon, NOW - timedelta(days=seen),
                )  # fmt: skip
            for kind, age in (("impression", 3), ("impression", 8), ("click", 8), ("click", 31)):
                await conn.execute(
                    "insert into public.events (id, user_id, article_id, type, created_at) "
                    "values ($1, $2, $3, $4, $5)",
                    uuid.uuid4(), active, uuid.uuid4(), kind, NOW - timedelta(days=age),
                )  # fmt: skip
            for age in (5, 31):
                await conn.execute(
                    "insert into public.served_slates (request_id, slate_id, user_id, page, "
                    "model_version, article_ids, components, created_at) "
                    "values ($1, $2, $3, 0, 'v', '{}', '{}', $4)",
                    uuid.uuid4().hex, uuid.uuid4(), active, NOW - timedelta(days=age),
                )  # fmt: skip
            out = await store.apply_user_retention(conn, NOW, store.UserRetention())
            assert out == {"events_impressions": 1, "events_other": 1, "served_slates": 1,
                           "inactive_anonymous_users": 1}  # fmt: skip
            left = await conn.fetch("select type, created_at from public.events order by 2")
            assert [r["type"] for r in left] == ["click", "impression"]  # 8 d click, 3 d impression
            profiles = {
                r["user_id"] for r in await conn.fetch("select user_id from public.profiles")
            }
            assert profiles == {active, signed_in}  # signed-in readers are kept
            await store.vacuum(conn)  # runs outside a transaction
        finally:
            await conn.close()
