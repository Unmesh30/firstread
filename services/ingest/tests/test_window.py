"""The runner-side window snapshot: persistence, the delta, and every reload trigger."""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from firstread_core.db import apply_migrations, connect
from firstread_core.dbtest import MIGRATIONS, skip_or_fail_if_unreachable_async, temp_database
from firstread_ingest import store, window
from firstread_ingest.window import WindowSnapshot

NOW = datetime(2026, 10, 15, 12, 20, tzinfo=UTC)
MODEL = "bge#abc"
WIN, FP = timedelta(hours=48), timedelta(hours=72)


def article(n: int, age: timedelta, text_source: str = "title") -> store.ArticleRow:
    aid = uuid.uuid4()
    v = np.random.default_rng(n).normal(size=384).astype(np.float32)
    v /= np.linalg.norm(v)
    return store.ArticleRow(
        id=aid, url=f"https://e.com/{n}", url_canonical=f"https://e.com/{n}", title=f"t{n}",
        source_name="e", source_domain=f"d{n % 3}.com", image_url=None if n % 2 else "i",
        language="en", source_country="US", geo="us", category="sports",
        category_confidence=0.9, category_source="model", published_at=None,
        first_seen_at=NOW - age, embedding=v, embedding_model="m", text_source=text_source,
        title_fingerprint=f"fp{n}", story_cluster_id=aid, origin="gdelt",
    )  # fmt: skip


async def db(dsn: str) -> Any:
    conn = await connect(dsn)
    await apply_migrations(conn, MIGRATIONS)
    await store.register_codecs(conn)
    return conn


def rows_of(snap: WindowSnapshot) -> dict[uuid.UUID, tuple[Any, ...]]:
    return {
        i: (c, u, f, e.tobytes())
        for i, c, u, f, e in zip(
            snap.ids, snap.clusters, snap.url_canonical, snap.fingerprint, snap.embedding,
            strict=True,
        )
    }  # fmt: skip


pytestmark = pytest.mark.integration


async def test_full_load_matches_the_old_load_recent_exactly() -> None:
    await skip_or_fail_if_unreachable_async()
    async with temp_database() as dsn:
        conn = await db(dsn)
        try:
            ages = [timedelta(hours=h) for h in (1, 30, 47.9, 50, 71, 73, 200)]
            await store.insert_articles(
                conn,
                [article(i, a, "title_summary" if i % 2 else "title") for i, a in enumerate(ages)],
            )
            snap = (await window.sync(conn, None, NOW, MODEL)).snapshot
            new = snap.recent(NOW, WIN, FP)
            old = await store.load_recent(conn, NOW, WIN, FP)
            assert sorted(new.ids) == sorted(old.ids)  # 48 h rows with vectors
            order = [old.ids.index(i) for i in new.ids]
            assert np.array_equal(new.embeddings, old.embeddings[order])  # bit-identical
            assert [old.text_sources[k] for k in order] == new.text_sources
            assert new.url_cluster == old.url_cluster  # 72 h
            assert new.fingerprint_cluster == old.fingerprint_cluster
        finally:
            await conn.close()


async def test_delta_fetches_only_new_rows_and_checksum_holds(tmp_path: Path) -> None:
    await skip_or_fail_if_unreachable_async()
    async with temp_database() as dsn:
        conn = await db(dsn)
        try:
            await store.insert_articles(conn, [article(i, timedelta(hours=i)) for i in range(5)])
            first = await window.sync(conn, None, NOW, MODEL)
            assert first.reloaded
            assert first.reason == "missing"
            assert first.fetched_rows == 5
            path = tmp_path / "w.npz"
            first.snapshot.save(path)
            await store.insert_articles(conn, [article(10 + i, timedelta(0)) for i in range(2)])
            cached, why = window.load_file(path)
            assert why == "loaded"
            assert cached is not None
            second = await window.sync(conn, cached, NOW, MODEL)
            assert not second.reloaded
            assert second.reason == "delta"
            assert second.fetched_rows == 2  # only the new rows cross the wire
            assert len(second.snapshot) == 7
            assert second.snapshot.checksum() == await window.check(conn, NOW - window.WINDOW)
        finally:
            await conn.close()


async def test_expired_rows_leave_the_window_without_a_reload() -> None:
    await skip_or_fail_if_unreachable_async()
    async with temp_database() as dsn:
        conn = await db(dsn)
        try:
            await store.insert_articles(conn, [article(1, timedelta(hours=71.5))])
            snap = (await window.sync(conn, None, NOW, MODEL)).snapshot
            later = await window.sync(conn, snap, NOW + timedelta(hours=1), MODEL)
            assert not later.reloaded
            assert len(later.snapshot) == 0
        finally:
            await conn.close()


@pytest.mark.parametrize("trigger", ["checksum", "schema", "embedding_model"])
async def test_every_mismatch_forces_a_full_reload(trigger: str) -> None:
    await skip_or_fail_if_unreachable_async()
    async with temp_database() as dsn:
        conn = await db(dsn)
        try:
            await store.insert_articles(conn, [article(i, timedelta(hours=i)) for i in range(3)])
            snap = (await window.sync(conn, None, NOW, MODEL)).snapshot
            model = MODEL
            if trigger == "checksum":  # a change the watermark delta can't see
                await conn.execute("delete from public.articles where id = $1", snap.ids[1])
            elif trigger == "schema":
                snap.schema_sha = "old"
            else:
                model = "another-model"
            result = await window.sync(conn, snap, NOW, model)
            assert result.reloaded
            assert result.reason == trigger
            assert result.snapshot.checksum() == await window.check(conn, NOW - window.WINDOW)
        finally:
            await conn.close()


def test_save_load_round_trip(tmp_path: Path) -> None:
    snap = WindowSnapshot.empty("sha", MODEL)
    assert len(snap) == 0
    path = tmp_path / "w.npz"
    snap.save(path)
    loaded, header = WindowSnapshot.load(path)
    assert header["format_version"] == window.FORMAT_VERSION
    assert len(loaded) == 0


@pytest.mark.parametrize("damage", ["missing", "corrupt", "format"])
def test_unusable_files_are_reported_not_trusted(tmp_path: Path, damage: str) -> None:
    path = tmp_path / "w.npz"
    if damage == "corrupt":
        path.write_bytes(b"not an npz")
    elif damage == "format":
        WindowSnapshot.empty("sha", MODEL).save(path)
        data = dict(np.load(path))
        data["header"] = np.frombuffer(b'{"format_version": 999}', np.uint8)
        np.savez(path, **data)
    snap, why = window.load_file(path)
    assert snap is None
    assert why == damage


def test_string_packing_round_trips_unicode() -> None:
    strings = ["", "https://e.com/ü", "日本", "a" * 500]
    assert window._unpack(window._pack(strings)) == strings
