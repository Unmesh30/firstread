"""The slate and its opaque cursor (ADR 0017)."""

import base64
import json
import uuid

import pytest

from firstread_api.errors import ApiError
from firstread_api.slate import MAX_SERVED, Cursor, Slate, SlateItem


def item(k: int) -> SlateItem:
    return SlateItem(f"id-{k}", 1.0 - k / 100, 0.5, 0.1, 0.0, [["new", "New in the last hour"]])


def slate(n: int = 10) -> Slate:
    return Slate(str(uuid.uuid4()), 3, "sig", [item(k) for k in range(n)])


def b64(d: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()


def test_cursor_round_trip_is_opaque() -> None:
    c = Cursor(str(uuid.uuid4()), 20)
    token = c.encode()
    assert Cursor.decode(token) == c
    assert c.slate_id not in token


@pytest.mark.parametrize(
    "token",
    [
        "",
        "not-base64!",
        "x" * 201,
        b64([1, 2]),
        b64({"s": str(uuid.uuid4()), "o": 20}),  # no version
        b64({"s": str(uuid.uuid4()), "o": 20, "v": 2}),  # unknown version
        b64({"s": str(uuid.uuid4()), "o": 20, "v": 1, "x": 1}),  # extra key
        b64({"s": "not-a-uuid", "o": 20, "v": 1}),
        b64({"s": str(uuid.uuid4()), "o": 0, "v": 1}),
        b64({"s": str(uuid.uuid4()), "o": -5, "v": 1}),
        b64({"s": str(uuid.uuid4()), "o": True, "v": 1}),
        b64({"s": str(uuid.uuid4()), "o": "20", "v": 1}),
        b64({"s": str(uuid.uuid4()), "o": 10_001, "v": 1}),
    ],
)
def test_tampered_cursors_are_400(token: str) -> None:
    with pytest.raises(ApiError) as e:
        Cursor.decode(token)
    assert e.value.status == 400
    assert e.value.code == "invalid_cursor"


def test_serving_marks_items_and_caches_the_page_for_60_s() -> None:
    s = slate()
    s.serve(0, s.remaining()[:4], now=1000.0, ttl_s=60)
    assert [i.id for i in s.remaining()][:2] == ["id-4", "id-5"]
    assert s.page == 1
    assert [i.id for i in s.cached_page(0, 1059.0, 60) or []] == ["id-0", "id-1", "id-2", "id-3"]
    assert s.cached_page(0, 1060.0, 60) is None
    assert s.cached_page(4, 1001.0, 60) is None


def test_old_pages_are_pruned_and_served_ids_are_capped() -> None:
    s = slate(4)
    s.serve(0, s.remaining()[:2], now=0.0, ttl_s=60)
    s.serve(2, s.remaining()[:2], now=100.0, ttl_s=60)
    assert set(s.pages) == {"2"}
    s.served = [f"x{k}" for k in range(MAX_SERVED)]
    s.serve(4, [item(99)], now=100.0, ttl_s=60)
    assert len(s.served) == MAX_SERVED
    assert s.served[-1] == "id-99"


def test_a_rerank_keeps_the_items_a_cached_page_names() -> None:
    s = slate(6)
    s.serve(0, s.remaining()[:2], now=0.0, ttl_s=60)
    s.rerank([item(50), item(51)], hist_version=4, hidden_sig="new")
    assert [i.id for i in s.remaining()] == ["id-50", "id-51"]
    assert [i.id for i in s.cached_page(0, 1.0, 60) or []] == ["id-0", "id-1"]
    assert (s.hist_version, s.hidden_sig) == (4, "new")


def test_the_stored_slate_round_trips_compressed() -> None:
    s = slate(100)
    s.serve(0, s.remaining()[:20], now=5.0, ttl_s=60)
    blob = s.dumps()
    assert Slate.loads(blob) == s
    assert len(blob) < 8_000
