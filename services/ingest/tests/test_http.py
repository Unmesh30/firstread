import random

import httpx
import pytest
import respx

from firstread_ingest.http import USER_AGENT, fetch_with_retry, make_client

URL = "https://feeds.example.com/rss"


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311 (deterministic jitter in tests)


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, s: float) -> None:
        self.calls.append(s)


async def test_retries_429_honouring_retry_after(no_network: respx.MockRouter) -> None:
    no_network.get(URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, text="ok"),
        ]
    )
    sleeps = Sleeps()
    async with make_client() as c:
        r = await fetch_with_retry(c, URL, sleep=sleeps, rng=seeded(0))
    assert r.status_code == 200
    assert sleeps.calls == [7.0]


async def test_backoff_uses_full_jitter_within_exponential_bounds(
    no_network: respx.MockRouter,
) -> None:
    no_network.get(URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(502), httpx.Response(200)]
    )
    sleeps = Sleeps()
    async with make_client() as c:
        r = await fetch_with_retry(c, URL, sleep=sleeps, rng=seeded(1), base_delay=2.0)
    assert r.status_code == 200
    assert 0 <= sleeps.calls[0] <= 2.0
    assert 0 <= sleeps.calls[1] <= 4.0


async def test_timeouts_are_retried_then_raised(no_network: respx.MockRouter) -> None:
    no_network.get(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    sleeps = Sleeps()
    async with make_client() as c:
        with pytest.raises(httpx.ReadTimeout):
            await fetch_with_retry(c, URL, sleep=sleeps, attempts=3)
    assert len(sleeps.calls) == 2


async def test_last_bad_response_is_returned_and_4xx_is_not_retried(
    no_network: respx.MockRouter,
) -> None:
    route = no_network.get(URL).mock(return_value=httpx.Response(500))
    async with make_client() as c:
        r = await fetch_with_retry(c, URL, sleep=Sleeps(), attempts=2)
    assert r.status_code == 500
    assert route.call_count == 2
    route404 = no_network.get(URL + "/missing").mock(return_value=httpx.Response(404))
    async with make_client() as c:
        assert (await fetch_with_retry(c, URL + "/missing", sleep=Sleeps())).status_code == 404
    assert route404.call_count == 1


async def test_user_agent_names_the_repo(no_network: respx.MockRouter) -> None:
    route = no_network.get(URL).mock(return_value=httpx.Response(200))
    async with make_client() as c:
        await fetch_with_retry(c, URL)
    assert route.calls.last.request.headers["user-agent"] == USER_AGENT
    assert "github.com/Unmesh30/firstread" in USER_AGENT
