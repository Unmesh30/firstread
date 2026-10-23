import pytest

from firstread_api.errors import ApiError
from firstread_api.ratelimit import RateLimiter


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_bursts_up_to_the_limit_then_refills() -> None:
    clock = Clock()
    rl = RateLimiter(60, max_keys=10, clock=clock)  # 1 token per second
    assert all(rl.take("a") == 0 for _ in range(60))
    assert rl.take("a") == pytest.approx(1.0)
    clock.t = 2.0
    assert rl.take("a") == 0
    assert rl.take("b") == 0  # keys are independent


def test_429_has_retry_after() -> None:
    rl = RateLimiter(1, max_keys=10, clock=Clock())
    rl.check("a", "events")
    with pytest.raises(ApiError) as e:
        rl.check("a", "events")
    assert e.value.status == 429
    assert e.value.headers["Retry-After"] == "60"


def test_least_recently_used_keys_are_forgotten() -> None:
    rl = RateLimiter(1, max_keys=2, clock=Clock())
    for k in ("a", "b", "c"):
        rl.take(k)
    assert set(rl.buckets) == {"b", "c"}
