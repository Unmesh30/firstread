import pytest

from firstread_core import __version__, normalize_whitespace


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Breaking   news\n\ttoday ", "Breaking news today"),
        ("", ""),
        ("already clean", "already clean"),
    ],
)
def test_normalize_whitespace(raw: str, expected: str) -> None:
    assert normalize_whitespace(raw) == expected


def test_version_is_set() -> None:
    assert __version__
