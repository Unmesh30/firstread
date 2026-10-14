import pytest

from firstread_core.countries import fips_to_iso


@pytest.mark.parametrize(
    ("fips", "iso"),
    [("US", "US"), ("UK", "GB"), ("CA", "CA"), ("AS", "AU"), ("EI", "IE"), ("NZ", "NZ")],
)
def test_every_gdelt_country_we_query_maps_to_iso(fips: str, iso: str) -> None:
    assert fips_to_iso(fips) == iso


def test_fips_codes_that_mean_something_else_in_iso() -> None:
    assert fips_to_iso("AS") != "AS"  # ISO AS is American Samoa
    assert fips_to_iso("IS") == "IL"  # FIPS IS is Israel; ISO IS is Iceland
    assert fips_to_iso(" uk ") == "GB"
    assert fips_to_iso("ZZ") is None
