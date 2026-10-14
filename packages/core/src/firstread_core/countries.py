"""Country codes. Postgres stores ISO 3166-1 alpha-2; GDELT speaks FIPS 10-4.

The two systems disagree on several codes we use (FIPS UK = ISO GB, AS = AU,
EI = IE), and some FIPS codes mean a different country in ISO (FIPS AS is
Australia, ISO AS is American Samoa). Every GDELT country goes through
:func:`fips_to_iso` before it is stored.
"""

# FIPS 10-4 -> ISO 3166-1 alpha-2 for the countries GDELT queries and lookups
# can return for our English-language scope, plus common ones seen in probes.
FIPS_TO_ISO: dict[str, str] = {
    "US": "US", "UK": "GB", "CA": "CA", "AS": "AU", "EI": "IE", "NZ": "NZ",
    "IN": "IN", "NI": "NG", "SF": "ZA", "KE": "KE", "IS": "IL", "SN": "SG",
    "RP": "PH", "PK": "PK", "MY": "MY", "JA": "JP", "CH": "CN", "HK": "HK",
    "GM": "DE", "FR": "FR", "SP": "ES", "IT": "IT", "MX": "MX", "BR": "BR",
    "AR": "AR", "UP": "UA", "RS": "RU", "QA": "QA", "AE": "AE", "SA": "SA",
    "BF": "BS", "GH": "GH", "UG": "UG", "TZ": "TZ", "ZI": "ZW", "BG": "BD",
    "CE": "LK", "NP": "NP", "TW": "TW", "KS": "KR", "TU": "TR", "EG": "EG",
}  # fmt: skip
US = "US"


def fips_to_iso(fips: str) -> str | None:
    """ISO alpha-2 for a FIPS 10-4 code, or None if we don't know it."""
    return FIPS_TO_ISO.get(fips.strip().upper())
