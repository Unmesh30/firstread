"""Country codes. Postgres stores ISO 3166-1 alpha-2; GDELT speaks FIPS 10-4.

The two systems disagree on several codes we use (FIPS UK = ISO GB, AS = AU,
EI = IE), and some FIPS codes mean a different country in ISO (FIPS AS is
Australia, ISO AS is American Samoa). Every GDELT country goes through
:func:`fips_to_iso` before it is stored.
"""

# FIPS 10-4 -> ISO 3166-1 alpha-2. GDELT's domain lookup can place an outlet in any of
# these; an unknown code stores NULL (the geo rule only needs "US or not").
FIPS_TO_ISO: dict[str, str] = {
    "AA": "AW", "AE": "AE", "AF": "AF", "AG": "DZ", "AJ": "AZ", "AL": "AL", "AM": "AM", "AN": "AD",
    "AO": "AO", "AR": "AR", "AS": "AU", "AU": "AT", "BA": "BH", "BB": "BB", "BC": "BW", "BD": "BM",
    "BE": "BE", "BF": "BS", "BG": "BD", "BH": "BZ", "BK": "BA", "BL": "BO", "BM": "MM", "BN": "BJ",
    "BO": "BY", "BP": "SB", "BR": "BR", "BT": "BT", "BU": "BG", "BX": "BN", "BY": "BI", "CA": "CA",
    "CB": "KH", "CD": "TD", "CE": "LK", "CF": "CG", "CG": "CD", "CH": "CN", "CI": "CL", "CJ": "KY",
    "CM": "CM", "CN": "KM", "CO": "CO", "CS": "CR", "CT": "CF", "CU": "CU", "CV": "CV", "CW": "CK",
    "CY": "CY", "DA": "DK", "DJ": "DJ", "DR": "DO", "EC": "EC", "EG": "EG", "EI": "IE", "EK": "GQ",
    "EN": "EE", "ER": "ER", "ES": "SV", "ET": "ET", "EZ": "CZ", "FI": "FI", "FJ": "FJ", "FM": "FM",
    "FO": "FO", "FP": "PF", "FR": "FR", "GA": "GM", "GB": "GA", "GG": "GE", "GH": "GH", "GI": "GI",
    "GK": "GG", "GL": "GL", "GM": "DE", "GQ": "GU", "GR": "GR", "GT": "GT", "GV": "GN", "GY": "GY",
    "HA": "HT", "HK": "HK", "HO": "HN", "HR": "HR", "HU": "HU", "IC": "IS", "ID": "ID", "IM": "IM",
    "IN": "IN", "IR": "IR", "IS": "IL", "IT": "IT", "IV": "CI", "IZ": "IQ", "JA": "JP", "JE": "JE",
    "JM": "JM", "JO": "JO", "KE": "KE", "KG": "KG", "KN": "KP", "KR": "KI", "KS": "KR", "KU": "KW",
    "KZ": "KZ", "LA": "LA", "LE": "LB", "LG": "LV", "LH": "LT", "LI": "LR", "LO": "SK", "LS": "LI",
    "LT": "LS", "LU": "LU", "LY": "LY", "MA": "MG", "MC": "MO", "MD": "MD", "MG": "MN", "MI": "MW",
    "MJ": "ME", "MK": "MK", "ML": "ML", "MN": "MC", "MO": "MA", "MP": "MU", "MR": "MR", "MT": "MT",
    "MU": "OM", "MV": "MV", "MX": "MX", "MY": "MY", "MZ": "MZ", "NC": "NC", "NE": "NU", "NG": "NE",
    "NH": "VU", "NI": "NG", "NL": "NL", "NO": "NO", "NP": "NP", "NR": "NR", "NS": "SR", "NU": "NI",
    "NZ": "NZ", "OD": "SS", "PA": "PY", "PE": "PE", "PK": "PK", "PL": "PL", "PM": "PA", "PO": "PT",
    "PP": "PG", "PS": "PW", "PU": "GW", "QA": "QA", "RI": "RS", "RM": "MH", "RO": "RO", "RP": "PH",
    "RQ": "PR", "RS": "RU", "RW": "RW", "SA": "SA", "SE": "SC", "SF": "ZA", "SG": "SN", "SI": "SI",
    "SL": "SL", "SM": "SM", "SN": "SG", "SO": "SO", "SP": "ES", "SU": "SD", "SW": "SE", "SY": "SY",
    "SZ": "CH", "TD": "TT", "TH": "TH", "TI": "TJ", "TL": "TK", "TN": "TO", "TO": "TG", "TP": "ST",
    "TS": "TN", "TU": "TR", "TV": "TV", "TW": "TW", "TX": "TM", "TZ": "TZ", "UG": "UG", "UK": "GB",
    "UP": "UA", "US": "US", "UV": "BF", "UY": "UY", "UZ": "UZ", "VE": "VE", "VI": "VG", "VM": "VN",
    "VQ": "VI", "VT": "VA", "WA": "NA", "WF": "WF", "WS": "WS", "WZ": "SZ", "YM": "YE", "ZA": "ZM",
    "ZI": "ZW",
}  # fmt: skip
US = "US"


def fips_to_iso(fips: str) -> str | None:
    """ISO alpha-2 for a FIPS 10-4 code, or None if we don't know it."""
    return FIPS_TO_ISO.get(fips.strip().upper())
