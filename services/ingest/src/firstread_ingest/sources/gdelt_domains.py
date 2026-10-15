"""Where is a news outlet based? GDELT's domain -> country lookup, plus our overrides.

GDELT published a lookup of every outlet it monitors to its most-mentioned country,
2015-2021 (https://blog.gdeltproject.org/?p=14261), as an estimate of where the outlet
is based. It is FIPS 10-4 coded and pinned here by SHA256; it is downloaded once
and cached. Our ``gdelt_domains.yaml`` overrides known misplacements and blocks
auto-generated or syndicated domains.

Credit: data from the GDELT Project, https://www.gdeltproject.org.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from firstread_core.embedding import default_model_cache_dir
from firstread_ingest.config import CONFIG_DIR

LOOKUP_URL = "https://blog.gdeltproject.org/wp-content/uploads/2021-news-outlets-by-countrycode-2015-2021.csv"
LOOKUP_SHA256 = "7cafd03b4465f89d93db74c53c8667f49e96b52b2b0a32893d82aa558846c989"


class DomainLookupError(RuntimeError):
    """The downloaded lookup isn't the pinned file."""


@dataclass
class DomainCountries:
    """domain -> FIPS country, with overrides and a blocklist applied."""

    countries: dict[str, str]
    overrides: dict[str, str]
    blocklist: frozenset[str]
    # Press-release wires: syndicated releases look like broad coverage and would win
    # the breadth ranking and trending, so they are dropped too (counted separately).
    press_wires: frozenset[str] = frozenset()

    def _candidates(self, domain: str) -> list[str]:
        """``a.b.example.co.uk`` -> itself, then each parent down to two labels."""
        labels = domain.lower().strip(".").removeprefix("www.").split(".")
        return [".".join(labels[i:]) for i in range(max(1, len(labels) - 1))]

    def blocked(self, domain: str) -> bool:
        return any(d in self.blocklist for d in self._candidates(domain))

    def press_wire(self, domain: str) -> bool:
        return any(d in self.press_wires for d in self._candidates(domain))

    def fips(self, domain: str) -> str | None:
        """The outlet's FIPS country, or None if neither it nor a parent domain is known."""
        for d in self._candidates(domain):
            if d in self.overrides:
                return self.overrides[d]
            if d in self.countries:
                return self.countries[d]
        return None


def parse_lookup(text: str) -> dict[str, str]:
    """``domain,countrycode,cnt`` rows (one per mentioned country) -> top country per domain."""
    best: dict[str, tuple[str, int]] = {}
    reader = csv.reader(io.StringIO(text))
    next(reader, None)
    for row in reader:
        if len(row) < 3 or not row[2].strip().isdigit():
            continue
        domain, code, count = row[0].strip().lower(), row[1].strip().upper(), int(row[2])
        if domain and code and count > best.get(domain, ("", -1))[1]:
            best[domain] = (code, count)
    return {d: c for d, (c, _) in best.items()}


def load_config(
    path: Path = CONFIG_DIR / "gdelt_domains.yaml",
) -> tuple[dict[str, str], frozenset[str]]:
    data: dict[str, Any] = yaml.safe_load(path.read_text())
    if data.get("version") != 1:
        raise ValueError(f"{path}: expected version: 1")
    overrides = {k.lower(): str(v).upper() for k, v in (data.get("overrides") or {}).items()}
    return overrides, frozenset(d.lower() for d in data.get("blocklist") or [])


def load_press_wires(path: Path = CONFIG_DIR / "gdelt_domains.yaml") -> frozenset[str]:
    data: dict[str, Any] = yaml.safe_load(path.read_text())
    return frozenset(d.lower() for d in data.get("press_wires") or [])


def lookup_path(cache_dir: Path | None = None) -> Path:
    return (cache_dir or default_model_cache_dir()) / "gdelt" / f"outlets-{LOOKUP_SHA256[:12]}.csv"


async def ensure_lookup(client: Any, cache_dir: Path | None = None) -> Path:
    """Download the pinned lookup once; verify its SHA256 every time it is used."""
    path = lookup_path(cache_dir)
    if not path.is_file():
        resp = await client.get(LOOKUP_URL)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(resp.content)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != LOOKUP_SHA256:
        path.unlink(missing_ok=True)
        raise DomainLookupError(f"GDELT domain lookup SHA256 {digest} != pinned {LOOKUP_SHA256}")
    return path


def load_domain_countries(lookup_file: Path, config: Path | None = None) -> DomainCountries:
    overrides, blocklist = load_config(config) if config else load_config()
    wires = load_press_wires(config) if config else load_press_wires()
    text = lookup_file.read_text(encoding="utf-8", errors="replace")
    return DomainCountries(parse_lookup(text), overrides, blocklist, wires)
