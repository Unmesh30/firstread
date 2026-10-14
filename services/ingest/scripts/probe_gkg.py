"""Manual probe of the GDELT 2.0 GKG 15-minute files (never run in CI).

Reads ``lastupdate.txt``, downloads the single ``*.gkg.csv.zip`` it lists
(checking size and MD5), and reports what an ingestion adapter could use:
records per file, URL / page-title / sharing-image coverage, translation info,
title lengths, and how much survives a source-country filter by top-level
domain. The raw file stays in ``data/probes/gkg/`` (gitignored); the summary
JSON holds aggregates only.

    uv run python services/ingest/scripts/probe_gkg.py
"""

import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

LASTUPDATE = "https://data.gdeltproject.org/gdeltv2/lastupdate.txt"
UA = "FirstRead/0.1 (+https://github.com/Unmesh30/firstread)"
OUT = Path(__file__).resolve().parents[3] / "data" / "probes" / "gkg"

# GKG 2.1 columns (http://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf)
COLS = [
    "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName",
    "DocumentIdentifier", "Counts", "V2Counts", "Themes", "V2Themes", "Locations",
    "V2Locations", "Persons", "V2Persons", "Organizations", "V2Organizations", "V2Tone",
    "Dates", "GCAM", "SharingImage", "RelatedImages", "SocialImageEmbeds",
    "SocialVideoEmbeds", "Quotations", "AllNames", "Amounts", "TranslationInfo", "Extras",
]  # fmt: skip
TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)
# Our GDELT scope (ISO 3166-1 alpha-2) by country-code TLD. Generic TLDs (.com,
# .org, .net, ...) say nothing about the publisher's country.
CCTLD = {
    "us": "US", "uk": "GB", "ca": "CA", "au": "AU", "ie": "IE", "nz": "NZ",
    "gov": "US", "edu": "US", "mil": "US",
}  # fmt: skip
GENERIC = {"com", "org", "net", "news", "info", "co", "io", "tv", "biz", "media", "online"}


def fetch(client: httpx.Client, url: str) -> bytes:
    r = client.get(url.replace("http://", "https://", 1), follow_redirects=True)
    r.raise_for_status()
    return r.content


def tld(domain: str) -> str:
    return domain.rsplit(".", 1)[-1].lower() if "." in domain else ""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(sys.maxsize)
    with httpx.Client(headers={"User-Agent": UA}, timeout=httpx.Timeout(120.0)) as client:
        lines = fetch(client, LASTUPDATE).decode().split("\n")
        size, md5, url = next(ln.split() for ln in lines if ln.endswith(".gkg.csv.zip"))
        blob = fetch(client, url)
    md5_ok = hashlib.md5(blob).hexdigest() == md5  # noqa: S324 (GDELT publishes MD5)
    (OUT / url.rsplit("/", 1)[-1]).write_bytes(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        text = z.read(z.namelist()[0]).decode("utf-8", errors="replace")

    rows = [r for r in csv.reader(io.StringIO(text), delimiter="\t") if len(r) == len(COLS)]
    recs = [dict(zip(COLS, r, strict=True)) for r in rows]
    titles = [m.group(1).strip() if (m := TITLE.search(r["Extras"])) else "" for r in recs]
    words = [len(t.split()) for t in titles if t]
    domains = [r["SourceCommonName"].lower() for r in recs]
    countries = Counter(
        CCTLD.get(tld(d), "generic" if tld(d) in GENERIC else "other") for d in domains
    )
    hosts = [urlsplit(r["DocumentIdentifier"]).hostname or "" for r in recs]
    n = len(recs)

    def share(k: int) -> float:
        return round(k / n, 4) if n else 0.0

    summary: dict[str, Any] = {
        "file": url.rsplit("/", 1)[-1],
        "zip_bytes": len(blob),
        "listed_bytes": int(size),
        "md5_ok": md5_ok,
        "records": n,
        "bad_rows": len(text.splitlines()) - n,
        "source_collection": dict(Counter(r["SourceCollectionIdentifier"] for r in recs)),
        "url_coverage": share(sum(r["DocumentIdentifier"].startswith("http") for r in recs)),
        "page_title_coverage": share(sum(bool(t) for t in titles)),
        "sharing_image_coverage": share(sum(bool(r["SharingImage"]) for r in recs)),
        "translation_info_present": share(sum(bool(r["TranslationInfo"]) for r in recs)),
        "title_words": {
            "p10": sorted(words)[len(words) // 10] if words else None,
            "median": sorted(words)[len(words) // 2] if words else None,
            "share_ge_3": round(sum(w >= 3 for w in words) / len(words), 4) if words else None,
        },
        "distinct_domains": len(set(domains)),
        "url_host_matches_domain": share(
            sum(d and d in h for d, h in zip(domains, hosts, strict=True))
        ),
        "tld_country": {k: share(v) for k, v in countries.most_common()},
        "in_scope_by_cctld": share(
            sum(v for k, v in countries.items() if k not in ("generic", "other"))
        ),
        "top_tlds": dict(Counter(tld(d) for d in domains).most_common(15)),
        "extras_tags": dict(
            Counter(t for r in recs for t in set(re.findall(r"<(PAGE_[A-Z_]+)>", r["Extras"])))
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
