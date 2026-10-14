"""Record sanitized RSS/Atom fixtures for the adapter tests (manual, never in CI).

Fetches a few real feeds of different shapes, keeps at most 5 items, and replaces
every title, description/summary/content and link with synthetic text, so the
fixtures keep the real *structure* (namespaces, media tags, dates, CDATA) but no
publisher text. Works on the raw text, so prefixes and namespaces stay untouched.

    uv run python services/ingest/scripts/record_fixtures.py
"""

import re
from pathlib import Path

import httpx

UA = "FirstRead/0.1 (+https://github.com/Unmesh30/firstread)"
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rss"
FEEDS = {
    "rss2_media_thumbnail.xml": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "rss2_media_content.xml": "https://www.theguardian.com/world/rss",
    "rss2_enclosure_cdata.xml": "https://www.pbs.org/newshour/feeds/rss/headlines",
    "atom.xml": "https://www.theverge.com/rss/index.xml",
    "rdf.xml": "https://rss.dw.com/rdf/rss-en-all",
}
MAX_ITEMS = 5
TEXT_TAGS = ("title", "description", "summary", "content:encoded", "content", "subtitle",
             "media:title", "media:description", "media:credit", "dc:creator", "name",
             "copyright", "rights", "dc:rights")  # fmt: skip


def keep_first_items(xml: str) -> str:
    for tag in ("item", "entry"):
        ends = [m.end() for m in re.finditer(rf"</{tag}>", xml)]
        if len(ends) > MAX_ITEMS:
            cut, close = ends[MAX_ITEMS - 1], xml.rfind(f"</{tag}>")
            xml = xml[:cut] + xml[close + len(f"</{tag}>") :]
    return xml


def sanitize(xml: str, name: str) -> str:
    counter = iter(range(1, 10_000))

    def text(m: re.Match[str]) -> str:
        tag, attrs = m.group(1), m.group(2) or ""
        n = next(counter)
        words = f"Synthetic {tag.split(':')[-1]} {n} for fixture {name}"
        return f"<{tag}{attrs}>{words}</{tag}>"

    for tag in TEXT_TAGS:
        xml = re.sub(rf"<({re.escape(tag)})(\s[^>]*)?>.*?</{re.escape(tag)}>", text, xml,
                     flags=re.S)  # fmt: skip
    # Links and guids point at example.com, keeping the path shape. Namespace
    # declarations must keep their real URIs, or parsers stop recognizing the tags.
    namespaces: list[str] = []

    def protect(m: re.Match[str]) -> str:
        namespaces.append(m.group(0))
        return f"@@NS{len(namespaces) - 1}@@"

    xml = re.sub(r"xmlns(:\w+)?=\"[^\"]*\"", protect, xml)
    xml = re.sub(r"https?://[^\s\"'<>\]]+", lambda m: _example(m.group(0)), xml)
    for i, ns in enumerate(namespaces):
        xml = xml.replace(f"@@NS{i}@@", ns)
    return xml


def _example(url: str) -> str:
    path = re.sub(r"^https?://[^/]+", "", url)
    if re.search(r"\.(jpe?g|png|webp|gif)(\?|$)", url, re.I):
        return "https://img.example.com/image.jpg"
    return "https://example.com" + (path[:60] or "/")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": UA}, timeout=30, follow_redirects=True) as c:
        for name, url in FEEDS.items():
            body = c.get(url).text
            (OUT / name).write_text(sanitize(keep_first_items(body), name))
            print(f"wrote {name}")


if __name__ == "__main__":
    main()
