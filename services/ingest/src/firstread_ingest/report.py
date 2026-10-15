"""Render a run summary as Markdown for the GitHub Actions job summary."""

from __future__ import annotations

from typing import Any

GDELT_CREDIT = "Contains data from [the GDELT Project](https://www.gdeltproject.org)."


def _table(rows: list[tuple[str, Any]], head: tuple[str, str]) -> list[str]:
    out = [f"| {head[0]} | {head[1]} |", "|---|---:|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return out


def markdown(summary: dict[str, Any], dry_run: bool) -> str:
    d = summary["details"]
    status = summary["status"]
    icon = {"ok": "✅", "partial": "⚠️", "failed": "❌"}.get(status, "")
    lines = [
        f"## FirstRead ingestion {icon} `{status}`"
        + (" (dry run, nothing written)" if dry_run else ""),
        "",
        *_table(
            [
                ("fetched", summary["fetched"]),
                ("inserted", summary["inserted"]),
                ("duplicates", summary["duplicates"]),
                ("near-duplicates", summary["near_duplicates"]),
                ("joined an existing story", summary["clustered"]),
                ("errors", summary["errors"]),
                ("GDELT requests", summary["gdelt_calls"]),
                ("Redis commands", summary.get("redis_commands", 0)),
            ],
            ("run", "count"),
        ),
    ]
    if d.get("dropped"):
        lines += ["", "### Dropped before dedupe", ""]
        lines += _table(sorted(d["dropped"].items(), key=lambda kv: -kv[1]), ("reason", "items"))
    g = d.get("gdelt") or {}
    if g:
        files = g.get("files", [])
        lines += [
            "",
            "### GDELT GKG",
            "",
            f"- Files: {len(files)} "
            f"({sum(1 for f in files if f.get('pending'))} pending, "
            f"{sum(1 for f in files if f.get('missing'))} missing); "
            f"newest available `{g.get('newest_available')}`; cursor `{g.get('cursor')}`.",
            f"- Records {g.get('records', 0)}; candidates {g.get('candidates', 0)}; "
            f"selected {g.get('selected', 0)}; capped {g.get('capped', {})}.",
            f"- Dropped at source: {g.get('dropped', {})}.",
        ]
        if g.get("stale"):
            lines.append("- ⚠️ **Stale:** the newest GKG file is over an hour old.")
        if g.get("top_domains"):
            lines += [
                "",
                "<details><summary>Top 20 GDELT outlets this run "
                "(after the blocklist and country filter)</summary>",
                "",
            ]
            lines += _table(list(g["top_domains"].items()), ("outlet", "records"))
            lines += ["", "</details>"]
    if d.get("topics"):
        lines += ["", "### Inserted by topic", ""]
        lines += _table(sorted(d["topics"].items(), key=lambda kv: -kv[1]), ("topic", "articles"))
    if d.get("pools"):
        lines += ["", "### Pools written", ""]
        lines += _table(sorted(d["pools"].items()), ("pool", "members"))
    if d.get("retention"):
        lines += ["", f"Retention removed: {d['retention']}."]
    if d.get("size"):
        mb = d["size"].get("db", 0) / 1e6
        warn = " ⚠️ over the warning threshold" if d.get("size_warning") else ""
        lines += [f"Database size: {mb:.1f} MB{warn}."]
    if d.get("feed_errors"):
        lines += ["", "### Errors", ""]
        lines += [f"- `{k}`: {v}" for k, v in sorted(d["feed_errors"].items())]
    lines += ["", GDELT_CREDIT, ""]
    return "\n".join(lines)
