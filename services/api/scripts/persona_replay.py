"""Persona replay report (P3e): the real feed on three replayed days of live news.

    uv run python services/api/scripts/persona_replay.py run      # ~25 min, local stack
    uv run python services/api/scripts/persona_replay.py report   # aggregates + labels

``run`` replays the cached GKG files (``make replay``'s window, 2026-10-12 02:30 →
10-15 02:15) into ``firstread_replay`` / Redis db 1 with ``--fresh``, through the
unchanged ingestion pipeline (which now refreshes the demo personas). After each run
whose slot is on a 3-hour mark (8 clock times a day) the database and the published
pool snapshot are exactly what production would have had then, and the real API app
(in-process, its clock set to the replay time) serves:

* **each persona** as a fresh activated reader: page 1 under rank-2 and under rank-3
  (the default), with the ranking pass traced (story groups, MMR's picks,
  story-cap drops);
* **the new-reader check** (ADR 0014 amendments): a reader follows 3 topics, reads
  page 1 (cold-segment weights), clicks the top trending story outside them, then
  reads page 2, under rank-1 (k = 0), rank-2 and rank-3 (k = 3).

Examples with headlines go to ``data/persona-replay/`` (gitignored). ``report``
writes aggregates only to ``docs/learn/data/03-personas.json``. The labelled
story-cap samples come from the kept rank-2 run (``data/persona-replay/rank2/``:
its ``report.json``, ``samples.json`` and ``labels.json``); grouping doesn't depend
on the blend weights, so the samples weren't re-drawn for rank-3.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import tempfile
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from pydantic import SecretStr

from firstread_api.deps import Services
from firstread_api.feed import FeedPipeline
from firstread_api.main import create_app
from firstread_api.testing import Signer, make_settings
from firstread_core import stories
from firstread_core.dbtest import with_database
from firstread_core.personas import parse_all
from firstread_core.ranking import RankingConfig, load_config, mmr
from firstread_ingest import replay
from firstread_ingest.config import Settings as IngestSettings
from firstread_ingest.sources.gkg import parse_slot

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "persona-replay"
RAW = OUT / "report.json"
LABELLED = OUT / "rank2"  # the rank-2 run whose story-cap samples were labelled
SAMPLES = LABELLED / "samples.json"
LABELS = LABELLED / "labels.json"
AGGREGATES = REPO / "docs" / "learn" / "data" / "03-personas.json"
# The labelled run (title-title tau_feed 0.875) whose sample triggered ADR 0013's
# fallback; its aggregates are kept next to the re-run under the shipped thresholds.
REVIEW_RUN = OUT / "tau0875" / "03-personas.json"
END, DAYS, DB, REDIS_DB = "20261015021500", 3, "firstread_replay", 1
EVERY_HOURS = 3  # 8 clock times a day
STEADY_AFTER = timedelta(hours=24)  # clock times with at least a day of news before them
NEW_READER_SETS = (
    ("sports", "health", "entertainment"),
    ("business", "tech_science", "politics"),
    ("world", "travel", "food_drink"),
)
PAGE = 20
SAMPLES_PER_PERSONA = 10
PERSONA_RANKINGS = ("rank-2", "rank-3")
NEW_READER_RANKINGS = ("rank-1", "rank-2", "rank-3")


class Probe:
    def __init__(self, svc: Services, client: httpx.AsyncClient, signer: Signer) -> None:
        self.svc = svc
        self.client = client
        self.signer = signer
        self.base = svc.ranking
        self.out: dict[str, Any] = {"times": []}
        self.first_slot: datetime | None = None

    def auth(self, uid: uuid.UUID) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.signer.token(uid)}"}

    async def call(self, method: str, url: str, uid: uuid.UUID, **kw: Any) -> dict[str, Any]:
        r = await self.client.request(method, url, headers=self.auth(uid), **kw)
        if r.status_code != 200:
            raise RuntimeError(f"{method} {url}: {r.status_code} {r.text[:200]}")
        body: dict[str, Any] = r.json()
        return body

    def title(self, aid: uuid.UUID) -> dict[str, Any]:
        a = self.svc.articles.rows[aid]
        return {"id": str(aid), "title": a.title, "domain": a.source_domain,
                "cluster": str(a.story_cluster_id), "category": a.category}  # fmt: skip

    async def persona_page(self, persona: str, now: datetime, cfg: RankingConfig) -> dict[str, Any]:
        """A fresh reader activates ``persona`` and gets page 1 through the pipeline."""
        self.svc.ranking = cfg
        uid = uuid.uuid4()
        await self.call("POST", f"/v1/demo/personas/{persona}/activate", uid)
        pipe = FeedPipeline(self.svc, uid, f"replay-{uuid.uuid4().hex[:12]}")
        items, _ = await pipe.run(None, PAGE)
        ages = [(now - a.first_seen_at).total_seconds() / 3600 for _, a in items]
        out: dict[str, Any] = {"ages_h": [round(x, 3) for x in ages], "degraded": pipe.degraded,
                               "categories": [a.category for _, a in items]}  # fmt: skip
        tr = pipe.trace
        if tr is None:
            return out
        res, groups = tr.mmr, tr.groups
        lam, k_slate = cfg.mmr_lambda, cfg.slate_size
        no_cap_page = set(mmr(tr.final, tr.vectors, lam, None, k=PAGE).order)
        no_cap_slate = set(mmr(tr.final, tr.vectors, lam, None, k=k_slate).order)
        sizes = Counter(groups)
        drops = []
        for i in res.capped:
            if i not in no_cap_slate:
                continue
            kept = [j for j in res.order if groups[j] == groups[i]]
            pairs = []
            for j in kept:
                cos = float(np.dot(tr.vectors[i], tr.vectors[j]))
                pair = stories.pair_type(tr.text_sources[i], tr.text_sources[j])  # type: ignore[arg-type]
                pairs.append({**self.title(tr.ids[j]), "cos": round(cos, 4),
                              "pair_type": "-".join(pair)})  # fmt: skip
            same_cluster = any(p["cluster"] == str(self.svc.articles.rows[tr.ids[i]]
                                                   .story_cluster_id) for p in pairs)  # fmt: skip
            drops.append({
                "dropped": self.title(tr.ids[i]), "kept": pairs,
                "on_page_1": i in no_cap_page,
                "via": "cluster" if same_cluster else "tau_feed",
            })  # fmt: skip
        out |= {
            "candidates": len(tr.ids),
            "largest_group": max(sizes.values()) if sizes else 0,
            "groups_over_cap": sum(1 for n in sizes.values() if n > cfg.story_cap),
            "cap_drops_page_1": sum(1 for d in drops if d["on_page_1"]),
            "cap_drops_slate": len(drops),
            "drops": drops,
        }
        return out

    async def new_reader(self, topics: tuple[str, ...], cfg: RankingConfig) -> dict[str, Any]:
        """Follow 3 topics, read page 1, click the top trending story outside them, read
        page 2: where does page 2 come from?"""
        self.svc.ranking = cfg
        uid = uuid.uuid4()
        await self.call("PUT", "/v1/me/topics", uid, json={"topics": list(topics)})
        first = await self.call("GET", "/v1/feed", uid, params={"limit": PAGE})
        shown = {i["id"] for i in first["items"]}
        page1_ages = [
            round((self.svc.clock() - datetime.fromisoformat(i["first_seen_at"]))
                  .total_seconds() / 3600, 3)
            for i in first["items"]
        ]  # fmt: skip
        snap = await self.svc.pools.get()
        if snap is None:
            return {"skipped": "no pool snapshot"}
        click = next(
            (t.id for t in snap.trending
             if (info := snap.articles.get(t.id)) is not None
             and info.category not in topics and str(t.id) not in shown),
            None,
        )  # fmt: skip
        if click is None:
            return {"skipped": "no off-topic trending story"}
        clicked_topic = snap.articles[click].category
        event = {"event_id": str(uuid.uuid4()), "type": "click", "article_id": str(click)}
        await self.call("POST", "/v1/events", uid, json={"events": [event]})
        second = await self.call(
            "GET", "/v1/feed", uid, params={"limit": PAGE, "cursor": first["next_cursor"]}
        )
        cats = [i["category"] for i in second["items"]]
        return {
            "page1_ages_h": page1_ages,
            "clicked_topic": clicked_topic,
            "clicked": str(click),
            "page2_categories": cats,
            "page2_followed": sum(1 for c in cats if c in topics),
            "page2_clicked_topic": sum(1 for c in cats if c == clicked_topic),
            "page2_items": len(cats),
            "degraded": sorted(set(first["degraded"]) | set(second["degraded"])),
        }

    async def after_run(self, slot: datetime, now: datetime) -> None:
        self.first_slot = self.first_slot or slot
        if slot.minute != 0 or slot.hour % EVERY_HOURS:
            return
        self.svc.clock = lambda: now
        self.svc.pools.checked_at = -1e18  # read the snapshot this run just published
        snap = await self.svc.pools.get()
        if snap is None:
            return
        entry: dict[str, Any] = {
            "now": now.isoformat(), "steady": now - self.first_slot >= STEADY_AFTER,
            "personas": {v: {} for v in PERSONA_RANKINGS}, "new_reader": [],
        }  # fmt: skip
        try:
            for p in parse_all(snap.personas):
                for v in PERSONA_RANKINGS:
                    entry["personas"][v][p.id] = await self.persona_page(p.id, now, load_config(v))
            for topics in NEW_READER_SETS:
                for cfg in (load_config(v) for v in NEW_READER_RANKINGS):
                    got = await self.new_reader(topics, cfg)
                    meta = {"topics": list(topics), "ranking": cfg.version,
                            "cold_start_k": cfg.cold_start_k}  # fmt: skip
                    entry["new_reader"].append(meta | got)
        finally:
            self.svc.ranking = self.base
        self.out["times"].append(entry)
        print(f"  persona probe at {now:%m-%d %H:%M}: {len(snap.personas)} personas", flush=True)


async def run() -> dict[str, Any]:
    ingest = IngestSettings()
    dsn = with_database(ingest.database_url.get_secret_value(), DB)
    base_redis = ingest.redis_url.get_secret_value()
    signer = Signer()
    with tempfile.TemporaryDirectory() as tmp:
        settings = make_settings(
            database_url=SecretStr(dsn),
            redis_url=SecretStr(base_redis.rsplit("/", 1)[0] + f"/{REDIS_DB}"),
            auth_local_jwks_file=signer.write_jwks(Path(tmp) / "jwks.json"),
            api_load_model=True,
            # Ranking quality, not latency: generous deadlines (degradations are counted).
            api_feed_timeout_s=10.0, api_state_timeout_s=2.0, api_encode_timeout_s=2.0,
            api_candidates_timeout_s=2.0, api_hydrate_timeout_s=2.0, api_persist_timeout_s=2.0,
            rl_ip_per_min=1_000_000, rl_user_writes_per_min=1_000,
        )  # fmt: skip
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://replay") as client:
                probe = Probe(app.state.services, client, signer)
                result = await replay.main(
                    DAYS, parse_slot(END), DB, REDIS_DB, fresh=True,
                    after_run=probe.after_run, write_reports=False,
                )  # fmt: skip
    # Per-run ingestion summaries (counts only, no headlines) next to the probes.
    return probe.out | {"runs": result.runs}


# ---- report ----


def mean(pages: list[dict[str, Any]], key: str) -> float:
    return round(float(statistics.mean(float(x[key]) for x in pages)), 2)


def page2_share(rows: list[dict[str, Any]], key: str, topics: tuple[str, ...]) -> float:
    mine = [r for r in rows if tuple(r["topics"]) == topics]
    n = max(sum(int(r["page2_items"]) for r in mine), 1)
    return round(sum(int(r[key]) for r in mine) / n, 4)


def share(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def freshness(pages: list[dict[str, Any]]) -> dict[str, Any]:
    ages = [a for p in pages for a in p["ages_h"]]
    return {
        "items": len(ages),
        "share_under_2h": share([a < 2 for a in ages]),
        "share_under_30min": share([a < 0.5 for a in ages]),
        "median_age_h": round(statistics.median(ages), 2) if ages else None,
    }


def pick_samples(times: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Up to 10 distinct drops per persona (steady times), τ_feed merges first: those are
    what the ADR 0013 fallback is about; same-cluster drops are ingestion's verdict."""
    out = []
    for persona in sorted({p for t in times for p in t["personas"]}):
        seen: set[tuple[str, ...]] = set()
        pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t in times:
            if not t["steady"]:
                continue
            for d in t["personas"].get(persona, {}).get("drops", []):
                key = (d["dropped"]["id"], *sorted(k["id"] for k in d["kept"]))
                if key not in seen:
                    seen.add(key)
                    pool[d["via"]].append({**d, "persona": persona, "at": t["now"]})
        rng = random.Random(0)  # noqa: S311 (sampling, not security)
        chosen = rng.sample(pool["tau_feed"], min(SAMPLES_PER_PERSONA, len(pool["tau_feed"])))
        rest = SAMPLES_PER_PERSONA - len(chosen)
        chosen += rng.sample(pool["cluster"], min(rest, len(pool["cluster"])))
        out += chosen
    for k, s in enumerate(out):
        s["n"] = k + 1
    return out


def grouping(pages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "largest_group": max((x["largest_group"] for x in pages), default=0),
        "largest_group_mean": mean(pages, "largest_group"),
        "groups_over_cap_mean": mean(pages, "groups_over_cap"),
        "cap_drops_page_1_mean": mean(pages, "cap_drops_page_1"),
        "cap_drops_slate_mean": mean(pages, "cap_drops_slate"),
        "cap_drops_by_way": dict(Counter(d["via"] for x in pages for d in x["drops"])),
    }


def new_reader_check(steady: list[dict[str, Any]], version: str) -> dict[str, Any]:
    rows = [r for t in steady for r in t["new_reader"]
            if r["ranking"] == version and "skipped" not in r]  # fmt: skip
    items = sum(r["page2_items"] for r in rows)
    return {
        "cold_start_k": rows[0]["cold_start_k"] if rows else None,
        "readers": len(rows),
        "page1_freshness_cold_segment": freshness(
            [{"ages_h": r["page1_ages_h"]} for r in rows if "page1_ages_h" in r]
        ),
        "page2_items": items,
        "share_followed_topics": round(sum(r["page2_followed"] for r in rows) / items, 4),
        "share_clicked_topic": round(sum(r["page2_clicked_topic"] for r in rows) / items, 4),
        "by_topic_set": {
            ",".join(s): {
                "share_followed_topics": page2_share(rows, "page2_followed", s),
                "share_clicked_topic": page2_share(rows, "page2_clicked_topic", s),
            } for s in NEW_READER_SETS
        },
        "skipped": sum(1 for t in steady for r in t["new_reader"]
                       if r["ranking"] == version and "skipped" in r),
        "degraded": sum(1 for r in rows if r["degraded"]),
    }  # fmt: skip


def labelled_samples() -> dict[str, Any]:
    """The rank-2 run's story-cap samples (headlines stay local) and their labels."""
    times = json.loads((LABELLED / "report.json").read_text())["times"]
    samples = pick_samples(times)  # deterministic: the same numbering as when labelled
    SAMPLES.write_text(json.dumps(samples, indent=1) + "\n")
    labels: dict[str, Any] = json.loads(LABELS.read_text()) if LABELS.is_file() else {}
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    for s in samples:
        lab = labels.get("labels", {}).get(str(s["n"]))
        if lab is not None:
            by_label[f"{s['via']}:{s['kept'][0]['pair_type']}"][lab] += 1
    return {
        "run": "rank-2, title-title tau_feed 0.925 (grouping doesn't depend on the weights)",
        "n": len(samples),
        "by_way": dict(Counter(s["via"] for s in samples)),
        "labels": {k: dict(v) for k, v in by_label.items()},
        "labelled_by": labels.get("labelled_by"),
        "note": "examples with headlines stay in data/persona-replay/ (gitignored) "
        "and the PR description",
    }


def report() -> None:
    times = json.loads(RAW.read_text())["times"]
    steady = [t for t in times if t["steady"]]
    personas = sorted({p for t in times for v in t["personas"].values() for p in v})
    per: dict[str, Any] = {}
    for p in personas:
        per[p] = {}
        for v in PERSONA_RANKINGS:
            pages = [t["personas"][v][p] for t in steady if p in t["personas"][v]]
            per[p][v] = {
                "clock_times": len(pages),
                "freshness": freshness(pages),
                **grouping(pages),
                "degraded_pages": sum(1 for x in pages if x["degraded"]),
            }
    agg = {
        "window": {"first": times[0]["now"], "last": times[-1]["now"],
                   "clock_times": len(times), "steady_clock_times": len(steady),
                   "steady_rule": "at least 24 h of replayed news before the clock time"},
        "rankings": {
            "rank-2": "w_fresh 0.05, w_trend 0.025 (cold segment 0.05 / 0.1)",
            "rank-3": "the default: w_fresh 0.1, w_trend 0.05 (cold segment unchanged)",
        },
        "personas": per,
        "all_personas": {
            v: {
                "freshness": freshness(
                    [t["personas"][v][p] for t in steady for p in t["personas"][v]]),
                **grouping([t["personas"][v][p] for t in steady for p in t["personas"][v]]),
            } for v in PERSONA_RANKINGS
        },
        "new_reader_check": {v: new_reader_check(steady, v) for v in NEW_READER_RANKINGS},
        "sampled_cap_drops": labelled_samples(),
    }  # fmt: skip
    if REVIEW_RUN.is_file():
        agg["labelled_review_run"] = {
            "why": "the sampled cap drops were first labelled on this run, with title-title "
            "tau_feed 0.875; one sample (#41, three different AP digests) was two different "
            "stories, so ADR 0013's fallback raised title-title tau_feed to 0.925",
            **json.loads(REVIEW_RUN.read_text()),
        }
    AGGREGATES.write_text(json.dumps(agg, indent=1) + "\n")
    print(f"wrote {AGGREGATES} and {SAMPLES}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument(
        "--out", type=Path, default=OUT,
        help="run only: where report.json goes (keep earlier runs by giving a new folder)",
    )  # fmt: skip
    args = parser.parse_args()
    if args.command == "run":
        out = asyncio.run(run())
        args.out.mkdir(parents=True, exist_ok=True)
        raw = args.out / RAW.name
        raw.write_text(json.dumps(out, indent=1, default=str) + "\n")
        print(f"wrote {raw}")
    else:
        report()


if __name__ == "__main__":
    main()
