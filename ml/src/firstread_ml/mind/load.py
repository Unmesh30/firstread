"""Parse MIND TSVs into int-indexed parquet (``data/mind/processed/<variant>``).

``news.tsv``: id, category, subcategory, title, abstract, url, title entities,
abstract entities. ``behaviors.tsv``: impression id, user id, time, history
(space-separated ids, oldest first), impressions (``N123-1 N456-0``).

Train and dev ``news.tsv`` overlap; we take their union, give every article a
dense int32 index (``nidx``) and rewrite histories and candidates as indices.
Article text stays in ``data/`` (gitignored) and is never committed.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from firstread_ml.mind.data import Impressions
from firstread_ml.mind.manifest import split_name
from firstread_ml.paths import mind_raw_dir, processed_dir

SPLITS = ("train", "dev")
NEWS_COLS = [
    "nid",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
FORMAT_VERSION = 1  # bump when the processed layout changes


def _read_tsv(path: Path, columns: list[str]) -> pl.DataFrame:
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=columns,
        quote_char=None,
        schema_overrides=dict.fromkeys(columns, pl.String),
    )


def read_news(path: Path) -> pl.DataFrame:
    return _read_tsv(path, NEWS_COLS).with_columns(pl.col("abstract").fill_null(""))


def _ids_to_index(df: pl.DataFrame, col: str, ids: pl.DataFrame) -> pl.Series:
    """Space-separated ids in ``col`` -> list[int32] of article indices (unknown -> -1).

    Explode + join is much faster than mapping inside ``list.eval``.
    """
    exploded = (
        df.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("row"), pl.col(col).fill_null(""))
        .with_columns(pl.col(col).str.split(" "))
        .explode(col)
        .filter(pl.col(col) != "")
        .with_row_index("pos")
        .join(ids, left_on=col, right_on="nid", how="left", maintain_order="left")
        .with_columns(pl.col("nidx").fill_null(-1))
    )
    lists = exploded.group_by("row", maintain_order=True).agg(pl.col("nidx").sort_by("pos"))
    full = pl.DataFrame({"row": pl.int_range(df.height, dtype=pl.UInt32, eager=True)})
    out = full.join(lists, on="row", how="left", maintain_order="left")["nidx"]
    return out.fill_null(pl.lit([], dtype=pl.List(pl.Int32))).alias(col)


def read_behaviors(path: Path, ids: pl.DataFrame) -> pl.DataFrame:
    df = _read_tsv(path, BEHAVIOR_COLS).with_columns(
        # "N123-1 N456-0": candidate ids without labels, and the labels alone.
        pl.col("impressions").str.replace_all(r"-\d", "").alias("cand_ids"),
        pl.col("impressions")
        .str.split(" ")
        .list.eval(pl.element().str.slice(-1).cast(pl.Int8))
        .alias("label"),
    )
    return pl.DataFrame(
        [
            df["impression_id"].cast(pl.Int64).alias("imp_id"),
            df["time"].str.to_datetime(TIME_FORMAT).dt.epoch("s").alias("time"),
            _ids_to_index(df, "history", ids),
            _ids_to_index(df, "cand_ids", ids).alias("cand"),
            df["label"],
        ]
    )


def _source_fingerprint(raw_dir: Path, variant: str) -> dict[str, object]:
    marks = {}
    for split in SPLITS:
        marker = raw_dir / split_name(variant, split) / ".verified"
        marks[split] = marker.read_text().strip() if marker.is_file() else "unverified"
    return {"format_version": FORMAT_VERSION, "sources": marks}


def prepare(variant: str, raw_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build processed parquet for ``variant`` (skipped if already up to date)."""
    raw_dir = raw_dir or mind_raw_dir()
    out_dir = out_dir or processed_dir(variant)
    split_dirs = {s: raw_dir / split_name(variant, s) for s in SPLITS}
    for d in split_dirs.values():
        if not (d / "behaviors.tsv").is_file():
            raise FileNotFoundError(
                f"{d} not found. Download it first: make mind-download VARIANT={variant} "
                "ACCEPT_LICENSE=1"
            )
    fingerprint = _source_fingerprint(raw_dir, variant)
    done = out_dir / "_SUCCESS.json"
    if done.is_file() and json.loads(done.read_text()) == fingerprint:
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    news = (
        pl.concat([read_news(split_dirs[s] / "news.tsv") for s in SPLITS])
        .unique("nid", keep="first", maintain_order=True)
        .with_row_index("nidx")
        .with_columns(pl.col("nidx").cast(pl.Int32))
        .select("nidx", "nid", "category", "subcategory", "title", "abstract")
    )
    news.write_parquet(out_dir / "news.parquet")
    ids = news.select("nid", "nidx")

    stats: dict[str, object] = {}
    for s in SPLITS:
        beh = read_behaviors(split_dirs[s] / "behaviors.tsv", ids)
        unknown = int(
            beh.select(
                (pl.col("history").list.eval(pl.element() < 0).list.sum()).sum()
                + (pl.col("cand").list.eval(pl.element() < 0).list.sum()).sum()
            ).item()
        )
        if unknown:
            raise ValueError(f"{s}: {unknown} article ids are missing from news.tsv")
        beh.sort("time", "imp_id").write_parquet(out_dir / f"behaviors_{s}.parquet")
        stats[s] = beh.height
    done.write_text(json.dumps(fingerprint, indent=1) + "\n")
    print(f"prepared MIND-{variant}: {news.height} articles, impressions {stats} -> {out_dir}")
    return out_dir


def load_news(variant: str, root: Path | None = None) -> pl.DataFrame:
    return pl.read_parquet((root or processed_dir(variant)) / "news.parquet")


def load_impressions(variant: str, split: str, root: Path | None = None) -> Impressions:
    df = pl.read_parquet((root or processed_dir(variant)) / f"behaviors_{split}.parquet")
    return impressions_from_frame(df)


def impressions_from_frame(df: pl.DataFrame) -> Impressions:
    def csr(col: str, dtype: type) -> tuple[np.ndarray, np.ndarray]:
        sizes = df[col].list.len().to_numpy().astype(np.int64)
        offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
        np.cumsum(sizes, out=offsets[1:])
        flat: np.ndarray = df[col].explode().drop_nulls().to_numpy().astype(dtype)
        return offsets, flat

    hist_off, hist = csr("history", np.int32)
    cand_off, cand = csr("cand", np.int32)
    _, label = csr("label", np.int8)
    return Impressions(
        imp_id=df["imp_id"].to_numpy().astype(np.int64),
        time=df["time"].to_numpy().astype(np.int64),
        hist_offsets=hist_off,
        hist=hist,
        cand_offsets=cand_off,
        cand=cand,
        label=label,
    )
