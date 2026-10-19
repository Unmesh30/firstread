import json
from pathlib import Path

import pytest

from firstread_core.db import MigrationError, check_migration_dsn
from firstread_ingest.main import finish, main
from firstread_ingest.report import GDELT_CREDIT, markdown

SUMMARY = {
    "fetched": 10, "inserted": 4, "duplicates": 3, "near_duplicates": 1, "clustered": 1,
    "errors": 0, "gdelt_calls": 2, "redis_commands": 12, "status": "ok",
    "details": {
        "dropped": {"stale": 2, "press_release:financial_results": 1},
        "topics": {"world": 3, "us": 1},
        "gdelt": {"files": [{"slot": "x", "pending": True}], "records": 9, "candidates": 4,
                  "selected": 3, "capped": {}, "dropped": {"blocked_domain": 1},
                  "top_domains": {"wisn.com": 3}, "stale": True,
                  "newest_available": "20261015000000", "cursor": "20261014234500"},
        "pools": {"fresh": 4}, "retention": {"articles": 0}, "size": {"db": 12_000_000},
        "feed_errors": {"bbc-world": "HTTP 500"},
    },
}  # fmt: skip


def test_markdown_summary_has_counts_gdelt_credit_and_warnings() -> None:
    md = markdown(SUMMARY, dry_run=True)
    assert "dry run" in md
    assert "| inserted | 4 |" in md
    assert "press_release:financial_results" in md
    assert "**Stale:**" in md
    assert "1 pending" in md
    assert "wisn.com" in md
    assert "`bbc-world`: HTTP 500" in md
    assert GDELT_CREDIT in md
    assert "https://www.gdeltproject.org" in md


@pytest.mark.parametrize(("status", "code"), [("ok", 0), ("partial", 0), ("failed", 1)])
def test_exit_codes_and_job_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, code: int
) -> None:
    out = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(out))
    assert finish(json.loads(json.dumps(SUMMARY)) | {"status": status}, dry_run=False) == code
    assert f"`{status}`" in out.read_text()


def test_non_dry_run_without_secrets_fails_fast(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    with pytest.raises(SystemExit) as e:
        main(["run", "--require-db"])
    assert e.value.code == 1
    assert "use --dry-run until P5" in capsys.readouterr().out


def test_migrations_refuse_the_transaction_pooler() -> None:
    with pytest.raises(MigrationError, match="5432"):
        check_migration_dsn("postgresql://u:p@aws-0-us-east-1.pooler.supabase.com:6543/postgres")
    check_migration_dsn("postgresql://u:p@db.abc.supabase.co:5432/postgres")
    check_migration_dsn("postgresql://u:p@localhost/firstread")
