import pytest

from firstread_ingest.main import describe, main


def test_describe_lists_pipeline_stages() -> None:
    assert describe().startswith("fetch")
    assert "upsert" in describe()


def test_main_runs(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    assert "firstread-ingest" in capsys.readouterr().out
