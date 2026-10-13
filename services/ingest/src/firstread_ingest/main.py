"""Entry point for the ingestion job. Phase 0 placeholder."""

from firstread_core import normalize_whitespace


def describe() -> str:
    """Return a one-line description of what the job will do."""
    return normalize_whitespace("fetch -> normalize -> dedupe -> classify -> embed -> upsert")


def main() -> None:
    print(f"firstread-ingest: not implemented yet ({describe()})")


if __name__ == "__main__":
    main()
