# Contributing

## Setup

```bash
make setup   # uv sync, npm ci, pre-commit hooks, .env
make up      # local Docker stack
```

## Workflow

1. Branch from `main`: `phase-N-<name>` for phases, otherwise `<type>/<short-name>`.
2. Make small commits using [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`, `refactor:`.
3. Every change ships with tests. Before pushing, run:
   ```bash
   make lint typecheck test
   ```
4. Open a PR with `gh pr create`. `main` is protected: CI (`python`, `mobile`, `docker`, `secrets`) must be green before merging.
5. Merge with rebase (`gh pr merge --rebase --delete-branch`) so each commit stays on `main`.

## Rules

- Python dependencies: `uv add <pkg>` inside the right member (`--package firstread-api`), never `pip install`.
- App dependencies: `npx expo install <pkg>` (picks SDK-compatible versions).
- Text cleaning, embeddings and feature code live only in `packages/core`.
- Never commit secrets, MIND data, caches or model binaries.
- Never add a paid service or billable tier.
- Significant decisions get an ADR in `docs/adr/`.
