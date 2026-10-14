# FirstRead — how to work in this repo

Read @SPEC.md before planning any phase. SPEC.md says *what* to build; this file says *how* to work.

## Project in one line
Real-time news recommender that ranks brand-new articles before they have clicks. Monorepo: `apps/mobile` (Expo), `services/api` (FastAPI on Cloud Run), `services/ingest` (GitHub Actions cron), `ml` (training/eval on MIND), `packages/core` (shared Python).

## Hard rules
- IMPORTANT: $0/month. Only the free services listed in SPEC.md §4. Never add a paid service, paid tier, or anything that can bill without asking me first.
- IMPORTANT: Never commit secrets or print secret values. `.env` is gitignored; document every variable in `.env.example`.
- Never commit MIND data, caches or model binaries. Data goes in `data/` (gitignored); models go to Hugging Face Hub with a model card.
- Embedding, text-cleaning and feature code live only in `packages/core` and are imported everywhere else. Never duplicate them (train/serve skew).
- Load tests run only against the local Docker stack, never production (free-tier quotas).

## Commands
- `make setup` · `make lint` · `make fmt` · `make typecheck` · `make test` · `make up` · `make down`
- Python: use `uv` only (`uv sync`, `uv run pytest`, `uv run ruff check`, `uv run mypy`). Never `pip install` directly.
- App: `cd apps/mobile && npx expo start`; web build `npx expo export -p web`.

## Workflow
- Work one phase at a time, in the order of SPEC.md §16. Don't start the next phase unless asked.
- Start each phase in plan mode and wait for my approval. If a requirement is ambiguous, ask me with AskUserQuestion instead of guessing.
- Branch `phase-N-<name>`, small commits using Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), open a PR with `gh`, merge only when CI is green.
- Every change ships with tests. Before saying "done", run lint, typecheck and tests, and show me the output as evidence.
- Verify external APIs and free-tier limits against current official docs before building on them; note what you checked in the learn chapter.
- Prefer boring, well-known libraries. Pin versions. Keep Docker images small (slim base, multi-stage builds).

## Documentation (IMPORTANT: the owner must be able to rebuild everything from the docs alone)
At the end of every phase, write `docs/learn/NN-<phase>.md` with these sections:
1. What we built and why (plain English, with an analogy)
2. Concepts from first principles (reader: a new grad who knows Python and ML basics)
3. Architecture of this phase (Mermaid diagram)
4. Walkthrough of the important files: what each does and why it exists
5. Rebuild from scratch: exact commands and code, in order
6. How to test and verify it, and how to read the results
7. Decisions and alternatives considered (link the ADRs)
8. What went wrong and the gotchas
9. Ten interview questions about this phase, with strong answers
10. Glossary of new terms

Also:
- Record significant decisions as ADRs in `docs/adr/NNNN-title.md` (Context, Decision, Alternatives, Consequences).
- Keep `docs/architecture.md` and `README.md` current.

## Gotchas
- Supabase pooler in transaction mode (port 6543) breaks prepared statements: use asyncpg with `statement_cache_size=0`.
- Expo web on GitHub Pages needs `experiments.baseUrl` = `/firstread` and a `.nojekyll` file.
- Expo `web.output: "static"` pre-renders routes in Node at build time, so never touch `window`/`localStorage`/`AsyncStorage` at module import time; access them only inside effects or behind platform checks.
- GitHub scheduled workflows can run late, and are disabled after 60 days without repo activity.
- Credit GDELT with a link to gdeltproject.org wherever its data appears.
- Embeddings are L2-normalized, so inner product equals cosine similarity. Use the matching pgvector operator class.
- pgvector lives in the `extensions` schema (Supabase convention, also locally), which isn't on the local `search_path`: write `extensions.halfvec(384)` and `OPERATOR(extensions.<#>)` explicitly in SQL.
