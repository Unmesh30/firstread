# ADR 0005: Model artifacts on the Hugging Face Hub, versioned by tag with a SHA256 manifest

- Status: Accepted
- Date: 2026-10-14

## Context

Phase 1 produced two trained models that later phases depend on:
- the NRMS-lite **user encoder**, which the API (P3) runs per feed request;
- the **category classifier**, which ingestion (P2) runs per new article.

Both must be:

- **Portable to CPU serving without PyTorch or scikit-learn.** The API image deliberately excludes ML libraries (ADR 0001; checked in CI).
- **Versioned and pinned.** A service must load exactly the model it was tested with, and a new release must never change a running service by surprise.
- **Verifiable.** A downloaded file must be provably the one that was evaluated.
- **Free.** SPEC §4 lists the Hugging Face Hub (public model repo) as the place for model artifacts.
- **Licensed correctly.** The weights are derived from MIND, whose Microsoft Research License Terms allow non-commercial research only. The repository code is MIT; the weights are not.

## Decision

- **Format: ONNX.**
  - The user encoder uses PyTorch's dynamo exporter (opset 18, dynamic batch dimension only).
  - The classifier uses skl2onnx with zipmap off.
  - Services need only `onnxruntime`.
  - A **parity check** (max abs diff ≤ 1e-4 against PyTorch/sklearn, on empty, partial, full and real histories, plus batch invariance) runs in CI on tiny models and again on the real weights at release.
- **Where: `Unmesh30/firstread-models` on the Hugging Face Hub, public.**
- **Versions are Hub git tags (`vX.Y.Z`).**
  - Releases are immutable: `release upload` refuses if the tag already exists.
  - Services pin a tag, never `main`.
  - Model versions are independent of Python package versions.
- **Contents are an allowlist.** The release contains exactly:
  - `user_encoder.onnx`, `category_classifier.onnx`, `labels.json`, `population.npy`, `config.yaml`, `metrics.json`;
  - `README.md` (the model card, a copy of `docs/model-card.md`);
  - `manifest.json`.

  A build fails on any other file, so MIND rows, text or embedding caches can't leak.
- **`manifest.json` records:**
  - the SHA256 and size of every file;
  - the embedding-model pins (source repo, revision, file, fastembed version, text-prep version) and feature constants (`HISTORY_LEN`, `EMBED_DIM`);
  - `score_scale`, the MIND revision and library versions;
  - **`git_commit`**, the FirstRead commit the release was built from.

  Services verify hashes after download. `upload` re-downloads every file at the tag and checks it.
- **Provenance.**
  - A real build runs only on a clean `main` equal to `origin/main`, so `git_commit` is the commit that gets the git tag `vX.Y.Z`. `release verify-tag` checks this.
  - On a PR branch, only a **dry run** is allowed. It is marked `dry-run:<sha>` and can't be uploaded. A branch SHA wouldn't survive a rebase-merge.
  - A build never writes a tracked file, and it fails if `git status` changes.
- **Gate.** A real-data eval gate (ADR 0003, amended) must pass before a build is accepted.
- **License metadata:**
  - `license: other`, with `license_name: microsoft-research-license-mind` and `license_link` to the MIND license.
  - The card states that the weights are non-commercial and MIND-derived, and that the code is MIT.
- **Token handling.** `HF_TOKEN` (a write token) lives only in the local `.env`. The upload target sources it in a subshell and never prints it. CI never needs it.

## Alternatives considered

- **GitHub Release assets.**
  - Pros: free, and next to the code.
  - Cons: no model card rendering, no per-file revision history, and no `hf_hub_download` caching. It also mixes non-commercial weights into the MIT repository's release page.
  - We still create a GitHub release, but it only **links** to the Hub tag.
- **Google Cloud Storage or Artifact Registry.**
  - Artifact Registry's free tier is 0.5 GiB, which is needed for API images (SPEC §14). GCS can bill beyond small free quotas.
  - Neither renders a model card, and both are private by default, while the demo wants public, inspectable models.
- **Committing ONNX files to the repo (or git-lfs).**
  - Violates the "no model binaries in git" rule. git-lfs has small free quotas.
  - It would also put MIND-derived weights under the repository's MIT license by implication.
- **TorchScript / legacy exporter.** Kept only as a fallback. The dynamo exporter handled `nn.MultiheadAttention` with a key-padding mask on the first try, at parity ≤ 1e-6.
- **Shipping PyTorch checkpoints.** Would force torch into the API image (hundreds of MB, slower cold starts on Cloud Run).
- **Tagging the PR-branch commit.** Rejected because rebase-merge rewrites SHAs, so the tagged commit wouldn't be the built one.

## Consequences

- P2 and P3 add `onnxruntime` and a small loader that downloads a pinned tag, verifies `manifest.json`, and checks that `embedding` and `features` match `packages/core`'s constants. A mismatch is a startup error, not a silent skew.
- Releasing takes a few deliberate steps after the merge: build on `main`, upload (asks the owner), then tag and create the GitHub release (asks again). The steps are in the Phase 1 learn chapter.
- A new model version means a new tag and an explicit pin bump in services. Rollback is a pin change.
- Stored article embeddings stay valid across releases because the item tower is the identity (ADR 0004). Only a change of embedding model or text prep forces a re-embed, and the manifest's `embedding` block makes that visible.
- The Hub repo is public and the weights are non-commercial. A commercial launch needs retrained weights and a new license (SPEC §4).
