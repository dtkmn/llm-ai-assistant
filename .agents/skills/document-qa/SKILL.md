---
name: document-qa-engineering
description: Use when changing document ingestion, text encoding, retrieval, LLM backend routing, Gradio upload/query behavior, or CI release policy for this RAG assistant.
---

# Document QA Engineering

## Purpose

Use this skill to evolve the app without weakening its core promise: answer
questions from uploaded documents honestly, with clear backend status, safe text
decoding, and reproducible validation.

## Operating Loop

1. Perceive: identify which user-facing contract is affected: ingestion,
   encoding, retrieval, LLM backend, UI status, dependency hygiene, or release
   automation.
2. Plan: write down the smallest behavior change that improves that contract.
3. Act: edit the owned files only. Keep unrelated refactors out of the patch.
4. Observe: run focused tests for the changed behavior and inspect failure modes,
   not just success paths.
5. Verify: run broader checks when shared behavior changes.
6. Review: when subagent tooling is available, ask a subagent for actionable
   findings on the actual diff. Otherwise, perform a direct review pass and
   report that no subagent tooling was available.
7. Ship: stage intentional files and report validation honestly.

## Runtime Contracts

- `auto` backend may fall back to `MockLLM` for demos when credentials are
  absent, but explicit `endpoint` and `local` backends must fail closed.
- `hf_token="dummy"` is valid only for mock/demo paths.
- Upload status must say `indexed` for real backends, because endpoint readiness
  is not proven until the first inference call.
- Upload replacement must be transactional. Failed uploads must not replace the
  previous successful document, vector store, retrieval chain, or query behavior.
- UI upload/runtime status must come from `DocumentQA.status()` and the latest
  `DocumentProcessingReport`, not ad hoc reads of internal attributes.
- Text encoding default is `Auto`. Ambiguous non-UTF legacy files must not be
  silently decoded as Western text.
- Explicit encoding selections are user intent. Preserve valid CP1250, CP1251,
  CP1252, CP1254, CP1257, Latin-1, UTF-8, UTF-16, and UTF-32 behavior when
  touching text ingestion.

## Files To Inspect First

- `src/DocumentQA.py`
- `src/app.py`
- `tests/test_document_qa.py`
- `tests/test_app.py`
- `.github/workflows/tests.yml`
- `.github/workflows/docker-publish.yml`

## Validation Matrix

For ingestion or encoding changes:

- `python -m pytest tests/test_document_qa.py -q`
- Include replacement-failure tests that prove the previous document remains
  active and queryable.
- Add hostile tests for env pollution, mojibake, invalid bytes, or unsupported
  explicit encodings as appropriate.

For UI status changes:

- `python -m pytest tests/test_app.py -q`
- Confirm mock mode and real backend wording remain honest.

For backend routing changes:

- Exercise `LLM_BACKEND=auto`, `mock`, `endpoint`, and `local` paths.
- Clear or set `HUGGINGFACEHUB_API_TOKEN`, `HF_ENDPOINT_URL`, and `LLM_BACKEND`
  inside tests so shell state cannot poison CI.

For release automation:

- Parse YAML if possible.
- `git diff --check`
- Ensure Docker publish remains limited to `push` on `refs/heads/main`.

## Stop Conditions

Stop and ask for direction if a change requires new model providers, persistent
storage, authentication, a database, or background job infrastructure. Those are
product architecture decisions, not cleanup.
