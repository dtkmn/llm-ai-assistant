# AGENTS.md

## Project Overview

This repository is a compact document QA/RAG assistant. It accepts PDF, DOCX,
TXT, and MD uploads, chunks document text, indexes chunks with FAISS, and answers
questions through a configurable Hugging Face LLM backend.

Primary runtime files:

- `src/DocumentQA.py`: ingestion, encoding detection, embeddings, vector search,
  LLM backend selection, retrieval chain, and query handling.
- `src/app.py`: Gradio UI wiring and user-facing status messages.
- `tests/`: regression coverage for backend honesty, ingestion, encoding, app
  status, and retrieval behavior.

## Setup Commands

- Install runtime dependencies: `python -m pip install -r requirements.txt`
- Install test/audit dependencies: `python -m pip install -r requirements-dev.txt`
- Run the app locally: `python src/app.py`
- Run tests: `python -m pytest`
- Compile check: `python -m py_compile src/app.py src/DocumentQA.py tests/test_app.py tests/test_document_qa.py`
- Dependency checks: `python -m pip check` and `python -m pip_audit -r requirements.txt --strict`

## Non-Negotiable Contracts

- Do not silently fall back from explicit real LLM backends. `LLM_BACKEND=endpoint`
  and `LLM_BACKEND=local` must fail closed when credentials or model loading are
  invalid. Mock mode must be explicit or an `auto` demo fallback.
- Do not let UI status imply inference readiness before inference has actually
  happened. Document upload means indexed, not proven ready.
- Text upload default is `Auto`. Ambiguous legacy bytes must fail closed instead
  of mojibaking. `UTF-8 / Western` and explicit legacy encodings are opt-ins.
- Docker image publication belongs to `main` only. `dev`, PR, and manual workflow
  runs may validate builds, but must not publish release images.
- Preserve deterministic generation for document QA unless a test-backed product
  reason requires changing it.

## Engineering Loop

Use this loop for every non-trivial change:

1. Explore the current code and tests before editing.
2. State the narrow behavior contract you are changing.
3. Make the smallest code change that improves that contract.
4. Add or update tests for the behavior, including hostile environment cases when
   env vars, encodings, credentials, or CI triggers are involved.
5. Run focused tests first, then the broader validation commands when risk
   touches shared behavior.
6. When subagent tooling is available, ask a review subagent for actionable
   findings before final handoff. Otherwise, perform a direct review pass and
   report that no subagent tooling was available.
7. Stage only intentional files and report any checks that could not be run.

## Refactoring Rules

- Prefer explicit status/configuration objects over UI code reading random
  internal attributes.
- Keep `DocumentQA` honest before making it clever. Reliability beats agentic
  theater.
- Add abstractions only when they reduce risk or remove repeated policy logic.
- Keep docs and comments aligned with runtime behavior. Stale comments are bugs
  waiting to be reintroduced.

## Recommended Validation

For narrow docs-only changes:

- `git diff --check`

For Python behavior changes:

- `python -m pytest`
- `python -m py_compile src/app.py src/DocumentQA.py tests/test_app.py tests/test_document_qa.py`
- `python -m pip check`

For dependency or security-sensitive changes:

- `python -m pip install --dry-run -r requirements.txt`
- `python -m pip_audit -r requirements.txt --strict`
