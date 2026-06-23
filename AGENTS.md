# AGENTS.md

## Project Overview

This repository is becoming a local-first Loop Engineering Workbench. It helps
operators inspect and improve agent loops: context selection, retrieval, answer
drafting, mechanical checks, verifier decisions, retries, refusals, evals, and
eventual replay. The current built-in capability is document context: PDF, DOCX,
TXT, and MD uploads are chunked, indexed with FAISS, and used as evidence for a
document-grounded agent loop.

Primary runtime files:

- `src/DocumentQA.py`: ingestion, encoding detection, embeddings, vector search,
  LLM backend selection, retrieval chain, Ollama adapter, and query handling.
- `src/loop_engine.py`: provider-neutral loop primitives for typed run, step,
  policy, verifier, human-review, and report records.
- `src/app.py`: Gradio UI wiring and user-facing status messages.
- `tests/`: regression coverage for backend honesty, ingestion, encoding, app
  status, retrieval behavior, evals, and loop primitives.

## Setup Commands

- Install runtime dependencies: `python -m pip install -r requirements.txt`
- Install test/audit dependencies: `python -m pip install -r requirements-dev.txt`
- Run the app locally: `python src/app.py`
- Run tests: `python -m pytest`
- Compile check: `python -m py_compile src/app.py src/DocumentQA.py src/golden_eval.py src/loop_engine.py src/ollama_model_eval.py tests/test_app.py tests/test_document_qa.py tests/test_golden_document_eval.py tests/test_loop_engine.py tests/test_ollama_model_eval.py`
- Dependency checks: `python -m pip check` and `python -m pip_audit -r requirements.txt --strict`

## Non-Negotiable Contracts

- Do not silently fall back from explicit real LLM backends. `LLM_BACKEND=endpoint`,
  `LLM_BACKEND=local`, and `LLM_BACKEND=ollama` must fail closed when credentials,
  model loading, server reachability, or model availability are invalid. Mock mode
  must be explicit or an `auto` demo fallback.
- Do not let UI status imply inference readiness before inference has actually
  happened. Document upload means indexed, not proven ready.
- Document upload replacement must be transactional. Failed uploads must preserve
  the previous successful document, vector store, retrieval chain, and query
  behavior while recording the failed attempt in the processing report.
- `DocumentQA.status()` and its processing report are the source of truth for UI
  status. UI code and tests should not inspect random internal attributes.
- Answer traces and citations must come from the retrieved chunks used to build
  the LLM prompt. Do not bolt on citations from a separate post-answer lookup.
- Answer self-checking must remain document-only. Cheap mechanical checks and
  deterministic refutation prefilters may reject bad answers, but only a real
  backend verifier may label an answer `supported`. Mock/demo mode must report
  mechanically valid answers as `not_verified`, not `supported`.
- Golden document evals must remain provider-free and deterministic. They should
  exercise upload, retrieval, citation trace, self-check, retry, and fail-closed
  behavior without requiring a live Ollama or Hugging Face backend in CI.
- Live Ollama model comparison is optional and manual. Do not add it to CI; use
  provider-free tests for CI, keep live model runs unload-aware, and keep
  multi-model local eval behind an explicit override because Mac unified memory
  can be exhausted quickly. The live eval command must accept only loopback
  Ollama base URLs.
- Text upload default is `Auto`. Ambiguous legacy bytes must fail closed instead
  of mojibaking. `UTF-8 / Western` and explicit legacy encodings are opt-ins.
- Docker image publication belongs to `main` only. `dev`, PR, and manual workflow
  runs may validate builds, but must not publish release images.
- Preserve deterministic generation for context-grounded answers unless a
  test-backed product reason requires changing it.
- Ollama support is explicit, not part of `auto`. Use `OLLAMA_MODEL`,
  `OLLAMA_BASE_URL`, and `OLLAMA_TIMEOUT`; do not route Ollama through
  Hugging Face model ids or tokens.
- Product direction is local-first. Keep Hugging Face support as an optional
  hosted/deployment path, but do not make new happy-path features require a
  Hugging Face token when they can run through Ollama.
- Product identity is Loop Engineering Workbench. Document answering is now a
  document context provider capability, not the repo's strategic identity.
- Typed loop records are the contract surface for future agent work. Add or
  update `LoopRun`, `LoopStep`, `LoopDecision`, `LoopReport`, `LoopPolicy`,
  `VerificationResult`, and `HumanReviewRequest` before adding planner,
  multi-agent, tool, replay, or framework-adapter behavior.

## Engineering Loop

Use this loop for every non-trivial change:

1. Explore the current code and tests before editing.
2. State the narrow behavior contract you are changing.
3. Make the smallest code change that improves that contract.
4. Add or update tests for the behavior, including hostile environment cases when
   env vars, encodings, credentials, local model servers, or CI triggers are involved.
5. Run focused tests first, then the broader validation commands when risk
   touches shared behavior.
6. When subagent tooling is available, ask a review subagent for actionable
   findings before final handoff. Otherwise, perform a direct review pass and
   report that no subagent tooling was available.
7. Stage only intentional files and report any checks that could not be run.

## Refactoring Rules

- Prefer explicit status/configuration objects over UI code reading random
  internal attributes.
- Treat failed replacement uploads as hostile state-integrity cases. Assert the
  old document is still active and queryable after every failed upload path.
- Preserve `DocumentQA.query()` as the simple string API; add richer answer
  evidence through structured result objects such as `query_with_trace()`.
- Keep `DocumentQA` honest before making it clever. Reliability beats agentic
  theater.
- Do not add OpenAI Agents SDK, LangGraph, or Microsoft Agent Framework as a core
  dependency until provider-neutral loop reports are real and test-covered.
- Do not add autonomous tools or multi-agent behavior until middleware,
  guardrail, telemetry, and human-review boundaries exist in the loop contract.
- Before adding planner/tool/agent loops, add or update golden document evals
  that prove the base RAG loop still cites, verifies, retries, and refuses
  honestly.
- Add abstractions only when they reduce risk or remove repeated policy logic.
- Keep docs and comments aligned with runtime behavior. Stale comments are bugs
  waiting to be reintroduced.

## Recommended Validation

For narrow docs-only changes:

- `git diff --check`

For Python behavior changes:

- `python -m pytest`
- `python -m pytest tests/test_golden_document_eval.py -q`
- `python -m py_compile src/app.py src/DocumentQA.py src/golden_eval.py src/loop_engine.py src/ollama_model_eval.py tests/test_app.py tests/test_document_qa.py tests/test_golden_document_eval.py tests/test_loop_engine.py tests/test_ollama_model_eval.py`
- `python -m pip check`

For dependency or security-sensitive changes:

- `python -m pip install --dry-run -r requirements.txt`
- `python -m pip_audit -r requirements.txt --strict`
