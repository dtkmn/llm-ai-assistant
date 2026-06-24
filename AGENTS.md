# AGENTS.md

## Project Overview

This repository is AI Loop Engine, a local-first engine for inspecting and
hardening AI loops: context selection, retrieval, answer
drafting, mechanical checks, verifier decisions, retries, refusals, evals, and
eventual replay. The current built-in capability is document context: PDF, DOCX,
TXT, and MD uploads are chunked, indexed with FAISS, and used as evidence for a
document-grounded agent loop.

Primary runtime files:

- `src/DocumentQA.py`: ingestion, encoding detection, embeddings, vector search,
  document context provider wrapper, LLM backend selection, retrieval chain,
  Ollama adapter, and query handling.
- `src/loop_engine.py`: provider-neutral loop primitives for typed run, step,
  policy, verifier, human-review, session, and report records.
- `src/loop_eval.py`: unified provider-free and optional live Ollama loop eval
  CLI with JSON artifacts containing scored `LoopReport` evidence.
- `src/adapters/`: dependency-free framework-shaped exports from loop reports.
  Current adapters are OpenAI trace-shaped JSON and LangGraph
  thread/checkpoint manifest JSON. They must not import framework SDKs or send
  data over the network.
- `src/loop_export.py`: local JSONL-to-adapter export CLI for OpenAI trace and
  LangGraph manifest artifacts.
- `src/app.py`: Gradio UI wiring and user-facing status messages.
- `docs/framework-adapter-strategy.md`: dependency-free adapter strategy for
  OpenAI trace-shaped export, LangGraph manifest export, and Microsoft workflow
  event export.
- `tests/`: regression coverage for backend honesty, ingestion, encoding, app
  status, retrieval behavior, evals, and loop primitives.

## Setup Commands

- Install local dependencies: `uv sync --dev`
- Pip fallback: `python -m pip install -r requirements-dev.txt`
- Run the app locally: `uv run ai-loop-engine` or `python -m src.app`
- Run tests: `uv run pytest` or `python -m pytest`
- Compile check: `python -m py_compile src/__init__.py src/app.py src/DocumentQA.py src/native_runtime.py src/golden_eval.py src/loop_engine.py src/loop_eval.py src/ollama_model_eval.py tests/test_app.py tests/test_document_qa.py tests/test_native_runtime.py tests/test_golden_document_eval.py tests/test_loop_engine.py tests/test_loop_eval.py tests/test_ollama_model_eval.py tests/test_packaging_metadata.py`
- Dependency checks: `python -m pip check` and `python -m pip_audit -r requirements.txt --strict`

## Non-Negotiable Contracts

- Do not silently fall back from explicit real LLM backends.
  `LLM_BACKEND=ollama` and `LLM_BACKEND=openai-compatible` must fail closed when
  model loading, server reachability, model availability, or gateway
  credentials are invalid. Mock mode must be explicit and must never be reached
  as a fallback. Removed backend names such as `endpoint` and `local` must fail
  closed as invalid configuration.
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
  behavior without requiring a live Ollama backend in CI. The
  unified `src.loop_eval --mode fake` CLI should emit JSON artifacts with the
  loop reports and pass/fail decisions for local inspection.
- Live Ollama model comparison is optional and manual. Do not add it to CI; use
  provider-free tests for CI, keep live model runs unload-aware, and keep
  multi-model local eval behind an explicit override because Mac unified memory
  can be exhausted quickly. The live `src.loop_eval --mode ollama` command must
  accept only loopback Ollama base URLs and should be judged by loop evidence,
  not answer text alone.
- Text upload default is `Auto`. Ambiguous legacy bytes must fail closed instead
  of mojibaking. `UTF-8 / Western` and explicit legacy encodings are opt-ins.
- Docker image publication belongs to `main` only. `dev`, PR, and manual workflow
  runs may validate builds, but must not publish release images.
- Preserve deterministic generation for context-grounded answers unless a
  test-backed product reason requires changing it.
- `LLM_BACKEND=auto` is local-first and real-backend-only: it selects Ollama
  and fails closed if Ollama or the configured model is unavailable. Explicit
  `LLM_BACKEND=mock` is only for tests/demos and must never be an automatic
  fallback.
- Cloud or gateway deployment should use `LLM_BACKEND=openai-compatible` with
  `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_MODEL`, optional
  `OPENAI_COMPAT_API_KEY`, and `OPENAI_COMPAT_TIMEOUT`. Plain HTTP is allowed
  only for loopback local development; remote gateways must use HTTPS.
- Native runtime defaults must be installed before Gradio, NumPy, FAISS,
  or other native-heavy imports in app entrypoints. Use `src.native_runtime`
  instead of duplicating env setup in modules that may be imported too late.
- Embedding runtime is separate from LLM runtime. On Apple Silicon/MPS,
  embeddings use the built-in CPU local hashing path for upload stability.
- `LLM_BACKEND=auto` must select Ollama only. It must not silently select mock,
  provider-specific hosted backends, or in-process model loading.
- Product direction is local-first. First-party model providers are Ollama and
  generic OpenAI-compatible gateways; do not reintroduce provider-token happy
  paths without an explicit product decision.
- Product identity is AI Loop Engine. Document answering is now a
  document context provider capability, not the repo's strategic identity.
- Typed loop records are the contract surface for future agent work. Add or
  update `LoopRun`, `LoopStep`, `LoopDecision`, `LoopReport`, `LoopPolicy`,
  `GuardrailDecision`, `LoopMiddleware`, `VerificationResult`, and
  `HumanReviewRequest` before adding planner, multi-agent, tool, replay, or
  framework-adapter behavior.
- `DocumentQA.query_with_trace()` must expose a `LoopReport` that matches the
  actual query path: prompt evidence, draft, mechanical check, verifier outcome,
  retry/refusal state, and final answer.
- Completed query loop reports must be retained in bounded in-memory
  `LoopSession` state keyed by `session_id`. Local replay JSONL export writes
  raw loop reports for developer diagnostics; public UI traces must keep using
  the redacted report surface.
- Loop middleware is a guardrail/telemetry boundary. It may block, refuse, retry,
  or request human review, but it must not introduce autonomous tools by itself.
- Framework adapters must export `LoopReport`/`LoopSession` surfaces before they
  execute framework runtimes. Follow `docs/framework-adapter-strategy.md`, keep
  default exports redacted/public, and do not add OpenAI Agents SDK, LangGraph,
  or Microsoft Agent Framework as core dependencies.
- Adapter public export is a safety boundary. Raw loop reports require explicit
  opt-in, and public adapter exports must fail closed/redact terminal
  guardrail-like decisions instead of leaking blocked draft content.
- `src.loop_export` must default to public/redacted output. Raw export is a
  local diagnostics path and must require an explicit `--raw` flag.
- `pyproject.toml` is the project metadata and local-development dependency
  contract. Keep `requirements.txt` and `requirements-dev.txt` as pip-compatible
  exports for deployment compatibility, and keep them synchronized with
  `pyproject.toml`.

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
- Keep replay/export behavior local and explicit. Do not add SQLite, server
  persistence, or background replay jobs until tests prove the product need.
- Keep `DocumentQA` honest before making it clever. Reliability beats agentic
  theater.
- Do not add OpenAI Agents SDK, LangGraph, or Microsoft Agent Framework as a core
  dependency until provider-neutral loop reports are real and test-covered.
- Before implementing framework adapters, read
  `docs/framework-adapter-strategy.md` and preserve its non-goals unless a new
  issue explicitly changes the adapter contract.
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

- `uv run pytest` or `python -m pytest`
- `uv run pytest tests/test_golden_document_eval.py -q`
- `uv run pytest tests/test_loop_eval.py -q`
- `uv run pytest tests/test_openai_trace_adapter.py -q`
- `uv run pytest tests/test_langgraph_manifest_adapter.py -q`
- `uv run pytest tests/test_loop_export.py -q`
- `uv lock --check`
- `python -m py_compile src/__init__.py src/app.py src/DocumentQA.py src/native_runtime.py src/golden_eval.py src/loop_engine.py src/loop_eval.py src/loop_export.py src/ollama_model_eval.py src/adapters/__init__.py src/adapters/base.py src/adapters/redaction.py src/adapters/openai_trace.py src/adapters/langgraph_manifest.py tests/test_app.py tests/test_document_qa.py tests/test_native_runtime.py tests/test_golden_document_eval.py tests/test_loop_engine.py tests/test_loop_eval.py tests/test_loop_export.py tests/test_ollama_model_eval.py tests/test_openai_trace_adapter.py tests/test_langgraph_manifest_adapter.py tests/test_packaging_metadata.py`
- `python -m pip check`

For dependency or security-sensitive changes:

- `python -m pip install --dry-run -r requirements.txt`
- `python -m pip_audit -r requirements.txt --strict`
