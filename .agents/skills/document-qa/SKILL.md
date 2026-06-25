---
name: loop-engineering
description: Use when changing loop contracts, document context, retrieval, LLM backend routing, Gradio upload/query behavior, evals, or CI release policy for AI Loop Engine.
---

# Loop Engineering

## Purpose

Use this skill to evolve the workbench without weakening its core promise: make
agent loops observable, testable, local-first, and honest. Document upload is the
first context provider, not the product boundary.

## Operating Loop

1. Perceive: identify which user-facing contract is affected: ingestion,
   encoding, retrieval, loop trace, LLM backend, UI status, dependency hygiene,
   or release automation.
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

- `auto` backend is local-first and real-backend-only: select Ollama and fail
  closed when Ollama or the configured model is unavailable. It must not fall
  back to mock.
- Ollama runtime URLs must be loopback local only. Remote/cloud model gateways
  must use `LLM_BACKEND=openai-compatible`, not `OLLAMA_BASE_URL`.
- Explicit real backends must fail closed: `ollama` and
  `openai-compatible`. Removed backend names such as `endpoint` and `local`
  must fail closed as invalid configuration.
- `openai-compatible` is the cloud/private-gateway deployment path. Use
  `OPENAI_COMPAT_BASE_URL`, `LLM_MODEL`, `EMBEDDINGS_MODEL`, optional
  `OPENAI_COMPAT_API_KEY`, and `OPENAI_COMPAT_TIMEOUT`. Plain HTTP is allowed
  only for loopback local development; remote gateways must use HTTPS.
- `LLM_BACKEND` is the single provider/runtime selector. Do not add a separate
  embedding backend variable. Configure the chat model with `LLM_MODEL` and the
  retrieval model with `EMBEDDINGS_MODEL`; provider-specific model env vars are
  compatibility aliases only.
- Native runtime defaults must be installed before Gradio, NumPy, FAISS,
  or other native-heavy imports in app entrypoints. Use `src.native_runtime`
  instead of duplicating env setup in modules that may be imported too late.
- Embedding runtime follows the selected provider. Ollama uses `/api/embed`;
  OpenAI-compatible gateways use `/embeddings`; mock mode uses built-in local
  hashing for deterministic demos/tests.
- `LLM_BACKEND=auto` must select Ollama only. It must not silently select mock,
  provider-specific hosted backends, or in-process model loading.
- Product identity is AI Loop Engine. Treat document answering as
  the first context provider capability, not the repo's strategic identity.
- Document context is the first `ContextProvider`; keep provider identity in
  loop reports instead of hardcoding document-specific assumptions in UI code.
- Typed loop records are the contract surface for future agent work. Add or
  update `LoopRun`, `LoopStep`, `LoopDecision`, `LoopReport`, `LoopPolicy`,
  `LoopSession`, `GuardrailDecision`, `LoopMiddleware`, `VerificationResult`, and
  `HumanReviewRequest` before adding planner, multi-agent, tool, replay, or
  framework-adapter behavior.
- `AILoopEngine.query_with_trace()` must expose a `LoopReport` that matches the
  actual query path: prompt evidence, draft, mechanical check, verifier outcome,
  retry/refusal state, and final answer.
- Completed query reports should be retained in bounded in-memory `LoopSession`
  state. Local JSONL export may write raw reports for developer replay/debug
  artifacts; public UI traces must continue using the redacted report surface.
- Loop middleware is a guardrail/telemetry boundary. It may block, refuse, retry,
  or request human review, but it must not introduce autonomous tools by itself.
- Upload status must say `indexed` for real backends, because endpoint readiness
  is not proven until the first inference call.
- Upload replacement must be transactional. Failed uploads must not replace the
  previous successful document, vector store, retrieval chain, or query behavior.
- UI upload/runtime status must come from `AILoopEngine.status()` and the latest
  `DocumentProcessingReport`, not ad hoc reads of internal attributes.
- Answer traces and citations must come from the same retrieved chunks used in
  the prompt. Keep `query()` string-compatible and expose richer evidence
  through structured APIs such as `query_with_trace()`.
- Answer self-checking should inspect the generated answer and trace. Cheap
  mechanical checks and deterministic refutation prefilters may reject bad
  answers, but only a real backend verifier may label an answer `supported`.
  Mock/demo mode must report mechanically valid answers as `not_verified`.
- Golden document evals must exercise the full provider-free QA loop: upload,
  retrieval, cited answer, self-check, retry, and fail-closed refusal. Do not
  require a live Ollama backend for these CI checks.
- `src.loop_eval --mode fake` is the provider-free CLI surface for JSON loop
  eval artifacts. It should include scored `LoopReport` evidence so humans can
  inspect phases, citations, verifier decisions, retries, refusals, and final
  decisions without a live model.
- Live Ollama model comparison is optional and manual. Keep it unload-aware,
  keep multi-model runs behind an explicit override, and never make CI require
  resident local models. Keep the live eval base URL loopback-only and prefer
  one-model, one-case smoke runs before full-model sweeps.
- OpenAI Agents SDK, LangGraph, and Microsoft Agent Framework are future adapter
  targets, not core dependencies, until provider-neutral loop reports are real
  and test-covered.
- Framework adapter work must follow `docs/framework-adapter-strategy.md`.
  Export framework-shaped artifacts first; do not let any framework runtime own
  loop execution until the internal loop report contract remains stable under
  tests.
- OpenAI trace-shaped export, LangGraph manifest export, and `src.loop_export`
  must remain dependency-free. Public/redacted export is the default; raw
  diagnostics require explicit opt-in.
- Do not add autonomous tool use or multi-agent behavior until middleware,
  guardrail, telemetry, and human-review boundaries exist in the loop contract.
- Text encoding default is `Auto`. Ambiguous non-UTF legacy files must not be
  silently decoded as Western text.
- Explicit encoding selections are user intent. Preserve valid CP1250, CP1251,
  CP1252, CP1254, CP1257, Latin-1, UTF-8, UTF-16, and UTF-32 behavior when
  touching text ingestion.
- `pyproject.toml` is the local-development dependency contract. Keep
  `requirements.txt` and `requirements-dev.txt` synchronized as pip-compatible
  deployment exports.

## Files To Inspect First

- `src/ai_loop_engine.py`
- `src/ai_loop_runtime.py`
- `src/context_providers.py`
- `src/document_ingestion.py`
- `src/runtime_config.py`
- `src/model_adapters.py`
- `src/DocumentQA.py`
- `src/loop_engine.py`
- `src/app.py`
- `src/golden_eval.py`
- `src/loop_eval.py`
- `src/ollama_model_eval.py`
- `src/adapters/`
- `src/loop_export.py`
- `docs/framework-adapter-strategy.md`
- `pyproject.toml`
- `requirements.txt`
- `requirements-dev.txt`
- `tests/test_document_qa.py`
- `tests/test_app.py`
- `tests/test_golden_document_eval.py`
- `tests/test_loop_engine.py`
- `tests/test_loop_eval.py`
- `tests/test_ollama_model_eval.py`
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

- Exercise `LLM_BACKEND=auto`, `mock`, `ollama`, `openai-compatible`,
  and invalid removed backend names such as `endpoint` and `local` when the
  change touches backend selection.
- Clear or set `LLM_BACKEND`, `LLM_MODEL`, `EMBEDDINGS_MODEL`,
  `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_BASE_URL`,
  `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_MODEL`,
  `OPENAI_COMPAT_EMBED_MODEL`, and `OPENAI_COMPAT_API_KEY` inside tests so
  shell state cannot poison CI.

For answer-loop or agent-pattern changes:

- `uv lock --check` when packaging metadata or dependency files changed.
- `python -m pytest tests/test_loop_engine.py -q`
- `python -m pytest tests/test_golden_document_eval.py -q`
- `python -m pytest tests/test_loop_eval.py -q`
- `python -m pytest tests/test_ollama_model_eval.py -q`
- `python -m pytest tests/test_openai_trace_adapter.py -q` when adapter export
  behavior changes.
- `python -m pytest tests/test_langgraph_manifest_adapter.py -q` when LangGraph
  manifest behavior changes.
- `python -m pytest tests/test_loop_export.py -q` when JSONL adapter export CLI
  behavior changes.
- Assert cited supported answers, unsupported-answer refusal, and retry behavior.
- Keep eval fixtures deterministic and provider-free.
- Keep live Ollama comparison manual. Prefer one-model, one-case smoke runs on
  memory-constrained Macs, and do not make multi-model live eval the default.

For release automation:

- Parse YAML if possible.
- `git diff --check`
- Ensure Docker publish remains limited to `push` on `refs/heads/main`.

## Stop Conditions

Stop and ask for direction if a change requires new model providers, persistent
storage, authentication, a database, or background job infrastructure. Those are
product architecture decisions, not cleanup.
