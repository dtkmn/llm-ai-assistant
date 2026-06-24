import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from src import app
from src.DocumentQA import (
    AnswerCitation,
    AnswerSelfCheck,
    AnswerTrace,
    DocumentProcessingError,
    DocumentProcessingReport,
    DocumentQAStatus,
    QueryResult,
)
from src.loop_engine import (
    LoopDecision,
    LoopPhase,
    LoopReport,
    LoopRun,
    LoopStep,
    VerificationOutcome,
    VerificationResult,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TOKENIZERS_PARALLELISM",
}


class FakeQA:
    fast_mode = True
    loaded_model_id = None
    loaded_model_label = None
    active_llm_backend = "mock"
    llm_backend = "mock"
    current_document_name = None
    latest_processing_report = None

    def process_document(self, document_path, text_encoding=None):
        self.document_path = document_path
        self.text_encoding = text_encoding
        self.current_document_name = document_path.split("/")[-1]
        active_backend = self.active_llm_backend or self.llm_backend
        active_model_label = (
            self.loaded_model_label
            or self.loaded_model_id
            or ("MockLLM (explicit demo)" if active_backend == "mock" else "unknown")
        )
        self.latest_processing_report = DocumentProcessingReport(
            attempted_document_name=self.current_document_name,
            active_document_name=self.current_document_name,
            success=True,
            phase="complete",
            file_extension=".txt",
            chunk_count=1,
            truncated=False,
            max_chunk_limit=2000,
            text_encoding_mode=text_encoding or "auto",
            backend=active_backend,
            model_label=active_model_label,
            error_message=None,
        )
        return self.status()

    def status(self):
        active_backend = self.active_llm_backend or self.llm_backend
        active_model_label = (
            self.loaded_model_label
            or self.loaded_model_id
            or ("MockLLM (explicit demo)" if active_backend == "mock" else "unknown")
        )
        return DocumentQAStatus(
            profile_label="FAST" if self.fast_mode else "QUALITY",
            configured_backend=self.llm_backend,
            active_backend=active_backend,
            active_model_label=active_model_label,
            loaded_model_id=self.loaded_model_id,
            loaded_model_label=self.loaded_model_label,
            embeddings_model="fake-embeddings",
            embeddings_device="cpu",
            device="cpu",
            document_name=self.current_document_name,
            ready_for_queries=True,
            processing_report=self.latest_processing_report,
        )

    def query_with_trace(self, message):
        active_backend = self.active_llm_backend or self.llm_backend
        active_model_label = (
            self.loaded_model_label
            or self.loaded_model_id
            or ("MockLLM (explicit demo)" if active_backend == "mock" else "unknown")
        )
        answer = "Project Phoenix is described in the uploaded document."
        return QueryResult(
            answer=answer,
            trace=AnswerTrace(
                question=message,
                document_name=self.current_document_name,
                backend=active_backend,
                model_label=active_model_label,
                retrieved_chunk_count=1,
                citations=[
                    AnswerCitation(
                        citation_id=1,
                        source_name=self.current_document_name or "demo.txt",
                        page=None,
                        chunk_index=0,
                        excerpt="Project Phoenix is a loop workbench.",
                    )
                ],
                self_check=AnswerSelfCheck(
                    outcome="not_verified",
                    reasons=[
                        "mechanical_checks_passed",
                        "verifier_unavailable_mock_backend",
                    ],
                    retry_attempted=False,
                ),
            ),
        )

    def clear_loop_session(self, session_id="default"):
        self.cleared_loop_session_id = session_id


class FakeEndpointQA(FakeQA):
    loaded_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    loaded_model_label = "Qwen/Qwen2.5-1.5B-Instruct"
    active_llm_backend = "endpoint"
    llm_backend = "endpoint"


class FakeCustomEndpointQA(FakeEndpointQA):
    loaded_model_id = None
    loaded_model_label = "Custom endpoint (https://example.invalid)"


class FakeOllamaQA(FakeQA):
    loaded_model_id = "nemotron-3-nano:4b"
    loaded_model_label = "Ollama (nemotron-3-nano:4b)"
    active_llm_backend = "ollama"
    llm_backend = "ollama"


class FakeOpenAICompatibleQA(FakeQA):
    loaded_model_id = "gpt-oss:20b"
    loaded_model_label = "OpenAI-compatible (gpt-oss:20b)"
    active_llm_backend = "openai-compatible"
    llm_backend = "openai-compatible"


def processed_report(
    *,
    document_name="good.txt",
    success=True,
    phase="complete",
    error_message=None,
):
    return DocumentProcessingReport(
        attempted_document_name=document_name,
        active_document_name=document_name,
        success=success,
        phase=phase,
        file_extension=".txt",
        chunk_count=1 if success else 0,
        truncated=False,
        max_chunk_limit=2000,
        text_encoding_mode="auto",
        backend="mock",
        model_label="MockLLM (explicit demo)",
        error_message=error_message,
    )


def test_process_document_reports_mock_mode_without_success_claim(monkeypatch, tmp_path):
    document = tmp_path / "demo.txt"
    document.write_text("demo", encoding="utf-8")
    fake_qa = FakeQA()
    monkeypatch.setattr(app, "qa_system", fake_qa)

    status, runtime_status = app.process_document(SimpleNamespace(name=str(document)))

    assert "processed in mock mode" in status
    assert "processed successfully" not in status
    assert "demonstration responses" in status
    assert "Chunks: `1`" in status
    runtime = json.loads(runtime_status)
    assert runtime["active_document"] == "demo.txt"
    assert runtime["last_attempted_document"] == "demo.txt"
    assert runtime["model_device"] == "cpu"
    assert runtime["embeddings_model"] == "fake-embeddings"
    assert runtime["embeddings_device"] == "cpu"
    assert runtime["ready_for_queries"] is True
    assert runtime["readiness_scope"] == "retrieval_pipeline"
    assert runtime["inference_validated"] is False
    assert runtime["last_success"] is True
    assert runtime["file_extension"] == ".txt"
    assert runtime["chunk_count"] == 1
    assert runtime["max_chunk_limit"] == 2000
    assert runtime["last_error"] is None
    assert fake_qa.text_encoding == "auto"


def test_app_bootstraps_native_defaults_before_gradio_import():
    env = os.environ.copy()
    for name in NATIVE_THREAD_ENV_VARS:
        env.pop(name, None)

    code = """
import builtins
import json
import os

real_import = builtins.__import__

def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "gradio" or name.startswith("gradio."):
        print(json.dumps({
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        }, sort_keys=True))
        raise SystemExit(0)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = tracking_import
import src.app
raise SystemExit("src.app did not import gradio")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(result.stdout) == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def test_text_encoding_dropdown_defaults_to_auto():
    assert app.text_encoding.value == "Auto"


def test_app_copy_frames_document_as_context():
    assert app.APP_TITLE == "AI Loop Engine"
    assert app.file_upload.label == "Upload Document Context"
    assert app.upload_button.value == "Index Context"
    assert app.upload_status.label == "Context Status"
    assert app.loop_summary.label == "Loop Summary"
    assert app.answer_trace.label == "Loop Trace"


def test_process_document_passes_selected_text_encoding(monkeypatch, tmp_path):
    document = tmp_path / "demo.txt"
    document.write_text("demo", encoding="utf-8")
    fake_qa = FakeQA()
    monkeypatch.setattr(app, "qa_system", fake_qa)

    status, _runtime_status = app.process_document(
        SimpleNamespace(name=str(document)), "Cyrillic (Windows-1251)"
    )

    assert "processed in mock mode" in status
    assert fake_qa.text_encoding == "cp1251"


def test_process_document_passes_utf8_or_western_only_when_selected(
    monkeypatch, tmp_path
):
    document = tmp_path / "demo.txt"
    document.write_text("demo", encoding="utf-8")
    fake_qa = FakeQA()
    monkeypatch.setattr(app, "qa_system", fake_qa)

    status, _runtime_status = app.process_document(
        SimpleNamespace(name=str(document)), "UTF-8 / Western"
    )

    assert "processed in mock mode" in status
    assert fake_qa.text_encoding == "utf-8-or-western"


def test_process_document_reports_endpoint_as_indexed_not_ready(monkeypatch, tmp_path):
    document = tmp_path / "demo.txt"
    document.write_text("demo", encoding="utf-8")
    monkeypatch.setattr(app, "qa_system", FakeEndpointQA())

    status, runtime_status = app.process_document(SimpleNamespace(name=str(document)))

    assert "indexed" in status
    assert "processed successfully" not in status
    assert "Inference will be validated on the first question" in status
    runtime = json.loads(runtime_status)
    assert runtime["backend"] == "endpoint"
    assert runtime["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert runtime["ready_for_queries"] is True
    assert runtime["readiness_scope"] == "retrieval_pipeline"
    assert runtime["inference_validated"] is False


def test_process_document_reports_ollama_as_indexed_not_mock(monkeypatch, tmp_path):
    document = tmp_path / "demo.txt"
    document.write_text("demo", encoding="utf-8")
    monkeypatch.setattr(app, "qa_system", FakeOllamaQA())

    status, runtime_status = app.process_document(SimpleNamespace(name=str(document)))

    assert "indexed" in status
    assert "processed in mock mode" not in status
    assert "Ollama (nemotron-3-nano:4b)" in status
    runtime = json.loads(runtime_status)
    assert runtime["backend"] == "ollama"
    assert runtime["model"] == "Ollama (nemotron-3-nano:4b)"
    assert runtime["ready_for_queries"] is True
    assert runtime["inference_validated"] is False


def test_process_document_reports_openai_compatible_as_indexed(
    monkeypatch, tmp_path
):
    document = tmp_path / "demo.txt"
    document.write_text("demo", encoding="utf-8")
    monkeypatch.setattr(app, "qa_system", FakeOpenAICompatibleQA())

    status, runtime_status = app.process_document(SimpleNamespace(name=str(document)))

    assert "indexed" in status
    assert "processed in mock mode" not in status
    assert "OpenAI-compatible (gpt-oss:20b)" in status
    runtime = json.loads(runtime_status)
    assert runtime["backend"] == "openai-compatible"
    assert runtime["model"] == "OpenAI-compatible (gpt-oss:20b)"
    assert runtime["ready_for_queries"] is True
    assert runtime["inference_validated"] is False


def test_process_document_uses_custom_endpoint_label(monkeypatch, tmp_path):
    document = tmp_path / "demo.txt"
    document.write_text("demo", encoding="utf-8")
    monkeypatch.setattr(app, "qa_system", FakeCustomEndpointQA())

    status, runtime_status = app.process_document(SimpleNamespace(name=str(document)))

    assert "Custom endpoint (https://example.invalid)" in status
    assert "Qwen" not in status
    assert (
        json.loads(runtime_status)["model"]
        == "Custom endpoint (https://example.invalid)"
    )


def test_process_document_reports_failure_without_losing_active_status(
    monkeypatch, tmp_path
):
    document = tmp_path / "bad.txt"
    document.write_text("bad", encoding="utf-8")
    fake_qa = FakeQA()
    fake_qa.current_document_name = "good.txt"
    fake_qa.latest_processing_report = DocumentProcessingReport(
        attempted_document_name="good.txt",
        active_document_name="good.txt",
        success=True,
        phase="complete",
        file_extension=".txt",
        chunk_count=1,
        truncated=False,
        max_chunk_limit=2000,
        text_encoding_mode="auto",
        backend="mock",
        model_label="MockLLM (explicit demo)",
        error_message=None,
    )

    def fail_process_document(document_path, text_encoding=None):
        fake_qa.text_encoding = text_encoding
        fake_qa.latest_processing_report = DocumentProcessingReport(
            attempted_document_name="bad.txt",
            active_document_name="good.txt",
            success=False,
            phase="load",
            file_extension=".txt",
            chunk_count=0,
            truncated=False,
            max_chunk_limit=2000,
            text_encoding_mode=text_encoding or "auto",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            error_message="Could not decode text document",
        )
        raise DocumentProcessingError(
            "Error loading document: Could not decode text document",
            fake_qa.status(),
        )

    fake_qa.process_document = fail_process_document
    monkeypatch.setattr(app, "qa_system", fake_qa)

    status, runtime_status = app.process_document(SimpleNamespace(name=str(document)))

    assert "Document context `bad.txt` failed during `load`" in status
    assert "Active document remains `good.txt`" in status
    assert "Could not decode text document" in status
    runtime = json.loads(runtime_status)
    assert runtime["active_document"] == "good.txt"
    assert runtime["last_attempted_document"] == "bad.txt"
    assert runtime["last_success"] is False
    assert runtime["file_extension"] == ".txt"
    assert runtime["last_error"] == "Could not decode text document"


def test_process_document_unexpected_error_uses_pre_upload_status(
    monkeypatch, tmp_path
):
    document = tmp_path / "bad.txt"
    document.write_text("bad", encoding="utf-8")

    class UnexpectedFailureQA(FakeQA):
        def __init__(self):
            self.current_document_name = "good.txt"
            self.latest_processing_report = processed_report()
            self.status_calls = 0

        def process_document(self, document_path, text_encoding=None):
            self.current_document_name = "mutated.txt"
            self.latest_processing_report = processed_report(
                document_name="mutated.txt"
            )
            raise RuntimeError("unexpected boom")

        def status(self):
            self.status_calls += 1
            return super().status()

    fake_qa = UnexpectedFailureQA()
    monkeypatch.setattr(app, "qa_system", fake_qa)

    status, runtime_status = app.process_document(SimpleNamespace(name=str(document)))

    assert fake_qa.status_calls == 1
    assert "Document context `bad.txt` failed during `unexpected`" in status
    assert "Active document remains `good.txt`" in status
    assert "unexpected boom" in status
    runtime = json.loads(runtime_status)
    assert runtime["active_document"] == "good.txt"
    assert runtime["last_attempted_document"] == "bad.txt"
    assert runtime["phase"] == "unexpected"
    assert runtime["last_success"] is False
    assert runtime["last_error"] == "unexpected boom"


def test_format_answer_trace_includes_citations():
    result = QueryResult(
        answer="Project Phoenix launches in June 2026 [1].",
        trace=AnswerTrace(
            question="When does it launch?",
            document_name="phoenix.txt",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            retrieved_chunk_count=1,
            citations=[
                AnswerCitation(
                    citation_id=1,
                    source_name="phoenix.txt",
                    page=None,
                    chunk_index=0,
                    excerpt="The launch date is June 2026.",
                )
            ],
            self_check=AnswerSelfCheck(
                outcome="not_verified",
                reasons=[
                    "mechanical_checks_passed",
                    "verifier_unavailable_mock_backend",
                ],
                retry_attempted=True,
            ),
        ),
    )

    trace = json.loads(app.format_answer_trace(result))

    assert trace["question"] == "When does it launch?"
    assert trace["answer"] == "Project Phoenix launches in June 2026 [1]."
    assert trace["document"] == "phoenix.txt"
    assert trace["retrieved_chunk_count"] == 1
    assert trace["citations"][0]["id"] == 1
    assert trace["citations"][0]["source"] == "phoenix.txt"
    assert trace["citations"][0]["chunk"] == 1
    assert "June 2026" in trace["citations"][0]["excerpt"]
    assert trace["self_check"]["outcome"] == "not_verified"
    assert trace["self_check"]["reasons"] == [
        "mechanical_checks_passed",
        "verifier_unavailable_mock_backend",
    ]
    assert trace["self_check"]["retry_attempted"] is True
    assert trace["loop_report"] is None
    assert trace["error"] is None


def test_format_answer_trace_includes_loop_report():
    loop_report = LoopReport(
        run=LoopRun(
            run_id="run_ui",
            user_input="When does it launch?",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            final_decision=LoopDecision.NOT_VERIFIED,
            final_answer="Project Phoenix launches in June 2026 [1].",
        )
    )
    result = QueryResult(
        answer="Project Phoenix launches in June 2026 [1].",
        trace=AnswerTrace(
            question="When does it launch?",
            document_name="phoenix.txt",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            retrieved_chunk_count=0,
            citations=[],
        ),
        loop_report=loop_report,
    )

    trace = json.loads(app.format_answer_trace(result))

    assert trace["loop_report"]["schema_version"] == "loop-report/v1"
    assert trace["loop_report"]["run"]["run_id"] == "run_ui"
    assert trace["loop_report"]["run"]["final_decision"] == "not_verified"


def test_format_loop_summary_empty_state():
    summary = json.loads(app.format_loop_summary(None))

    assert summary == {
        "context_provider": None,
        "document": None,
        "backend": None,
        "model": None,
        "retrieved_chunk_count": 0,
        "draft_attempt_count": 0,
        "mechanical_check": None,
        "verifier": None,
        "retry_attempted": False,
        "refused": False,
        "final_decision": None,
        "last_error": None,
    }


def test_format_loop_summary_shows_compact_loop_fields():
    loop_report = LoopReport(
        run=LoopRun(
            run_id="run_summary",
            user_input="When does it launch?",
            context_provider="document",
            backend="ollama",
            model_label="Ollama (nemotron-3-nano:4b)",
            steps=(
                LoopStep(
                    phase=LoopPhase.RETRIEVE,
                    decision=LoopDecision.CONTINUE,
                    metadata={"retrieved_chunk_count": 1},
                ),
                LoopStep(
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                ),
                LoopStep(
                    phase=LoopPhase.RETRY,
                    decision=LoopDecision.RETRY,
                    metadata={"reasons": ["missing_inline_citation"]},
                ),
                LoopStep(
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    retry_count=1,
                ),
                LoopStep(
                    phase=LoopPhase.MECHANICAL_CHECK,
                    decision=LoopDecision.CONTINUE,
                    output_summary="mechanical_checks_passed",
                    retry_count=1,
                ),
                LoopStep(
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.SUPPORTED,
                    output_summary="supported",
                    retry_count=1,
                    verification=VerificationResult(
                        outcome=VerificationOutcome.SUPPORTED,
                        reasons=(
                            "mechanical_checks_passed",
                            "llm_verifier_supported",
                        ),
                        verifier="ollama",
                    ),
                ),
                LoopStep(
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.SUPPORTED,
                ),
            ),
            final_decision=LoopDecision.SUPPORTED,
            final_answer="Project Phoenix launches in June 2026 [1].",
        )
    )
    result = QueryResult(
        answer="Project Phoenix launches in June 2026 [1].",
        trace=AnswerTrace(
            question="When does it launch?",
            document_name="phoenix.txt",
            backend="ollama",
            model_label="Ollama (nemotron-3-nano:4b)",
            retrieved_chunk_count=1,
            citations=[],
            self_check=AnswerSelfCheck(
                outcome="supported",
                reasons=["mechanical_checks_passed", "llm_verifier_supported"],
                retry_attempted=True,
            ),
        ),
        loop_report=loop_report,
    )

    summary = json.loads(app.format_loop_summary(result))

    assert summary["context_provider"] == "document"
    assert summary["document"] == "phoenix.txt"
    assert summary["backend"] == "ollama"
    assert summary["retrieved_chunk_count"] == 1
    assert summary["draft_attempt_count"] == 2
    assert summary["mechanical_check"] == "mechanical_checks_passed"
    assert summary["verifier"] == {
        "decision": "supported",
        "outcome": "supported",
        "reasons": ["mechanical_checks_passed", "llm_verifier_supported"],
    }
    assert summary["retry_attempted"] is True
    assert summary["refused"] is False
    assert summary["final_decision"] == "supported"
    assert summary["last_error"] is None


def test_format_answer_trace_redacts_guardrail_blocked_draft():
    blocked_draft = "Sensitive blocked draft should not be public."
    loop_report = LoopReport(
        run=LoopRun(
            run_id="run_blocked",
            user_input="Generate unsafe content",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            steps=(
                LoopStep(
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    name="Draft answer",
                    input_summary="Generate unsafe content",
                    output_summary=blocked_draft,
                    metadata={"draft_preview": blocked_draft},
                ),
                LoopStep(
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.BLOCK,
                    name="Guardrail decision",
                    output_summary=blocked_draft,
                    error_message=blocked_draft,
                    metadata={
                        "guardrail_decision": "block",
                        "guardrail_reason": blocked_draft,
                    },
                ),
                LoopStep(
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.BLOCK,
                    name="Final answer",
                    output_summary="block",
                ),
            ),
            final_decision=LoopDecision.BLOCK,
            final_answer="A loop guardrail blocked this query before it could complete.",
            error_message=blocked_draft,
            metadata={"guardrail_detail": blocked_draft},
        )
    )
    result = QueryResult(
        answer="A loop guardrail blocked this query before it could complete.",
        trace=AnswerTrace(
            question="Generate unsafe content",
            document_name="phoenix.txt",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            retrieved_chunk_count=0,
            citations=[],
            error_message=blocked_draft,
        ),
        loop_report=loop_report,
    )

    assert blocked_draft in json.dumps(loop_report.to_dict())

    trace_json = app.format_answer_trace(result)
    summary_json = app.format_loop_summary(result)
    trace = json.loads(trace_json)
    summary = json.loads(summary_json)
    redacted_step = trace["loop_report"]["run"]["steps"][0]
    guardrail_step = trace["loop_report"]["run"]["steps"][1]

    assert blocked_draft not in trace_json
    assert blocked_draft not in summary_json
    assert summary["last_error"] == "terminal_guardrail_decision"
    assert summary["final_decision"] == "block"
    assert trace["error"] == "terminal_guardrail_decision"
    assert trace["loop_report"]["run"]["error_message"] == (
        "terminal_guardrail_decision"
    )
    assert trace["loop_report"]["public_redaction"]["applied"] is True
    assert redacted_step["output_summary"] == (
        "[redacted: terminal guardrail decision]"
    )
    assert redacted_step["metadata"]["redacted"] is True
    assert guardrail_step["output_summary"] == (
        "[redacted: terminal guardrail decision]"
    )
    assert guardrail_step["error_message"] == "terminal_guardrail_decision"
    assert guardrail_step["metadata"]["redacted"] is True


def test_chat_returns_answer_and_trace(monkeypatch):
    fake_qa = FakeQA()
    fake_qa.current_document_name = "demo.txt"
    monkeypatch.setattr(app, "qa_system", fake_qa)

    history, message, loop_summary, answer_trace = app.chat(
        "What is Project Phoenix?", []
    )

    assert message == ""
    assert history[-1]["role"] == "assistant"
    assert "Project Phoenix" in history[-1]["content"]
    summary = json.loads(loop_summary)
    assert summary["document"] == "demo.txt"
    assert summary["retrieved_chunk_count"] == 1
    assert summary["draft_attempt_count"] == 0
    assert summary["final_decision"] is None
    trace = json.loads(answer_trace)
    assert trace["question"] == "What is Project Phoenix?"
    assert trace["document"] == "demo.txt"
    assert trace["retrieved_chunk_count"] == 1
    assert trace["citations"][0]["source"] == "demo.txt"
    assert trace["self_check"]["outcome"] == "not_verified"


def test_clear_chat_resets_answer_trace(monkeypatch):
    fake_qa = FakeQA()
    fake_qa.chat_history = [{"question": "old"}]
    monkeypatch.setattr(app, "qa_system", fake_qa)

    history, loop_summary, answer_trace = app.clear_chat()

    assert history == []
    assert fake_qa.chat_history == []
    assert fake_qa.cleared_loop_session_id == "default"
    assert json.loads(loop_summary)["draft_attempt_count"] == 0
    assert json.loads(answer_trace)["citations"] == []
