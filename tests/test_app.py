import json
from types import SimpleNamespace

from src import app
from src.DocumentQA import (
    AnswerCitation,
    AnswerTrace,
    DocumentProcessingError,
    DocumentProcessingReport,
    DocumentQAStatus,
    QueryResult,
)


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
            or ("MockLLM (fallback)" if active_backend == "mock" else "unknown")
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
            or ("MockLLM (fallback)" if active_backend == "mock" else "unknown")
        )
        return DocumentQAStatus(
            profile_label="FAST" if self.fast_mode else "QUALITY",
            configured_backend=self.llm_backend,
            active_backend=active_backend,
            active_model_label=active_model_label,
            loaded_model_id=self.loaded_model_id,
            loaded_model_label=self.loaded_model_label,
            embeddings_model="fake-embeddings",
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
            or ("MockLLM (fallback)" if active_backend == "mock" else "unknown")
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
                        excerpt="Project Phoenix is a document QA assistant.",
                    )
                ],
            ),
        )


class FakeEndpointQA(FakeQA):
    loaded_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    loaded_model_label = "Qwen/Qwen2.5-1.5B-Instruct"
    active_llm_backend = "endpoint"
    llm_backend = "endpoint"


class FakeCustomEndpointQA(FakeEndpointQA):
    loaded_model_id = None
    loaded_model_label = "Custom endpoint (https://example.invalid)"


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
        model_label="MockLLM (fallback)",
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
    assert runtime["ready_for_queries"] is True
    assert runtime["readiness_scope"] == "retrieval_pipeline"
    assert runtime["inference_validated"] is False
    assert runtime["last_success"] is True
    assert runtime["file_extension"] == ".txt"
    assert runtime["chunk_count"] == 1
    assert runtime["max_chunk_limit"] == 2000
    assert runtime["last_error"] is None
    assert fake_qa.text_encoding == "auto"


def test_text_encoding_dropdown_defaults_to_auto():
    assert app.text_encoding.value == "Auto"


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
        model_label="MockLLM (fallback)",
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
            model_label="MockLLM (fallback)",
            error_message="Could not decode text document",
        )
        raise DocumentProcessingError(
            "Error loading document: Could not decode text document",
            fake_qa.status(),
        )

    fake_qa.process_document = fail_process_document
    monkeypatch.setattr(app, "qa_system", fake_qa)

    status, runtime_status = app.process_document(SimpleNamespace(name=str(document)))

    assert "failed during `load`" in status
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
    assert "failed during `unexpected`" in status
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
            model_label="MockLLM (fallback)",
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
    assert trace["error"] is None


def test_chat_returns_answer_and_trace(monkeypatch):
    fake_qa = FakeQA()
    fake_qa.current_document_name = "demo.txt"
    monkeypatch.setattr(app, "qa_system", fake_qa)

    history, message, answer_trace = app.chat("What is Project Phoenix?", [])

    assert message == ""
    assert history[-1]["role"] == "assistant"
    assert "Project Phoenix" in history[-1]["content"]
    trace = json.loads(answer_trace)
    assert trace["question"] == "What is Project Phoenix?"
    assert trace["document"] == "demo.txt"
    assert trace["retrieved_chunk_count"] == 1
    assert trace["citations"][0]["source"] == "demo.txt"


def test_clear_chat_resets_answer_trace(monkeypatch):
    fake_qa = FakeQA()
    fake_qa.chat_history = [{"question": "old"}]
    monkeypatch.setattr(app, "qa_system", fake_qa)

    history, answer_trace = app.clear_chat()

    assert history == []
    assert fake_qa.chat_history == []
    assert json.loads(answer_trace)["citations"] == []
