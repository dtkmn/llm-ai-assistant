from types import SimpleNamespace

from src import app
from src.DocumentQA import DocumentQAStatus


class FakeQA:
    fast_mode = True
    loaded_model_id = None
    loaded_model_label = None
    active_llm_backend = "mock"
    llm_backend = "mock"

    def process_document(self, document_path, text_encoding=None):
        self.document_path = document_path
        self.text_encoding = text_encoding

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
            document_name=getattr(self, "document_path", None),
            ready_for_queries=True,
        )


class FakeEndpointQA(FakeQA):
    loaded_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    loaded_model_label = "Qwen/Qwen2.5-1.5B-Instruct"
    active_llm_backend = "endpoint"
    llm_backend = "endpoint"


class FakeCustomEndpointQA(FakeEndpointQA):
    loaded_model_id = None
    loaded_model_label = "Custom endpoint (https://example.invalid)"


def test_process_document_reports_mock_mode_without_success_claim(monkeypatch, tmp_path):
    document = tmp_path / "demo.txt"
    document.write_text("demo", encoding="utf-8")
    fake_qa = FakeQA()
    monkeypatch.setattr(app, "qa_system", fake_qa)

    status = app.process_document(SimpleNamespace(name=str(document)))

    assert "processed in mock mode" in status
    assert "processed successfully" not in status
    assert "demonstration responses" in status
    assert fake_qa.text_encoding == "auto"


def test_text_encoding_dropdown_defaults_to_auto():
    assert app.text_encoding.value == "Auto"


def test_process_document_passes_selected_text_encoding(monkeypatch, tmp_path):
    document = tmp_path / "demo.txt"
    document.write_text("demo", encoding="utf-8")
    fake_qa = FakeQA()
    monkeypatch.setattr(app, "qa_system", fake_qa)

    status = app.process_document(
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

    status = app.process_document(
        SimpleNamespace(name=str(document)), "UTF-8 / Western"
    )

    assert "processed in mock mode" in status
    assert fake_qa.text_encoding == "utf-8-or-western"


def test_process_document_reports_endpoint_as_indexed_not_ready(monkeypatch, tmp_path):
    document = tmp_path / "demo.txt"
    document.write_text("demo", encoding="utf-8")
    monkeypatch.setattr(app, "qa_system", FakeEndpointQA())

    status = app.process_document(SimpleNamespace(name=str(document)))

    assert "indexed" in status
    assert "processed successfully" not in status
    assert "Inference will be validated on the first question" in status


def test_process_document_uses_custom_endpoint_label(monkeypatch, tmp_path):
    document = tmp_path / "demo.txt"
    document.write_text("demo", encoding="utf-8")
    monkeypatch.setattr(app, "qa_system", FakeCustomEndpointQA())

    status = app.process_document(SimpleNamespace(name=str(document)))

    assert "Custom endpoint (https://example.invalid)" in status
    assert "Qwen" not in status
