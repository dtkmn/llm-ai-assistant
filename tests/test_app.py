from types import SimpleNamespace

from src import app


class FakeQA:
    fast_mode = True
    loaded_model_id = None
    active_llm_backend = "mock"
    llm_backend = "mock"

    def process_document(self, document_path):
        self.document_path = document_path


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
