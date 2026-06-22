import pytest
from langchain_core.embeddings import Embeddings

from src.DocumentQA import DocumentQA, MockLLM


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        lower = text.lower()
        return [
            float(len(text)),
            float(lower.count("project")),
            float(lower.count("phoenix")),
            float(lower.count("launch")),
        ]


def test_query_before_document_asks_for_upload_first():
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    assert qa.query("What is this?") == "Please upload and process a document first."


def test_process_text_document_with_mock_llm_and_fake_embeddings(tmp_path):
    document = tmp_path / "phoenix.txt"
    document.write_text(
        "Project Phoenix is a document QA assistant. "
        "The launch date is June 2026.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")
    qa.embeddings = FakeEmbeddings()

    qa.process_document(str(document))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is not None
    assert qa.retrieval_chain is not None
    assert qa.active_llm_backend == "mock"
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."
    assert "demonstration response" in qa.query("What is Project Phoenix?")


def test_rejects_unsupported_file_type_before_model_initialization(tmp_path):
    document = tmp_path / "data.csv"
    document.write_text("a,b\n1,2\n", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    with pytest.raises(RuntimeError, match="Unsupported file type: .csv"):
        qa.process_document(str(document))

    assert qa.llm is None


def test_rejects_oversized_document_before_model_initialization(tmp_path):
    document = tmp_path / "too-large.txt"
    document.write_text("too large", encoding="utf-8")
    qa = DocumentQA(
        fast_mode=True, hf_token="dummy", llm_backend="mock", max_document_bytes=1
    )

    with pytest.raises(RuntimeError, match="File is too large"):
        qa.process_document(str(document))

    assert qa.llm is None


def test_auto_backend_uses_endpoint_on_cpu_with_token(monkeypatch):
    def fake_endpoint_loader(self, model_id):
        return MockLLM()

    monkeypatch.delenv("HF_ENDPOINT_URL", raising=False)
    monkeypatch.setattr(DocumentQA, "_load_endpoint_model", fake_endpoint_loader)
    qa = DocumentQA(device="cpu", hf_token="real-token", llm_backend="auto")

    qa._initialize_llm()

    assert qa.active_llm_backend == "endpoint"
    assert qa.loaded_model_id == "Qwen/Qwen2.5-1.5B-Instruct"


def test_endpoint_url_does_not_also_set_repo_id(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT_URL", "https://example.invalid")
    qa = DocumentQA(device="cpu", hf_token="real-token", llm_backend="endpoint")

    llm = qa._load_endpoint_model("Qwen/Qwen2.5-1.5B-Instruct")

    assert llm.endpoint_url == "https://example.invalid"
    assert llm.repo_id is None


def test_custom_endpoint_records_endpoint_label_not_candidate_model(monkeypatch):
    def fake_endpoint_loader(self, model_id):
        return MockLLM()

    monkeypatch.setenv("HF_ENDPOINT_URL", "https://example.invalid")
    monkeypatch.setattr(DocumentQA, "_load_endpoint_model", fake_endpoint_loader)
    qa = DocumentQA(device="cpu", hf_token="real-token", llm_backend="endpoint")

    qa._initialize_llm()

    assert qa.loaded_model_id is None
    assert qa.loaded_model_label == "Custom endpoint (https://example.invalid)"


def test_auto_backend_uses_local_on_accelerated_device_with_token(monkeypatch):
    def fake_local_loader(self, model_id):
        return MockLLM()

    monkeypatch.setattr(DocumentQA, "_load_local_model", fake_local_loader)
    qa = DocumentQA(device="mps", hf_token="real-token", llm_backend="auto")

    qa._initialize_llm()

    assert qa.active_llm_backend == "local"
    assert qa.loaded_model_id == "Qwen/Qwen2.5-1.5B-Instruct"


def test_auto_endpoint_without_token_falls_back_to_mock(monkeypatch):
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    qa = DocumentQA(device="cpu", llm_backend="auto", hf_token=None)

    qa._initialize_llm()

    assert qa.active_llm_backend == "mock"
    assert isinstance(qa.llm, MockLLM)


def test_explicit_endpoint_without_token_fails_closed(monkeypatch):
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    qa = DocumentQA(device="cpu", llm_backend="endpoint", hf_token=None)

    with pytest.raises(RuntimeError, match="token is required for endpoint inference"):
        qa._initialize_llm()


def test_explicit_local_backend_failure_fails_closed(monkeypatch):
    def failing_local_loader(self, model_id):
        raise ValueError("boom")

    monkeypatch.setattr(DocumentQA, "_load_local_model", failing_local_loader)
    qa = DocumentQA(device="cpu", llm_backend="local", hf_token="real-token")

    with pytest.raises(RuntimeError, match="Unable to initialize local LLM"):
        qa._initialize_llm()

    assert qa.active_llm_backend is None
    assert qa.llm is None


@pytest.mark.parametrize("backend", ["endpoint", "local"])
def test_explicit_real_backend_with_dummy_token_fails_closed(backend):
    qa = DocumentQA(device="cpu", llm_backend=backend, hf_token="dummy")

    with pytest.raises(RuntimeError, match="Dummy HuggingFace token"):
        qa._initialize_llm()

    assert qa.active_llm_backend is None
    assert qa.llm is None


def test_local_backend_without_token_still_attempts_local_public_model(monkeypatch):
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)

    def fake_local_loader(self, model_id):
        return MockLLM()

    monkeypatch.setattr(DocumentQA, "_load_local_model", fake_local_loader)
    qa = DocumentQA(device="cpu", llm_backend="local", hf_token=None)

    qa._initialize_llm()

    assert qa.active_llm_backend == "local"
    assert qa.loaded_model_id == "Qwen/Qwen2.5-1.5B-Instruct"
