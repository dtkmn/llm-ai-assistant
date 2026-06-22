import threading

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from src.DocumentQA import DocumentQA, FaissVectorStore, MockLLM


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


def create_processed_mock_qa(tmp_path):
    document = tmp_path / "phoenix.txt"
    document.write_text(
        "Project Phoenix is a document QA assistant. "
        "The launch date is June 2026.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")
    qa.embeddings = FakeEmbeddings()
    qa.process_document(str(document))
    return qa, document


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

    status = qa.status()
    assert status.profile_label == "FAST"
    assert status.active_backend == "mock"
    assert status.active_model_label == "MockLLM (fallback)"
    assert status.document_name == "phoenix.txt"
    assert status.ready_for_queries is True
    assert status.mock_mode is True
    assert status.processing_report is not None
    assert status.processing_report.attempted_document_name == "phoenix.txt"
    assert status.processing_report.active_document_name == "phoenix.txt"
    assert status.processing_report.success is True
    assert status.processing_report.phase == "complete"
    assert status.processing_report.file_extension == ".txt"
    assert status.processing_report.chunk_count > 0
    assert status.processing_report.truncated is False
    assert status.processing_report.max_chunk_limit == qa.max_document_chunks
    assert status.processing_report.text_encoding_mode == "auto"
    assert status.processing_report.backend == "mock"
    assert status.processing_report.model_label == "MockLLM (fallback)"
    assert status.processing_report.error_message is None


def test_status_reports_configured_backend_before_initialization():
    qa = DocumentQA(
        device="cpu",
        fast_mode=False,
        hf_token="real-token",
        llm_backend="endpoint",
        model_id="example/model",
    )

    status = qa.status()

    assert status.profile_label == "QUALITY"
    assert status.configured_backend == "endpoint"
    assert status.active_backend == "endpoint"
    assert status.active_model_label == "example/model"
    assert status.ready_for_queries is False
    assert status.mock_mode is False
    assert status.processing_report is None


def test_failed_unsupported_replacement_keeps_previous_document_queryable(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    replacement = tmp_path / "replacement.csv"
    replacement.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unsupported file type: .csv"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "validate"
    assert report.attempted_document_name == "replacement.csv"
    assert report.active_document_name == "phoenix.txt"
    assert report.file_extension == ".csv"
    assert report.chunk_count == 0
    assert "Unsupported file type: .csv" in report.error_message


def test_query_uses_active_state_snapshot_not_legacy_mixed_state(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)

    qa.current_document_name = "replacement.txt"
    qa.vector_store = None
    qa.retrieval_chain = None

    assert qa.status().document_name == "phoenix.txt"
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."


def test_concurrent_replacement_uploads_are_serialized(monkeypatch, tmp_path):
    first = tmp_path / "first.txt"
    first.write_text("First document for Project Phoenix.", encoding="utf-8")
    second = tmp_path / "second.txt"
    second.write_text("Second document for Project Phoenix.", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")
    qa.embeddings = FakeEmbeddings()
    original_process_document = DocumentQA.process_document
    original_load_documents = DocumentQA._load_documents
    second_upload_attempted = threading.Event()
    first_load_started = threading.Event()
    second_load_started = threading.Event()
    release_first_load = threading.Event()
    errors = []

    def tracked_process_document(self, document_path, text_encoding=None):
        if document_path == str(second):
            second_upload_attempted.set()
        return original_process_document(
            self, document_path, text_encoding=text_encoding
        )

    def delayed_first_load(self, document_path, file_extension, text_encoding=None):
        if document_path == str(first):
            first_load_started.set()
            assert release_first_load.wait(timeout=5)
        elif document_path == str(second):
            second_load_started.set()
        return original_load_documents(
            self, document_path, file_extension, text_encoding=text_encoding
        )

    def process(path):
        try:
            qa.process_document(str(path))
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(DocumentQA, "process_document", tracked_process_document)
    monkeypatch.setattr(DocumentQA, "_load_documents", delayed_first_load)

    first_thread = threading.Thread(target=process, args=(first,))
    first_thread.start()
    assert first_load_started.wait(timeout=5)

    second_thread = threading.Thread(target=process, args=(second,))
    second_thread.start()
    assert second_upload_attempted.wait(timeout=5)
    assert not second_load_started.wait(timeout=0.2)
    release_first_load.set()

    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert second_load_started.is_set()
    assert errors == []
    assert qa.status().document_name == "second.txt"
    assert qa.query("Which file did I upload?") == "The uploaded document is `second.txt`."


def test_failed_ambiguous_text_replacement_keeps_previous_document_queryable(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    replacement = tmp_path / "ambiguous.txt"
    replacement.write_bytes("Привет Phoenix".encode("cp1251"))

    with pytest.raises(RuntimeError, match="Could not decode text document"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "load"
    assert report.attempted_document_name == "ambiguous.txt"
    assert report.active_document_name == "phoenix.txt"
    assert report.file_extension == ".txt"
    assert report.text_encoding_mode == "auto"
    assert "Could not decode text document" in report.error_message


def test_failed_embedding_initialization_keeps_previous_document_queryable(
    monkeypatch, tmp_path
):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    qa.embeddings = None
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("Replacement document should not commit.", encoding="utf-8")

    def fail_initialize_embeddings(self):
        self.embeddings_error = "boom"
        self.embeddings = None

    monkeypatch.setattr(DocumentQA, "_initialize_embeddings", fail_initialize_embeddings)

    with pytest.raises(RuntimeError, match="Embedding model is unavailable"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "initialize_embeddings"
    assert report.attempted_document_name == "replacement.txt"
    assert report.active_document_name == "phoenix.txt"
    assert "Embedding model is unavailable" in report.error_message


def test_empty_replacement_keeps_previous_document_queryable(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    replacement = tmp_path / "empty.txt"
    replacement.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="No text chunks were generated"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "split"
    assert report.attempted_document_name == "empty.txt"
    assert report.active_document_name == "phoenix.txt"
    assert report.chunk_count == 0
    assert "No text chunks were generated" in report.error_message


def test_index_failure_keeps_previous_document_queryable(monkeypatch, tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("Replacement document should not commit.", encoding="utf-8")

    def fail_from_documents(documents, embedding):
        raise ValueError("index boom")

    monkeypatch.setattr(FaissVectorStore, "from_documents", fail_from_documents)

    with pytest.raises(RuntimeError, match="index boom"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "index"
    assert report.attempted_document_name == "replacement.txt"
    assert report.active_document_name == "phoenix.txt"
    assert report.chunk_count > 0
    assert "index boom" in report.error_message


def test_chain_failure_keeps_previous_document_queryable(monkeypatch, tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("Replacement document should not commit.", encoding="utf-8")

    def fail_build_retrieval_chain(self, vector_store, document_name):
        raise RuntimeError("chain boom")

    monkeypatch.setattr(
        DocumentQA, "_build_retrieval_chain", fail_build_retrieval_chain
    )

    with pytest.raises(RuntimeError, match="chain boom"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "chain"
    assert report.attempted_document_name == "replacement.txt"
    assert report.active_document_name == "phoenix.txt"
    assert report.chunk_count > 0
    assert "chain boom" in report.error_message


def test_successful_upload_report_records_truncation(tmp_path):
    document = tmp_path / "long.txt"
    document.write_text(("Project Phoenix launch notes.\n" * 500), encoding="utf-8")
    qa = DocumentQA(
        fast_mode=True,
        hf_token="dummy",
        llm_backend="mock",
        max_document_chunks=1,
    )
    qa.embeddings = FakeEmbeddings()

    qa.process_document(str(document))

    report = qa.status().processing_report
    assert report.success is True
    assert report.phase == "complete"
    assert report.chunk_count == 1
    assert report.truncated is True
    assert report.max_chunk_limit == 1


def test_text_loader_sets_source_metadata(tmp_path):
    document = tmp_path / "notes.md"
    document.write_text("# Notes\nProject Phoenix launch notes.", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    loaded_documents = qa._load_documents(str(document), ".md")

    assert len(loaded_documents) == 1
    assert loaded_documents[0].page_content.startswith("# Notes")
    assert loaded_documents[0].metadata["source"] == str(document)


@pytest.mark.parametrize(
    ("encoding", "text"),
    [
        ("latin-1", "Café"),
        ("latin-1", "Crème brûlée"),
        ("latin-1", "Voilà"),
        ("latin-1", "Résumé"),
        ("latin-1", "naïve façade"),
        ("latin-1", "piñata"),
        ("latin-1", "paño Phoenix"),
        ("latin-1", "mañana Phoenix"),
        ("latin-1", "Málaga"),
        ("latin-1", "Córdoba"),
        ("latin-1", "García"),
        ("latin-1", "María"),
        ("latin-1", "Hélène"),
        ("latin-1", "Québec"),
        ("latin-1", "Montréal"),
        ("latin-1", "Montréal Québec"),
        ("latin-1", "München"),
        ("latin-1", "Zürich"),
        ("latin-1", "Düsseldorf"),
        ("latin-1", "Göteborg"),
        ("latin-1", "Köln"),
        ("latin-1", "Jürgen"),
        ("latin-1", "Bücher"),
        ("latin-1", "Søren"),
        ("latin-1", "København"),
        ("latin-1", "Å"),
        ("latin-1", "Ångström"),
        ("latin-1", "Tromsø"),
        ("latin-1", "smørrebrød"),
        ("latin-1", "A Coruña"),
        ("latin-1", "açaí"),
        ("latin-1", "Ação"),
        ("latin-1", "coração"),
        ("latin-1", "João"),
        ("latin-1", "mãe"),
        ("latin-1", "smörgåsbord"),
        ("latin-1", "El Niño"),
        ("latin-1", "São Paulo"),
        ("latin-1", "François"),
        ("latin-1", "über"),
        ("latin-1", "garçon"),
        ("latin-1", "Area 10 m²"),
        ("latin-1", "Volume 5 cm³"),
        ("latin-1", "Temp 20 °C"),
        ("latin-1", "Price £10"),
        ("latin-1", "Half ½ cup"),
        ("latin-1", "Café Phoenix résumé"),
        ("cp1252", "Café Phoenix — résumé"),
        ("cp1252", "Price €10 — Phoenix"),
        ("cp1252", "¿Cómo estás? Phoenix"),
        ("cp1252", "Trademark ™ Phoenix"),
    ],
)
def test_text_loader_detects_common_non_utf8_encodings(tmp_path, encoding, text):
    document = tmp_path / "legacy.txt"
    document.write_bytes(text.encode(encoding))
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    loaded_documents = qa._load_documents(
        str(document), ".txt", text_encoding=encoding
    )

    assert loaded_documents[0].page_content == text
    assert "\ufffd" not in loaded_documents[0].page_content


@pytest.mark.parametrize(
    ("encoding", "text"),
    [
        ("utf-8", "Café Phoenix"),
        ("utf-8", "Résumé Phoenix"),
        ("utf-8", "München Phoenix"),
        ("utf-8", "Price €10 — Phoenix"),
        ("utf-8-sig", "Café Phoenix"),
        ("latin-1", "Göteborg Phoenix"),
        ("cp1252", "Price €10 — Phoenix"),
    ],
)
def test_text_loader_utf8_or_western_mode_preserves_text(tmp_path, encoding, text):
    document = tmp_path / "default_text.txt"
    document.write_bytes(text.encode(encoding))
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    loaded_documents = qa._load_documents(
        str(document), ".txt", text_encoding="utf-8-or-western"
    )

    assert loaded_documents[0].page_content == text
    assert "Ã" not in loaded_documents[0].page_content
    assert "â" not in loaded_documents[0].page_content


@pytest.mark.parametrize(
    ("encoding", "text"),
    [
        ("utf-16-le", "Project Phoenix"),
        ("utf-16-be", "Project Phoenix"),
        ("utf-32-le", "Project Phoenix"),
        ("utf-32-be", "Project Phoenix"),
        ("utf-32", "Project Phoenix"),
    ],
)
def test_text_loader_detects_utf_family_encodings(tmp_path, encoding, text):
    document = tmp_path / "unicode.txt"
    document.write_bytes(text.encode(encoding))
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    loaded_documents = qa._load_documents(str(document), ".txt")

    assert loaded_documents[0].page_content == text
    assert "\x00" not in loaded_documents[0].page_content


def test_text_loader_uses_confident_detector_for_legacy_encoding(tmp_path):
    text = "Zażółć gęślą jaźń"
    document = tmp_path / "polish.txt"
    document.write_bytes(text.encode("cp1250"))
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    loaded_documents = qa._load_documents(str(document), ".txt")

    assert loaded_documents[0].page_content == text


@pytest.mark.parametrize(
    ("encoding", "text"),
    [
        ("cp1250", "Dvořák Phoenix"),
        ("cp1250", "město Phoenix"),
        ("cp1251", "Привет Phoenix"),
        ("cp1254", "İstanbul Phoenix"),
        ("cp1257", "māja Phoenix"),
        ("cp1257", "Rīga Phoenix"),
    ],
)
def test_text_loader_uses_explicit_legacy_encoding(tmp_path, encoding, text):
    document = tmp_path / "legacy.txt"
    document.write_bytes(text.encode(encoding))
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    loaded_documents = qa._load_documents(
        str(document), ".txt", text_encoding=encoding
    )

    assert loaded_documents[0].page_content == text


def test_text_loader_reports_invalid_explicit_encoding(tmp_path):
    document = tmp_path / "legacy.txt"
    document.write_bytes("Phoenix".encode("utf-8"))
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    with pytest.raises(ValueError, match="Could not decode text document"):
        qa._load_documents(str(document), ".txt", text_encoding="not-a-codec")


@pytest.mark.parametrize(
    ("encoding", "text", "mojibake"),
    [
        ("cp1251", "Привет Phoenix", "Ïðèâåò Phoenix"),
        ("cp1251", "аб Phoenix", "àá Phoenix"),
        ("cp1251", "я и ты Phoenix", "ÿ è òû Phoenix"),
        ("cp1251", "Привет — Phoenix", "Ïðèâåò — Phoenix"),
        ("cp1251", "Привет – Phoenix", "Ïðèâåò – Phoenix"),
        ("cp1251", "Привет… Phoenix", "Ïðèâåò… Phoenix"),
        ("cp1251", "Ј Phoenix", "£ Phoenix"),
        ("cp1251", "Јован Phoenix", "£îâàí Phoenix"),
        ("cp1251", "Ђ Phoenix", "€ Phoenix"),
        ("cp1251", "№ Phoenix", "¹ Phoenix"),
        ("gb18030", "项目 Phoenix", "ÏîÄ¿ Phoenix"),
        ("cp1250", "Ł10 Phoenix", "£10 Phoenix"),
        ("cp1250", "Łódź Phoenix", "£ódŸ Phoenix"),
        ("cp1254", "İstanbul Phoenix", "Ýstanbul Phoenix"),
        ("cp1257", "māja Phoenix", "mâja Phoenix"),
        ("cp1257", "Rīga Phoenix", "Rîga Phoenix"),
        ("cp1250", "Dvořák Phoenix", "Dvoøák Phoenix"),
        ("cp1250", "město Phoenix", "mìsto Phoenix"),
        ("cp1250", "książka Phoenix", "ksi¹¿ka Phoenix"),
        ("cp1250", "Dąb Phoenix", "D¹b Phoenix"),
        ("cp1250", "zażalenie Phoenix", "za¿alenie Phoenix"),
        ("iso-8859-2", "Łódź Phoenix", "£ód¼ Phoenix"),
        ("big5", "項目 Phoenix", "¶µ¥Ø Phoenix"),
    ],
)
def test_text_loader_rejects_ambiguous_legacy_text_instead_of_mojibake(
    tmp_path, encoding, text, mojibake
):
    document = tmp_path / "ambiguous.txt"
    document.write_bytes(text.encode(encoding))
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    with pytest.raises(ValueError, match="Could not decode text document"):
        qa._load_documents(str(document), ".txt")

    assert document.read_bytes().decode("cp1252") == mojibake


def test_faiss_vector_store_returns_relevant_documents():
    documents = [
        Document(page_content="Project Phoenix launch notes"),
        Document(page_content="Unrelated accounting memo"),
    ]
    vector_store = FaissVectorStore.from_documents(documents, FakeEmbeddings())

    results = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 1, "fetch_k": 2}
    ).invoke("Phoenix project")

    assert results == [documents[0]]


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
