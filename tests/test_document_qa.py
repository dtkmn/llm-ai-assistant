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
