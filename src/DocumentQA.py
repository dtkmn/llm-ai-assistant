import logging
import os
import sys
import unicodedata
from getpass import getpass
from typing import Dict, List, Optional

import docx2txt
import faiss
import numpy as np
import torch
from charset_normalizer import from_bytes
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ConfigDict, Field
from pypdf import PdfReader

LOGGER = logging.getLogger(__name__)
HF_TOKEN_ENV_VAR = "HUGGINGFACEHUB_API_TOKEN"
FAST_MODE_ENV_VAR = "FAST_MODE"
LLM_BACKEND_ENV_VAR = "LLM_BACKEND"
HF_ENDPOINT_URL_ENV_VAR = "HF_ENDPOINT_URL"
HF_ENDPOINT_TIMEOUT_ENV_VAR = "HF_ENDPOINT_TIMEOUT"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
SUPPORTED_LLM_BACKENDS = {"auto", "endpoint", "local", "mock"}
MAX_DOCUMENT_BYTES = 25 * 1024 * 1024
MAX_DOCUMENT_CHUNKS = 2_000
DEFAULT_QUALITY_EMBEDDINGS_MODEL = "Alibaba-NLP/gte-modernbert-base"
DEFAULT_FAST_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_QUALITY_MODELS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
)
DEFAULT_FAST_MODELS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
)
QUALITY_PROFILE = {
    "max_new_tokens": 384,
    "retrieval_k": 6,
    "retrieval_fetch_k": 24,
    "retrieval_lambda_mult": 0.7,
    "context_chunks": 6,
    "context_chars_per_chunk": 700,
    "context_total_chars": 4200,
    "splitter_chunk_size": 1200,
    "splitter_chunk_overlap": 200,
}
FAST_PROFILE = {
    "max_new_tokens": 160,
    "retrieval_k": 3,
    "retrieval_fetch_k": 10,
    "retrieval_lambda_mult": 0.8,
    "context_chunks": 3,
    "context_chars_per_chunk": 450,
    "context_total_chars": 1800,
    "splitter_chunk_size": 900,
    "splitter_chunk_overlap": 120,
}
DOCUMENT_IDENTITY_QUESTION_HINTS = (
    "uploaded document",
    "which document",
    "what document",
    "document name",
    "file name",
    "filename",
    "what file",
    "which file",
)
TEXT_ENCODING_FALLBACKS = ("cp1252", "latin-1")
TEXT_ENCODING_ALIASES = {
    "cp1252": {"cp1252", "windows_1252"},
    "latin_1": {"latin_1", "iso8859_1", "iso_8859_1"},
}
MAX_SHORT_WESTERN_FALLBACK_BYTES = 256
COMMON_WESTERN_PUNCTUATION = set(
    "'\".,;:!?()[]{}<>-/\\|_@#%&*+=~`^"
    "\u00a0\u00a1\u00ab\u00bb\u00bf\u2013\u2014"
    "\u2018\u2019\u201c\u201d\u2026\u20ac"
)
COMMON_WESTERN_SYMBOLS = {"\u00b0", "\u2122"}
CONFLICTING_LEGACY_ENCODINGS = {
    "big5",
    "big5hkscs",
    "cp932",
    "cp949",
    "cp950",
    "cp1250",
    "cp1251",
    "cp1253",
    "cp1254",
    "cp1255",
    "cp1256",
    "cp1257",
    "cp1258",
    "euc_jis_2004",
    "euc_jisx0213",
    "euc_jp",
    "euc_kr",
    "gb18030",
    "gb2312",
    "gbk",
    "iso8859_2",
    "iso8859_3",
    "iso8859_4",
    "iso8859_5",
    "iso8859_6",
    "iso8859_7",
    "iso8859_8",
    "iso8859_9",
    "iso8859_10",
    "iso8859_11",
    "iso8859_13",
    "iso8859_14",
    "iso8859_16",
    "johab",
    "koi8_r",
    "koi8_t",
    "koi8_u",
    "mac_cyrillic",
    "shift_jis",
    "shift_jis_2004",
    "shift_jisx0213",
}
TEXT_ENCODING_BOMS = (
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
)
UTF_NUL_PATTERN_THRESHOLD = 0.6
DETECTED_ENCODING_MIN_COHERENCE = 0.2
DETECTED_ENCODING_CHAOS_GAP = 5.0
DETECTED_ENCODING_MAX_CHAOS = 5.0
UTF_NUL_FAMILY_ENCODINGS = {
    "utf_16",
    "utf_16_be",
    "utf_16_le",
    "utf_32",
    "utf_32_be",
    "utf_32_le",
}
ALLOWED_TEXT_CONTROL_CHARACTERS = {"\n", "\r", "\t", "\f"}


class FaissVectorStore:
    def __init__(self, documents: List[Document], embeddings, vectors: np.ndarray):
        if vectors.ndim != 2:
            raise ValueError("Embedding vectors must be a 2D array.")
        self.documents = documents
        self.embeddings = embeddings
        self.vectors = vectors.astype("float32")
        self.index = faiss.IndexFlatL2(self.vectors.shape[1])
        self.index.add(self.vectors)

    @classmethod
    def from_documents(cls, documents: List[Document], embedding) -> "FaissVectorStore":
        texts = [document.page_content for document in documents]
        vectors = np.asarray(embedding.embed_documents(texts), dtype="float32")
        if len(vectors) != len(documents):
            raise ValueError("Embedding count does not match document count.")
        if len(documents) == 0:
            raise ValueError("Cannot build a vector store from zero documents.")
        return cls(documents, embedding, vectors)

    def as_retriever(
        self, search_type: str = "similarity", search_kwargs: Optional[Dict] = None
    ) -> "FaissRetriever":
        return FaissRetriever(
            vector_store=self,
            search_type=search_type,
            search_kwargs=search_kwargs or {},
        )

    def search(self, query: str, search_type: str, search_kwargs: Dict) -> List[Document]:
        k = int(search_kwargs.get("k", 4))
        if k <= 0:
            return []

        query_vector = np.asarray([self.embeddings.embed_query(query)], dtype="float32")
        if search_type == "mmr":
            fetch_k = min(int(search_kwargs.get("fetch_k", max(k, 20))), len(self.documents))
            _, indices = self.index.search(query_vector, fetch_k)
            candidate_indices = [int(index) for index in indices[0] if index >= 0]
            selected_indices = self._maximal_marginal_relevance(
                query_vector[0],
                candidate_indices,
                k,
                float(search_kwargs.get("lambda_mult", 0.5)),
            )
            return [self.documents[index] for index in selected_indices]

        search_k = min(k, len(self.documents))
        _, indices = self.index.search(query_vector, search_k)
        return [self.documents[int(index)] for index in indices[0] if index >= 0]

    def _maximal_marginal_relevance(
        self,
        query_vector: np.ndarray,
        candidate_indices: List[int],
        k: int,
        lambda_mult: float,
    ) -> List[int]:
        if not candidate_indices:
            return []

        selected: List[int] = []
        query_norm = self._normalize(query_vector)
        candidate_vectors = self.vectors[candidate_indices]
        candidate_norms = self._normalize_rows(candidate_vectors)
        query_similarities = candidate_norms @ query_norm

        while candidate_indices and len(selected) < k:
            if not selected:
                best_position = int(np.argmax(query_similarities))
            else:
                selected_vectors = self._normalize_rows(self.vectors[selected])
                diversity_penalties = candidate_norms @ selected_vectors.T
                max_diversity_penalties = diversity_penalties.max(axis=1)
                scores = (
                    lambda_mult * query_similarities
                    - (1.0 - lambda_mult) * max_diversity_penalties
                )
                best_position = int(np.argmax(scores))

            selected.append(candidate_indices.pop(best_position))
            candidate_norms = np.delete(candidate_norms, best_position, axis=0)
            query_similarities = np.delete(query_similarities, best_position, axis=0)

        return selected

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _normalize_rows(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)


class FaissRetriever(BaseRetriever):
    vector_store: FaissVectorStore
    search_type: str = "similarity"
    search_kwargs: Dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.vector_store.search(query, self.search_type, self.search_kwargs)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        LOGGER.warning("Invalid integer for %s=%r. Using %s.", name, value, default)
        return default


def normalize_encoding_name(encoding: Optional[str]) -> Optional[str]:
    if not encoding:
        return None
    return encoding.replace("-", "_").lower()


def has_binary_control_characters(text: str) -> bool:
    return any(
        char not in ALLOWED_TEXT_CONTROL_CHARACTERS
        and unicodedata.category(char) == "Cc"
        for char in text
    )


def is_latin_letter(char: str) -> bool:
    return char.isalpha() and "LATIN" in unicodedata.name(char, "")


def western_text_penalty(text: str) -> Optional[int]:
    latin_letter_count = 0
    non_ascii_latin_count = 0
    current_token = ""

    for index, char in enumerate(text):
        if char.isascii() and char.isalpha():
            latin_letter_count += 1
            current_token += char
            continue

        if char.isascii() and (char.isdigit() or char.isspace()):
            if current_token and latin_token_is_suspicious(current_token):
                return None
            current_token = ""
            continue

        if char.isascii() and char in COMMON_WESTERN_PUNCTUATION:
            if current_token and latin_token_is_suspicious(current_token):
                return None
            current_token = ""
            continue

        if is_latin_letter(char):
            latin_letter_count += 1
            if ord(char) > 127:
                non_ascii_latin_count += 1
            current_token += char
            continue

        if char in {"\u00a1", "\u00bf"}:
            if current_token:
                return None
            continue

        if unicodedata.category(char) == "No":
            previous_char = text[index - 1] if index > 0 else ""
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if previous_char.isalpha() and (
                next_char.isalpha() or next_char in {"\u00a1", "\u00bf"}
            ):
                return None
            if current_token and latin_token_is_suspicious(current_token):
                return None
            current_token = ""
            continue

        if char == "\u00a3":
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if not next_char.isascii() or not next_char.isdigit():
                return None
            if current_token and latin_token_is_suspicious(current_token):
                return None
            current_token = ""
            continue

        if char in COMMON_WESTERN_SYMBOLS:
            if current_token and latin_token_is_suspicious(current_token):
                return None
            current_token = ""
            continue

        if char.isspace() or char in COMMON_WESTERN_PUNCTUATION:
            if current_token and latin_token_is_suspicious(current_token):
                return None
            current_token = ""
            continue

        if unicodedata.category(char).startswith(("P", "S")):
            if current_token and latin_token_is_suspicious(current_token):
                return None
            current_token = ""
            continue

        return None

    if current_token and latin_token_is_suspicious(current_token):
        return None

    if (
        latin_letter_count >= 3
        and non_ascii_latin_count / latin_letter_count > 0.85
    ):
        return None

    return 0


def latin_token_is_suspicious(token: str) -> bool:
    latin_letters = [char for char in token if is_latin_letter(char)]
    if len(latin_letters) < 2:
        return False
    return all(ord(char) > 127 for char in latin_letters)


def nul_ratio(raw_content: bytes, offset: int, stride: int) -> float:
    sampled_bytes = raw_content[offset::stride]
    if not sampled_bytes:
        return 0.0
    return sampled_bytes.count(0) / len(sampled_bytes)


class MockLLM(LLM):
    """Fallback LLM when no HuggingFace token/model is available."""

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        if "context:" in prompt.lower():
            parts = prompt.split("Question:")
            if len(parts) > 1:
                question = parts[1].strip()
                return (
                    "Based on the provided context, I can identify information related to "
                    f"{question.lower()}. This is a demonstration response because no real "
                    "language model is currently configured."
                )
        return (
            "This is a mock response. Configure a valid HuggingFace token to enable real "
            "AI-powered answers."
        )


class DocumentQA:
    def __init__(
        self,
        model_id: Optional[str] = None,
        embeddings_model: Optional[str] = None,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        fast_mode: Optional[bool] = None,
        llm_backend: Optional[str] = None,
        allow_interactive_token: bool = False,
        max_document_bytes: int = MAX_DOCUMENT_BYTES,
        max_document_chunks: int = MAX_DOCUMENT_CHUNKS,
    ):
        self.device = device or self._detect_device()
        self.fast_mode = env_flag(FAST_MODE_ENV_VAR, False) if fast_mode is None else fast_mode
        default_models = DEFAULT_FAST_MODELS if self.fast_mode else DEFAULT_QUALITY_MODELS
        default_embeddings_model = (
            DEFAULT_FAST_EMBEDDINGS_MODEL
            if self.fast_mode
            else DEFAULT_QUALITY_EMBEDDINGS_MODEL
        )
        self.model_id = model_id or os.getenv("LLM_MODEL_ID", default_models[0])
        self.embeddings_model = embeddings_model or default_embeddings_model
        self.hf_token = hf_token or self._get_hf_token(allow_interactive_token)
        self.llm_backend = self._normalize_llm_backend(llm_backend)
        self.max_document_bytes = max_document_bytes
        self.max_document_chunks = max_document_chunks
        self.profile = FAST_PROFILE if self.fast_mode else QUALITY_PROFILE

        # Lazy initialization keeps web startup fast on constrained environments
        # (for example Hugging Face Spaces CPU instances).
        self.llm = None
        self.loaded_model_id: Optional[str] = None
        self.loaded_model_label: Optional[str] = None
        self.active_llm_backend: Optional[str] = None
        self.embeddings = None
        self.vector_store = None
        self.retrieval_chain = None
        self.embeddings_error: Optional[str] = None
        self.current_document_name: Optional[str] = None
        self.chat_history: List[Dict[str, str]] = []

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _normalize_llm_backend(self, llm_backend: Optional[str]) -> str:
        backend = (llm_backend or os.getenv(LLM_BACKEND_ENV_VAR, "auto")).strip().lower()
        if backend not in SUPPORTED_LLM_BACKENDS:
            LOGGER.warning(
                "Unsupported %s=%r. Supported values: %s. Using auto.",
                LLM_BACKEND_ENV_VAR,
                backend,
                ", ".join(sorted(SUPPORTED_LLM_BACKENDS)),
            )
            return "auto"
        return backend

    def _select_llm_backend(self) -> str:
        if self.llm_backend != "auto":
            return self.llm_backend
        if self.device in {"cuda", "mps"}:
            return "local"
        return "endpoint"

    def _get_hf_token(self, allow_interactive_token: bool) -> Optional[str]:
        token = os.getenv(HF_TOKEN_ENV_VAR, "").strip()
        if token:
            return token

        if not allow_interactive_token:
            return None

        if not sys.stdin or not sys.stdin.isatty():
            LOGGER.warning(
                "Interactive token prompt requested, but no TTY is available. "
                "Set %s in the environment instead.",
                HF_TOKEN_ENV_VAR,
            )
            return None

        try:
            token = getpass("Enter HF token (or press Enter to skip): ").strip()
        except (KeyboardInterrupt, EOFError):
            LOGGER.warning("No HuggingFace token provided.")
            return None

        if token:
            os.environ[HF_TOKEN_ENV_VAR] = token
            return token
        return None

    def _model_dtype(self) -> torch.dtype:
        if self.device in {"cuda", "mps"}:
            return torch.float16
        return torch.float32

    def _endpoint_url(self) -> Optional[str]:
        return os.getenv(HF_ENDPOINT_URL_ENV_VAR, "").strip() or None

    def _loaded_model_label(self, model_id: str, backend: str) -> str:
        endpoint_url = self._endpoint_url()
        if backend == "endpoint" and endpoint_url:
            return f"Custom endpoint ({endpoint_url})"
        return model_id

    def _load_endpoint_model(self, model_id: str) -> LLM:
        from langchain_huggingface import HuggingFaceEndpoint

        endpoint_url = self._endpoint_url()
        LOGGER.info(
            "Configuring Hugging Face endpoint %s",
            endpoint_url if endpoint_url else model_id,
        )
        endpoint_kwargs = {
            "task": "text-generation",
            "huggingfacehub_api_token": self.hf_token,
            "max_new_tokens": self.profile["max_new_tokens"],
            "do_sample": False,
            "repetition_penalty": 1.05,
            "return_full_text": False,
            "temperature": 0.1,
            "timeout": env_int(HF_ENDPOINT_TIMEOUT_ENV_VAR, 120),
        }
        if endpoint_url:
            endpoint_kwargs["endpoint_url"] = endpoint_url
        else:
            endpoint_kwargs["repo_id"] = model_id
        return HuggingFaceEndpoint(**endpoint_kwargs)

    def _load_local_model(self, model_id: str) -> HuggingFacePipeline:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        LOGGER.info("Loading model %s on %s", model_id, self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.hf_token)

        model_kwargs = {
            "dtype": self._model_dtype(),
            "token": self.hf_token,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        if self.device == "mps":
            model = model.to("mps")

        # Some model configs still carry a legacy max_length default, which
        # causes transformers to warn when max_new_tokens is also provided.
        model.generation_config.max_length = None

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.profile["max_new_tokens"],
            do_sample=False,
            repetition_penalty=1.05,
            return_full_text=False,
        )
        return HuggingFacePipeline(pipeline=text_pipeline)

    def _candidate_models(self) -> List[str]:
        candidates = [self.model_id]
        default_models = DEFAULT_FAST_MODELS if self.fast_mode else DEFAULT_QUALITY_MODELS
        for fallback_model in default_models:
            if fallback_model not in candidates:
                candidates.append(fallback_model)
        return candidates

    def _initialize_llm(self) -> None:
        requested_backend = self._select_llm_backend()
        if self.llm_backend == "mock":
            LOGGER.warning("Mock LLM backend selected. Real model inference is disabled.")
            self.active_llm_backend = "mock"
            self.llm = MockLLM()
            return

        if self.hf_token == "dummy":
            message = (
                "Dummy HuggingFace token is only allowed with LLM_BACKEND=mock "
                "or auto demo fallback."
            )
            if self.llm_backend in {"endpoint", "local"}:
                raise RuntimeError(message)
            LOGGER.warning("%s Falling back to MockLLM.", message)
            self.active_llm_backend = "mock"
            self.llm = MockLLM()
            return

        if requested_backend == "endpoint" and not self.hf_token:
            message = (
                "HuggingFace token is required for endpoint inference. "
                f"Set {HF_TOKEN_ENV_VAR} or use LLM_BACKEND=mock for demo mode."
            )
            if self.llm_backend == "auto":
                LOGGER.warning("%s Falling back to MockLLM.", message)
                self.active_llm_backend = "mock"
                self.llm = MockLLM()
                return
            raise RuntimeError(message)

        last_error = None
        for candidate_model in self._candidate_models():
            try:
                if requested_backend == "endpoint":
                    self.llm = self._load_endpoint_model(candidate_model)
                else:
                    self.llm = self._load_local_model(candidate_model)
                self.active_llm_backend = requested_backend
                self.loaded_model_label = self._loaded_model_label(
                    candidate_model, requested_backend
                )
                self.loaded_model_id = (
                    None
                    if requested_backend == "endpoint" and self._endpoint_url()
                    else candidate_model
                )
                self.model_id = candidate_model
                LOGGER.info("Using %s model %s", requested_backend, candidate_model)
                return
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "Failed to load model %s: %s. Trying next fallback.",
                    candidate_model,
                    exc,
                )

        candidate_list = ", ".join(self._candidate_models()) or "none"
        if last_error and self.llm_backend in {"endpoint", "local"}:
            raise RuntimeError(
                f"Unable to initialize {requested_backend} LLM after trying "
                f"{candidate_list}. Last error: {last_error}"
            ) from last_error

        if last_error:
            LOGGER.exception("All model candidates failed. Falling back to MockLLM.")
        else:
            LOGGER.warning("No model candidates available. Falling back to MockLLM.")
        self.active_llm_backend = "mock"
        self.llm = MockLLM()

    def _ensure_llm_initialized(self) -> None:
        if self.llm is None:
            self._initialize_llm()

    def _initialize_embeddings(self) -> None:
        if self.embeddings is not None:
            return

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embeddings_model,
                model_kwargs={"device": self.device},
            )
        except Exception as exc:
            self.embeddings_error = str(exc)
            self.embeddings = None
            LOGGER.exception(
                "Embeddings initialization failed. Document processing will be unavailable."
            )

    def _create_retrieval_chain(self) -> None:
        if not self.llm:
            raise RuntimeError("LLM is not initialized.")
        if not self.vector_store:
            raise RuntimeError("Vector store is not initialized.")

        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.profile["retrieval_k"],
                "fetch_k": self.profile["retrieval_fetch_k"],
                "lambda_mult": self.profile["retrieval_lambda_mult"],
            },
        )
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful AI assistant. Answer using only the provided context. "
            "If the answer is not in the context, say that clearly.\n\n"
            "Uploaded document name: {document_name}\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        def format_docs(docs):
            filtered_chunks = []
            for doc in docs:
                content = doc.page_content.strip()
                if not content:
                    continue
                metadata = doc.metadata or {}
                source = metadata.get("source")
                source_name = (
                    os.path.basename(str(source))
                    if source
                    else (self.current_document_name or "uploaded document")
                )
                source_label = f"Source: {source_name}"
                page = metadata.get("page")
                if isinstance(page, int):
                    source_label = f"{source_label} (page {page + 1})"
                filtered_chunks.append(
                    f"{source_label}\n{content[: self.profile['context_chars_per_chunk']]}"
                )
            return "\n\n".join(filtered_chunks[: self.profile["context_chunks"]])[
                : self.profile["context_total_chars"]
            ]

        self.retrieval_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "document_name": lambda _: self.current_document_name or "unknown",
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _validate_document(self, document_path: str) -> str:
        if not os.path.exists(document_path):
            raise ValueError("Uploaded file could not be found.")

        file_extension = os.path.splitext(document_path)[1].lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_extension}")

        file_size = os.path.getsize(document_path)
        if file_size > self.max_document_bytes:
            limit_mb = self.max_document_bytes // (1024 * 1024)
            raise ValueError(
                f"File is too large ({file_size} bytes). Maximum supported size is {limit_mb} MB."
            )

        return file_extension

    def _load_documents(
        self,
        document_path: str,
        file_extension: str,
        text_encoding: Optional[str] = None,
    ) -> List[Document]:
        if file_extension == ".pdf":
            reader = PdfReader(document_path)
            documents = []
            for page_number, page in enumerate(reader.pages):
                content = page.extract_text() or ""
                if content.strip():
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": document_path, "page": page_number},
                        )
                    )
            return documents

        if file_extension == ".docx":
            return [
                Document(
                    page_content=docx2txt.process(document_path) or "",
                    metadata={"source": document_path},
                )
            ]

        try:
            content = self._decode_text_file(document_path, text_encoding)
        except (LookupError, UnicodeDecodeError) as exc:
            raise ValueError(f"Could not decode text document: {exc}") from exc
        return [Document(page_content=content, metadata={"source": document_path})]

    def _decode_text_file(
        self, document_path: str, text_encoding: Optional[str] = None
    ) -> str:
        with open(document_path, "rb") as file:
            raw_content = file.read()

        if text_encoding:
            normalized_text_encoding = text_encoding.strip().lower()
            if normalized_text_encoding == "utf-8-or-western":
                return self._decode_utf8_or_western_text(raw_content)
            if normalized_text_encoding != "auto":
                return self._decode_supported_text(raw_content, text_encoding.strip())

        for bom, encoding in TEXT_ENCODING_BOMS:
            if raw_content.startswith(bom):
                return self._decode_supported_text(raw_content, encoding)

        nul_pattern_text = self._decode_nul_pattern_text(raw_content)
        if nul_pattern_text is not None:
            return nul_pattern_text

        try:
            return self._decode_supported_text(raw_content, "utf-8")
        except UnicodeDecodeError:
            pass

        detected_matches = list(from_bytes(raw_content))
        detected_text = self._decode_confident_detected_text(
            raw_content, detected_matches
        )
        if detected_text is not None:
            return detected_text

        for encoding in TEXT_ENCODING_FALLBACKS:
            try:
                fallback_text = self._decode_supported_text(raw_content, encoding)
            except (LookupError, UnicodeDecodeError):
                continue
            if not self._fallback_encoding_is_plausible(
                raw_content, detected_matches, encoding, fallback_text
            ):
                continue
            return fallback_text

        raise UnicodeDecodeError(
            "text-file",
            raw_content,
            0,
            len(raw_content),
            "unable to detect a supported text encoding",
        )

    def _decode_supported_text(self, raw_content: bytes, encoding: str) -> str:
        decoded_text = raw_content.decode(encoding)
        if has_binary_control_characters(decoded_text):
            raise UnicodeDecodeError(
                encoding,
                raw_content,
                0,
                len(raw_content),
                "decoded text contains binary control characters",
            )
        return decoded_text

    def _decode_utf8_or_western_text(self, raw_content: bytes) -> str:
        for bom, encoding in TEXT_ENCODING_BOMS:
            if raw_content.startswith(bom):
                return self._decode_supported_text(raw_content, encoding)

        try:
            return self._decode_supported_text(raw_content, "utf-8")
        except UnicodeDecodeError:
            pass

        for encoding in TEXT_ENCODING_FALLBACKS:
            try:
                return self._decode_supported_text(raw_content, encoding)
            except (LookupError, UnicodeDecodeError):
                continue

        raise UnicodeDecodeError(
            "text-file",
            raw_content,
            0,
            len(raw_content),
            "unable to decode as UTF-8 or Western text",
        )

    def _decode_confident_detected_text(
        self, raw_content: bytes, matches
    ) -> Optional[str]:
        for index, match in enumerate(matches):
            if not self._detected_encoding_is_confident(matches, index):
                continue
            candidates = [match.encoding, *(match.could_be_from_charset or [])]
            for candidate in candidates:
                if not candidate:
                    continue
                normalized = normalize_encoding_name(candidate)
                if normalized in UTF_NUL_FAMILY_ENCODINGS and b"\x00" not in raw_content:
                    continue
                try:
                    return self._decode_supported_text(raw_content, candidate)
                except (LookupError, UnicodeDecodeError):
                    continue
        return None

    def _fallback_encoding_is_plausible(
        self, raw_content: bytes, matches, encoding: str, fallback_text: str
    ) -> bool:
        fallback_penalty = western_text_penalty(fallback_text)
        if fallback_penalty is None:
            return False

        normalized_encoding = normalize_encoding_name(encoding)
        aliases = TEXT_ENCODING_ALIASES.get(
            normalized_encoding or encoding, {normalized_encoding or encoding}
        )
        fallback_listed_as_low_chaos = False

        for match in matches:
            chaos = float(getattr(match, "percent_chaos", 100) or 0)
            if chaos > DETECTED_ENCODING_MAX_CHAOS:
                continue
            candidates = [match.encoding, *(match.could_be_from_charset or [])]
            for candidate in candidates:
                normalized_candidate = normalize_encoding_name(candidate)
                if normalized_candidate in aliases:
                    fallback_listed_as_low_chaos = True
                    continue
                if normalized_candidate not in CONFLICTING_LEGACY_ENCODINGS:
                    continue
                try:
                    candidate_text = self._decode_supported_text(raw_content, candidate)
                except (LookupError, UnicodeDecodeError):
                    continue
                if candidate_text != fallback_text:
                    return False

        # Auto mode must fail closed on ambiguous legacy candidates. The UI
        # defaults to Auto; Western and other legacy codecs are explicit opt-ins.
        if len(raw_content) <= MAX_SHORT_WESTERN_FALLBACK_BYTES:
            return fallback_penalty == 0

        return fallback_listed_as_low_chaos and fallback_penalty == 0

    def _detected_encoding_is_confident(self, matches, index: int) -> bool:
        if index != 0:
            return False

        match = matches[index]
        coherence = float(getattr(match, "coherence", 0) or 0)
        if coherence >= DETECTED_ENCODING_MIN_COHERENCE:
            return True

        chaos = float(getattr(match, "percent_chaos", 100) or 0)
        if chaos > DETECTED_ENCODING_MAX_CHAOS:
            return False
        if index + 1 >= len(matches):
            return False

        next_chaos = float(getattr(matches[index + 1], "percent_chaos", 100) or 0)
        return next_chaos - chaos >= DETECTED_ENCODING_CHAOS_GAP

    def _decode_nul_pattern_text(self, raw_content: bytes) -> Optional[str]:
        if b"\x00" not in raw_content:
            return None

        for encoding in self._nul_pattern_encodings(raw_content):
            try:
                return self._decode_supported_text(raw_content, encoding)
            except (LookupError, UnicodeDecodeError):
                continue
        return None

    def _nul_pattern_encodings(self, raw_content: bytes) -> List[str]:
        encodings: List[str] = []

        if len(raw_content) % 4 == 0:
            utf32le_nul_ratio = (
                nul_ratio(raw_content, 1, 4)
                + nul_ratio(raw_content, 2, 4)
                + nul_ratio(raw_content, 3, 4)
            ) / 3
            utf32be_nul_ratio = (
                nul_ratio(raw_content, 0, 4)
                + nul_ratio(raw_content, 1, 4)
                + nul_ratio(raw_content, 2, 4)
            ) / 3
            utf32le_text_ratio = 1 - nul_ratio(raw_content, 0, 4)
            utf32be_text_ratio = 1 - nul_ratio(raw_content, 3, 4)

            if (
                utf32le_nul_ratio >= UTF_NUL_PATTERN_THRESHOLD
                and utf32le_text_ratio >= UTF_NUL_PATTERN_THRESHOLD
            ):
                encodings.append("utf-32-le")
            if (
                utf32be_nul_ratio >= UTF_NUL_PATTERN_THRESHOLD
                and utf32be_text_ratio >= UTF_NUL_PATTERN_THRESHOLD
            ):
                encodings.append("utf-32-be")

        if len(raw_content) % 2 == 0:
            even_nul_ratio = nul_ratio(raw_content, 0, 2)
            odd_nul_ratio = nul_ratio(raw_content, 1, 2)
            if (
                odd_nul_ratio >= UTF_NUL_PATTERN_THRESHOLD
                and even_nul_ratio <= 1 - UTF_NUL_PATTERN_THRESHOLD
            ):
                encodings.append("utf-16-le")
            if (
                even_nul_ratio >= UTF_NUL_PATTERN_THRESHOLD
                and odd_nul_ratio <= 1 - UTF_NUL_PATTERN_THRESHOLD
            ):
                encodings.append("utf-16-be")

        return encodings

    def process_document(
        self, document_path: str, text_encoding: Optional[str] = None
    ) -> None:
        """Process a document and build a vector store."""
        try:
            file_extension = self._validate_document(document_path)
            self.current_document_name = os.path.basename(document_path)
            self._ensure_llm_initialized()
            if not self.embeddings:
                self._initialize_embeddings()
            if not self.embeddings:
                raise RuntimeError(
                    "Embedding model is unavailable. Check network access/model cache and retry."
                )

            documents = self._load_documents(
                document_path, file_extension, text_encoding=text_encoding
            )
            if not documents:
                raise ValueError("No readable content found in the uploaded document.")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.profile["splitter_chunk_size"],
                chunk_overlap=self.profile["splitter_chunk_overlap"],
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
            )
            chunks = text_splitter.split_documents(documents)
            if not chunks:
                raise ValueError("No text chunks were generated from the document.")

            if len(chunks) > self.max_document_chunks:
                LOGGER.warning(
                    "Document produced %s chunks; truncating to %s to keep memory bounded.",
                    len(chunks),
                    self.max_document_chunks,
                )
                chunks = chunks[: self.max_document_chunks]

            self.vector_store = FaissVectorStore.from_documents(
                documents=chunks, embedding=self.embeddings
            )
            self._create_retrieval_chain()
        except Exception as exc:
            raise RuntimeError(f"Error loading document: {exc}") from exc

    def _is_document_identity_question(self, prompt: str) -> bool:
        lowered = prompt.lower()
        return any(hint in lowered for hint in DOCUMENT_IDENTITY_QUESTION_HINTS)

    def query(self, prompt: str, session_id: str = "default") -> str:
        """Answer a user query using retrieved document context."""
        if not self.retrieval_chain:
            return "Please upload and process a document first."

        if not self.llm:
            return "Language model is not initialized. Please check your HuggingFace token setup."

        clean_prompt = prompt.strip()
        if not clean_prompt:
            return "Please provide a question."

        if self.current_document_name and self._is_document_identity_question(clean_prompt):
            return f"The uploaded document is `{self.current_document_name}`."

        try:
            response = self.retrieval_chain.invoke(clean_prompt)
            if not isinstance(response, str):
                response = str(response)
            response = response.strip()
            if len(response) < 3:
                response = (
                    "I could not find enough relevant information in the document to answer that."
                )

            self.chat_history.append(
                {"session_id": session_id, "question": clean_prompt, "answer": response}
            )
            return response
        except Exception:
            LOGGER.exception("Error while processing query.")
            return "I hit an internal error while processing your question. Please try again."
