import hashlib
import json
import logging
import math
import os
import re
import threading
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

try:
    from .native_runtime import apply_native_runtime_defaults
except ImportError:
    from native_runtime import apply_native_runtime_defaults


apply_native_runtime_defaults()

import docx2txt
import faiss
import numpy as np
from charset_normalizer import from_bytes
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ConfigDict, Field
from pypdf import PdfReader

try:
    from .loop_engine import (
        GuardrailDecision,
        LoopDecision,
        LoopMiddleware,
        LoopPhase,
        LoopPolicy,
        LoopReport,
        LoopRun,
        LoopSession,
        LoopStep,
        VerificationOutcome,
        VerificationResult,
    )
except ImportError:
    from loop_engine import (
        GuardrailDecision,
        LoopDecision,
        LoopMiddleware,
        LoopPhase,
        LoopPolicy,
        LoopReport,
        LoopRun,
        LoopSession,
        LoopStep,
        VerificationOutcome,
        VerificationResult,
    )

try:
    from .runtime_config import (
        DEFAULT_EMBEDDINGS_MODEL,
        DEFAULT_OLLAMA_BASE_URL,
        DEFAULT_OLLAMA_EMBEDDINGS_MODEL,
        DEFAULT_OLLAMA_MODEL,
        EMBEDDINGS_DEVICE_ENV_VAR,
        EMBEDDINGS_MODEL_ENV_VAR,
        FAST_MODE_ENV_VAR,
        LLM_BACKEND_ENV_VAR,
        LLM_MODEL_ENV_VAR,
        OLLAMA_BASE_URL_ENV_VAR,
        OLLAMA_EMBEDDINGS_MODEL_ENV_VAR,
        OLLAMA_MODEL_ENV_VAR,
        OLLAMA_TIMEOUT_ENV_VAR,
        OPENAI_COMPAT_API_KEY_ENV_VAR,
        OPENAI_COMPAT_BASE_URL_ENV_VAR,
        OPENAI_COMPAT_EMBEDDINGS_MODEL_ENV_VAR,
        OPENAI_COMPAT_MODEL_ENV_VAR,
        OPENAI_COMPAT_TIMEOUT_ENV_VAR,
        SUPPORTED_EMBEDDINGS_MODELS,
        SUPPORTED_LLM_BACKENDS,
        env_flag,
        env_int,
        first_env_value,
        is_loopback_host,
        is_loopback_openai_compatible_host,
        normalize_ollama_base_url,
        normalize_openai_compatible_base_url,
        safe_ollama_base_url_for_error,
        safe_openai_compatible_base_url_for_error,
    )
except ImportError:
    from runtime_config import (
        DEFAULT_EMBEDDINGS_MODEL,
        DEFAULT_OLLAMA_BASE_URL,
        DEFAULT_OLLAMA_EMBEDDINGS_MODEL,
        DEFAULT_OLLAMA_MODEL,
        EMBEDDINGS_DEVICE_ENV_VAR,
        EMBEDDINGS_MODEL_ENV_VAR,
        FAST_MODE_ENV_VAR,
        LLM_BACKEND_ENV_VAR,
        LLM_MODEL_ENV_VAR,
        OLLAMA_BASE_URL_ENV_VAR,
        OLLAMA_EMBEDDINGS_MODEL_ENV_VAR,
        OLLAMA_MODEL_ENV_VAR,
        OLLAMA_TIMEOUT_ENV_VAR,
        OPENAI_COMPAT_API_KEY_ENV_VAR,
        OPENAI_COMPAT_BASE_URL_ENV_VAR,
        OPENAI_COMPAT_EMBEDDINGS_MODEL_ENV_VAR,
        OPENAI_COMPAT_MODEL_ENV_VAR,
        OPENAI_COMPAT_TIMEOUT_ENV_VAR,
        SUPPORTED_EMBEDDINGS_MODELS,
        SUPPORTED_LLM_BACKENDS,
        env_flag,
        env_int,
        first_env_value,
        is_loopback_host,
        is_loopback_openai_compatible_host,
        normalize_ollama_base_url,
        normalize_openai_compatible_base_url,
        safe_ollama_base_url_for_error,
        safe_openai_compatible_base_url_for_error,
    )

LOGGER = logging.getLogger(__name__)
OLLAMA_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
OPENAI_COMPAT_NO_PROXY_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({})
)
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_DOCUMENT_BYTES = 25 * 1024 * 1024
MAX_DOCUMENT_CHUNKS = 2_000
DEFAULT_MAX_SESSION_REPORTS = 200
LOCAL_HASHING_EMBEDDINGS_DIMENSIONS = 384
LOCAL_HASHING_ALIASES = {
    "go_live": ("launch", "launch_date", "release", "release_date"),
    "launch": ("go_live", "release", "start"),
    "launched": ("launch", "go_live", "released"),
    "launches": ("launch", "go_live", "releases"),
    "launching": ("launch", "go_live", "releasing"),
    "release": ("launch", "go_live"),
    "released": ("release", "launch"),
    "releases": ("release", "launch"),
    "release_date": ("launch_date", "go_live"),
    "launch_date": ("release_date", "go_live"),
    "live": ("go_live",),
}
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
SELF_CHECK_REFUSAL_ANSWER = (
    "I could not find enough relevant information in the document to answer that."
)
SELF_CHECK_PASS_OUTCOMES = {"supported", "not_verified"}
VERIFIER_OUTCOMES = {"supported", "unsupported", "insufficient"}


def open_ollama_request_no_proxy(request, *, timeout: int):
    try:
        parsed = urllib.parse.urlsplit(request.full_url)
        _port = parsed.port
    except ValueError:
        raise RuntimeError(
            f"{OLLAMA_BASE_URL_ENV_VAR} must be a valid loopback HTTP(S) URL."
        ) from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or not is_loopback_host(parsed.hostname)
        or parsed.username
        or parsed.password
    ):
        raise RuntimeError(
            f"{OLLAMA_BASE_URL_ENV_VAR} must point to loopback local Ollama. "
            "Use LLM_BACKEND=openai-compatible for remote gateways."
        )
    return OLLAMA_NO_PROXY_OPENER.open(request, timeout=timeout)


def open_openai_compatible_request(request, *, timeout: int):
    try:
        parsed = urllib.parse.urlsplit(request.full_url)
    except ValueError:
        parsed = None
    if (
        parsed is not None
        and parsed.hostname
        and is_loopback_openai_compatible_host(parsed.hostname)
    ):
        return OPENAI_COMPAT_NO_PROXY_OPENER.open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)


ANSWER_SUPPORT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "because",
    "by",
    "can",
    "configured",
    "context",
    "currently",
    "demonstration",
    "did",
    "do",
    "does",
    "document",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "identify",
    "in",
    "information",
    "is",
    "it",
    "language",
    "model",
    "no",
    "of",
    "on",
    "or",
    "provided",
    "real",
    "related",
    "response",
    "that",
    "the",
    "this",
    "to",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class DocumentProcessingReport:
    attempted_document_name: Optional[str]
    active_document_name: Optional[str]
    success: bool
    phase: str
    file_extension: Optional[str]
    chunk_count: int
    truncated: bool
    max_chunk_limit: int
    text_encoding_mode: str
    backend: str
    model_label: str
    error_message: Optional[str]


@dataclass(frozen=True)
class DocumentQAStatus:
    profile_label: str
    configured_backend: str
    active_backend: str
    active_model_label: str
    loaded_model_id: Optional[str]
    loaded_model_label: Optional[str]
    embeddings_model: str
    embeddings_device: str
    device: str
    document_name: Optional[str]
    ready_for_queries: bool
    processing_report: Optional[DocumentProcessingReport]

    @property
    def mock_mode(self) -> bool:
        return self.active_backend == "mock"


@dataclass(frozen=True)
class AnswerCitation:
    citation_id: int
    source_name: str
    page: Optional[int]
    chunk_index: Optional[int]
    excerpt: str


@dataclass(frozen=True)
class AnswerTrace:
    question: str
    document_name: Optional[str]
    backend: str
    model_label: str
    retrieved_chunk_count: int
    citations: List[AnswerCitation]
    self_check: Optional["AnswerSelfCheck"] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class QueryResult:
    answer: str
    trace: AnswerTrace
    loop_report: Optional[LoopReport] = None


@dataclass(frozen=True)
class AnswerSelfCheck:
    outcome: str
    reasons: List[str]
    retry_attempted: bool = False


class ContextProvider(Protocol):
    provider_type: str
    display_name: Optional[str]
    vector_store: Optional["FaissVectorStore"]
    retrieval_chain: Optional[object]


@dataclass(frozen=True)
class DocumentContextProvider:
    document_name: Optional[str]
    vector_store: Optional["FaissVectorStore"]
    retrieval_chain: Optional[object]
    provider_type: str = "document"

    @property
    def display_name(self) -> Optional[str]:
        return self.document_name

    @property
    def ready(self) -> bool:
        return self.retrieval_chain is not None


@dataclass(frozen=True)
class ActiveDocumentState:
    document_name: Optional[str]
    vector_store: Optional["FaissVectorStore"]
    retrieval_chain: Optional[object]
    context_provider: Optional[ContextProvider] = None


class DocumentProcessingError(RuntimeError):
    def __init__(self, message: str, status: DocumentQAStatus):
        super().__init__(message)
        self.status = status


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


@dataclass(frozen=True)
class RetrievalChainResult:
    answer: str
    retrieved_chunk_count: int
    citations: List[AnswerCitation]
    context: str = ""


@dataclass(frozen=True)
class RetrievedContext:
    retrieved_chunk_count: int
    citations: List[AnswerCitation]
    context: str


class DocumentRetrievalChain:
    def __init__(
        self,
        *,
        retriever: FaissRetriever,
        prompt: ChatPromptTemplate,
        llm: LLM,
        document_name: Optional[str],
        profile: Dict,
    ):
        self.retriever = retriever
        self.answer_chain = prompt | llm | StrOutputParser()
        self.document_name = document_name
        self.profile = profile

    def invoke(self, question: str) -> str:
        return self.invoke_with_trace(question).answer

    def invoke_with_trace(
        self, question: str, self_check_instruction: str = ""
    ) -> RetrievalChainResult:
        retrieved_context = self.retrieve_with_trace(question)
        return self.draft_with_trace(
            question,
            retrieved_context,
            self_check_instruction=self_check_instruction,
        )

    def retrieve_with_trace(self, question: str) -> RetrievedContext:
        docs = self.retriever.invoke(question)
        context, citations = self._format_context_and_citations(docs)
        return RetrievedContext(
            retrieved_chunk_count=len(citations),
            citations=citations,
            context=context,
        )

    def draft_with_trace(
        self,
        question: str,
        retrieved_context: RetrievedContext,
        self_check_instruction: str = "",
    ) -> RetrievalChainResult:
        response = self._generate_answer(
            question=question,
            context=retrieved_context.context,
            self_check_instruction=self_check_instruction,
        )
        return RetrievalChainResult(
            answer=response,
            retrieved_chunk_count=retrieved_context.retrieved_chunk_count,
            citations=retrieved_context.citations,
            context=retrieved_context.context,
        )

    def retry_with_trace(
        self,
        question: str,
        previous_result: RetrievalChainResult,
        self_check_instruction: str,
    ) -> RetrievalChainResult:
        response = self._generate_answer(
            question=question,
            context=previous_result.context,
            self_check_instruction=self_check_instruction,
        )
        return RetrievalChainResult(
            answer=response,
            retrieved_chunk_count=previous_result.retrieved_chunk_count,
            citations=previous_result.citations,
            context=previous_result.context,
        )

    def _generate_answer(
        self, *, question: str, context: str, self_check_instruction: str = ""
    ) -> str:
        response = self.answer_chain.invoke(
            {
                "context": context,
                "question": question,
                "document_name": self.document_name or "unknown",
                "self_check_instruction": self_check_instruction,
            }
        )
        return str(response).strip()

    def _format_context_and_citations(
        self, docs: List[Document]
    ) -> Tuple[str, List[AnswerCitation]]:
        context_parts = []
        citations = []
        remaining_chars = self.profile["context_total_chars"]

        for doc in docs:
            if len(citations) >= self.profile["context_chunks"] or remaining_chars <= 0:
                break

            content = doc.page_content.strip()
            if not content:
                continue

            citation_id = len(citations) + 1
            metadata = doc.metadata or {}
            source_name = self._source_name(metadata)
            page = metadata.get("page")
            display_page = page + 1 if isinstance(page, int) else None
            chunk_index = metadata.get("chunk_index")
            if not isinstance(chunk_index, int):
                chunk_index = None

            source_label = f"[{citation_id}] Source: {source_name}"
            if display_page is not None:
                source_label = f"{source_label} (page {display_page})"
            if chunk_index is not None:
                source_label = f"{source_label}, chunk {chunk_index + 1}"

            allowed_content_chars = min(
                self.profile["context_chars_per_chunk"],
                max(0, remaining_chars - len(source_label) - 1),
            )
            if allowed_content_chars <= 0:
                break

            excerpt = content[:allowed_content_chars]
            context_entry = f"{source_label}\n{excerpt}"
            context_parts.append(context_entry)
            remaining_chars -= len(context_entry) + 2
            citations.append(
                AnswerCitation(
                    citation_id=citation_id,
                    source_name=source_name,
                    page=display_page,
                    chunk_index=chunk_index,
                    excerpt=" ".join(excerpt.split()),
                )
            )

        return "\n\n".join(context_parts), citations

    def _source_name(self, metadata: Dict) -> str:
        source = metadata.get("source")
        if source:
            return os.path.basename(str(source))
        return self.document_name or "uploaded document"


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


class LocalHashingEmbeddings(Embeddings):
    """Deterministic local lexical embeddings with no model download."""

    def __init__(self, dimensions: int = LOCAL_HASHING_EMBEDDINGS_DIMENSIONS):
        self.dimensions = max(32, int(dimensions))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        phrase_tokens = [
            f"{left}_{right}" for left, right in zip(tokens, tokens[1:])
        ]
        weighted_terms = [(token, 1.0) for token in tokens]
        weighted_terms.extend((token, 0.8) for token in phrase_tokens)
        for token in tokens + phrase_tokens:
            weighted_terms.extend(
                (alias, 0.65) for alias in LOCAL_HASHING_ALIASES.get(token, ())
            )

        for token, weight in weighted_terms:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = -1.0 if digest[4] & 1 else 1.0
            vector[bucket] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]


def _validate_embedding_vectors(
    embeddings: Any, provider_label: str
) -> List[List[float]]:
    if not isinstance(embeddings, list) or not embeddings:
        raise RuntimeError(f"{provider_label} embeddings response lacked vectors")

    validated: List[List[float]] = []
    for embedding in embeddings:
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(f"{provider_label} embeddings response lacked vectors")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in embedding
        ):
            raise RuntimeError(
                f"{provider_label} embeddings response contained non-numeric values"
            )
        vector = [float(value) for value in embedding]
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError(
                f"{provider_label} embeddings response contained non-finite values"
            )
        validated.append(vector)

    dimensions = {len(embedding) for embedding in validated}
    if len(dimensions) != 1:
        raise RuntimeError(
            f"{provider_label} embeddings response had inconsistent dimensions"
        )
    return validated


def _validate_single_embedding_vector(
    embeddings: List[List[float]], provider_label: str
) -> List[float]:
    if len(embeddings) != 1:
        raise RuntimeError(f"{provider_label} embeddings response had unexpected length")
    return embeddings[0]


class OllamaEmbeddings(Embeddings):
    """Ollama embedding adapter using the selected local provider runtime."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = normalize_ollama_base_url(base_url)
        self.timeout = timeout

    def _post_embed(self, inputs: str | List[str]) -> List[List[float]]:
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=json.dumps(
                {
                    "model": self.model,
                    "input": inputs,
                    "truncate": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with open_ollama_request_no_proxy(
                request, timeout=self.timeout
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Ollama embeddings API returned HTTP {exc.code} for /api/embed."
            ) from None
        except urllib.error.URLError:
            raise RuntimeError(
                f"Could not connect to Ollama embeddings at {self.base_url}/api/embed."
            ) from None

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama embeddings endpoint returned invalid JSON") from exc

        embeddings = parsed.get("embeddings") if isinstance(parsed, dict) else None
        return _validate_embedding_vectors(embeddings, "Ollama")

    def validate_model_available(self) -> None:
        _validate_single_embedding_vector(self._post_embed("ok"), "Ollama")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._post_embed(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError("Ollama embeddings response had unexpected length")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embeddings = self._post_embed(text)
        return _validate_single_embedding_vector(embeddings, "Ollama")


class MockLLM(LLM):
    """Explicit deterministic demo/test LLM."""

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        if "context:" in prompt.lower():
            context = prompt.split("Context:", 1)[1].split("Question:", 1)[0]
            citation_id = None
            for line in context.splitlines():
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                citation_match = re.match(r"\[(\d+)\]", stripped_line)
                if citation_match:
                    citation_id = citation_match.group(1)
                    continue
                if citation_id:
                    sentence = re.split(r"(?<=[.!?])\s+", stripped_line, maxsplit=1)[0]
                    return f"{sentence} [{citation_id}]"
        return (
            "This is a mock response. Configure Ollama or an OpenAI-compatible "
            "backend to enable real AI-powered answers."
        )


class OllamaLLM(LLM):
    """Minimal Ollama adapter for deterministic local model inference."""

    model: str
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    timeout: int = 120
    options: Dict[str, object] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "ollama"

    @property
    def _api_base_url(self) -> str:
        return normalize_ollama_base_url(self.base_url)

    def _post_json(self, path: str, payload: Dict[str, object]) -> Dict[str, object]:
        request = urllib.request.Request(
            f"{self._api_base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with open_ollama_request_no_proxy(
                request, timeout=self.timeout
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Ollama API returned HTTP {exc.code} for {path}.") from None
        except urllib.error.URLError:
            raise RuntimeError(
                f"Could not connect to Ollama at {self._api_base_url}{path}."
            ) from None

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama returned invalid JSON for {path}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError(f"Ollama returned unexpected JSON for {path}")
        return parsed

    def validate_model_available(self) -> None:
        self._post_json("/api/show", {"model": self.model})

    def _strip_thinking_text(self, text: str) -> str:
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        dangling_think_end = re.search(r"</think>", cleaned, flags=re.IGNORECASE)
        if dangling_think_end:
            cleaned = cleaned[dangling_think_end.end() :]
        dangling_think_start = re.search(r"<think>", cleaned, flags=re.IGNORECASE)
        if dangling_think_start:
            cleaned = cleaned[: dangling_think_start.start()]
        return cleaned.strip()

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        options = dict(self.options)
        if stop:
            options["stop"] = stop
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": options,
        }
        response = self._post_json("/api/generate", payload)
        generated_text = response.get("response")
        if not isinstance(generated_text, str):
            raise RuntimeError("Ollama response did not include generated text.")
        return self._strip_thinking_text(generated_text)


class OpenAICompatibleLLM(LLM):
    """Minimal OpenAI-compatible chat-completions adapter."""

    model: str
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 120
    max_tokens: int = 384
    model_config = ConfigDict(validate_assignment=True)

    def model_post_init(self, __context) -> None:
        self.base_url = normalize_openai_compatible_base_url(self.base_url)

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

    @property
    def _api_base_url(self) -> str:
        return normalize_openai_compatible_base_url(self.base_url)

    @property
    def _chat_completions_url(self) -> str:
        return f"{self._api_base_url}/chat/completions"

    def _post_chat(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self._chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        error_message = None
        try:
            with open_openai_compatible_request(
                request, timeout=self.timeout
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status_code = exc.code
            error_message = (
                "OpenAI-compatible API returned HTTP "
                f"{status_code} for /chat/completions."
            )
        except urllib.error.URLError:
            safe_base_url = safe_openai_compatible_base_url_for_error(self.base_url)
            error_message = (
                "Could not connect to OpenAI-compatible endpoint at "
                f"{safe_base_url}/chat/completions."
            )
        if error_message:
            raise RuntimeError(error_message) from None

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenAI-compatible endpoint returned invalid JSON") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("OpenAI-compatible endpoint returned unexpected JSON")
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI-compatible response did not include choices")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("OpenAI-compatible response choice was invalid")
        message = first_choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()
        if isinstance(first_choice.get("text"), str):
            return first_choice["text"].strip()
        raise RuntimeError("OpenAI-compatible response did not include generated text")

    def validate_model_available(self) -> None:
        self._post_chat("Respond with ok.", max_tokens=1)

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return self._post_chat(prompt, stop=stop)


class OpenAICompatibleEmbeddings(Embeddings):
    """OpenAI-compatible embeddings adapter using /embeddings."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = normalize_openai_compatible_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout

    @property
    def _embeddings_url(self) -> str:
        return f"{self.base_url}/embeddings"

    def _post_embeddings(self, inputs: str | List[str]) -> List[List[float]]:
        expected_count = 1 if isinstance(inputs, str) else len(inputs)
        payload = {
            "model": self.model,
            "input": inputs,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self._embeddings_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        error_message = None
        try:
            with open_openai_compatible_request(
                request, timeout=self.timeout
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_message = (
                "OpenAI-compatible embeddings API returned HTTP "
                f"{exc.code} for /embeddings."
            )
        except urllib.error.URLError:
            safe_base_url = safe_openai_compatible_base_url_for_error(self.base_url)
            error_message = (
                "Could not connect to OpenAI-compatible embeddings endpoint at "
                f"{safe_base_url}/embeddings."
            )
        if error_message:
            raise RuntimeError(error_message) from None

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "OpenAI-compatible embeddings endpoint returned invalid JSON"
            ) from exc

        data = parsed.get("data") if isinstance(parsed, dict) else None
        if not isinstance(data, list):
            raise RuntimeError("OpenAI-compatible embeddings response was invalid")
        if len(data) != expected_count:
            raise RuntimeError(
                "OpenAI-compatible embeddings response had unexpected length"
            )
        ordered_embeddings: List[Any] = [None] * expected_count
        seen_indices = set()
        for item in data:
            if not isinstance(item, dict):
                raise RuntimeError(
                    "OpenAI-compatible embeddings response item was invalid"
                )
            index = item.get("index")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= expected_count
                or index in seen_indices
            ):
                raise RuntimeError(
                    "OpenAI-compatible embeddings response had invalid indices"
                )
            seen_indices.add(index)
            ordered_embeddings[index] = item.get("embedding")
        if len(seen_indices) != expected_count or any(
            embedding is None for embedding in ordered_embeddings
        ):
            raise RuntimeError(
                "OpenAI-compatible embeddings response had invalid indices"
            )
        embeddings = ordered_embeddings
        return _validate_embedding_vectors(embeddings, "OpenAI-compatible")

    def validate_model_available(self) -> None:
        _validate_single_embedding_vector(
            self._post_embeddings("ok"), "OpenAI-compatible"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._post_embeddings(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError(
                "OpenAI-compatible embeddings response had unexpected length"
            )
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embeddings = self._post_embeddings(text)
        return _validate_single_embedding_vector(
            embeddings, "OpenAI-compatible"
        )


class AILoopEngine:
    def __init__(
        self,
        model_id: Optional[str] = None,
        embeddings_model: Optional[str] = None,
        device: Optional[str] = None,
        embeddings_device: Optional[str] = None,
        fast_mode: Optional[bool] = None,
        llm_backend: Optional[str] = None,
        max_document_bytes: int = MAX_DOCUMENT_BYTES,
        max_document_chunks: int = MAX_DOCUMENT_CHUNKS,
        max_session_reports: int = DEFAULT_MAX_SESSION_REPORTS,
        loop_middlewares: Optional[Tuple[LoopMiddleware, ...]] = None,
    ):
        self.device = device or "cpu"
        self.fast_mode = env_flag(FAST_MODE_ENV_VAR, False) if fast_mode is None else fast_mode
        self.llm_backend = self._normalize_llm_backend(llm_backend)
        self.model_id = model_id
        self.max_document_bytes = max_document_bytes
        self.max_document_chunks = max_document_chunks
        self.max_session_reports = max(0, int(max_session_reports))
        self.profile = FAST_PROFILE if self.fast_mode else QUALITY_PROFILE
        self.ollama_base_url = (
            os.getenv(OLLAMA_BASE_URL_ENV_VAR, DEFAULT_OLLAMA_BASE_URL).strip()
            or DEFAULT_OLLAMA_BASE_URL
        )
        generic_llm_model = (model_id or first_env_value(LLM_MODEL_ENV_VAR) or "").strip()
        self.ollama_model = (
            generic_llm_model
            or os.getenv(OLLAMA_MODEL_ENV_VAR, DEFAULT_OLLAMA_MODEL).strip()
            or DEFAULT_OLLAMA_MODEL
        )
        self.ollama_timeout = env_int(OLLAMA_TIMEOUT_ENV_VAR, 120)
        self.openai_compat_base_url = os.getenv(
            OPENAI_COMPAT_BASE_URL_ENV_VAR, ""
        ).strip()
        self.openai_compat_api_key = (
            os.getenv(OPENAI_COMPAT_API_KEY_ENV_VAR, "").strip()
            or None
        )
        self.openai_compat_model = generic_llm_model or os.getenv(
            OPENAI_COMPAT_MODEL_ENV_VAR, ""
        ).strip()
        self.openai_compat_timeout = env_int(OPENAI_COMPAT_TIMEOUT_ENV_VAR, 120)
        self.embeddings_model = self._resolve_embeddings_model(embeddings_model)
        self.embeddings_device = self._normalize_embeddings_device(embeddings_device)

        # Lazy initialization keeps web startup fast on constrained environments.
        self.llm = None
        self.loaded_model_id: Optional[str] = None
        self.loaded_model_label: Optional[str] = None
        self.active_llm_backend: Optional[str] = None
        self.embeddings = None
        self._state_lock = threading.RLock()
        self._upload_lock = threading.Lock()
        self._session_lock = threading.RLock()
        self._active_document_state = ActiveDocumentState(
            document_name=None,
            vector_store=None,
            retrieval_chain=None,
            context_provider=DocumentContextProvider(
                document_name=None,
                vector_store=None,
                retrieval_chain=None,
            ),
        )
        self.vector_store = None
        self.retrieval_chain = None
        self.embeddings_error: Optional[str] = None
        self.current_document_name: Optional[str] = None
        self.latest_processing_report: Optional[DocumentProcessingReport] = None
        self.chat_history: List[Dict[str, str]] = []
        self.loop_sessions: Dict[str, LoopSession] = {}
        self.loop_middlewares = tuple(loop_middlewares or ())

    def _default_embeddings_device(self) -> str:
        if self._select_llm_backend() in {"ollama", "openai-compatible"}:
            return "external"
        return "cpu"

    def _resolve_embeddings_model(self, embeddings_model: Optional[str]) -> str:
        selected_backend = self._select_llm_backend()
        requested = (embeddings_model or "").strip()
        generic_model = first_env_value(EMBEDDINGS_MODEL_ENV_VAR)

        if selected_backend == "mock":
            requested = requested or generic_model or DEFAULT_EMBEDDINGS_MODEL
            if requested in SUPPORTED_EMBEDDINGS_MODELS:
                return requested
            raise RuntimeError(
                "Unsupported embeddings_model="
                f"{requested!r}. Mock document embeddings support only "
                f"{DEFAULT_EMBEDDINGS_MODEL!r}."
            )

        if selected_backend == "ollama":
            return (
                requested
                or generic_model
                or first_env_value(OLLAMA_EMBEDDINGS_MODEL_ENV_VAR)
                or DEFAULT_OLLAMA_EMBEDDINGS_MODEL
            )

        if selected_backend == "openai-compatible":
            return (
                requested
                or generic_model
                or first_env_value(OPENAI_COMPAT_EMBEDDINGS_MODEL_ENV_VAR)
                or ""
            )

        raise RuntimeError(
            f"Unsupported embeddings backend for {LLM_BACKEND_ENV_VAR}={self.llm_backend!r}."
        )

    def _normalize_embeddings_device(
        self, embeddings_device: Optional[str] = None
    ) -> str:
        requested = (
            embeddings_device
            or os.getenv(EMBEDDINGS_DEVICE_ENV_VAR, "auto")
        ).strip().lower()
        if not requested or requested == "auto":
            return self._default_embeddings_device()
        if self._default_embeddings_device() == "external":
            LOGGER.warning(
                "%s=%s requested, but %s=%s uses external provider embeddings.",
                EMBEDDINGS_DEVICE_ENV_VAR,
                requested,
                LLM_BACKEND_ENV_VAR,
                self.llm_backend,
            )
            return "external"
        if requested in {"cuda", "mps"}:
            LOGGER.warning(
                "%s=%s requested, but built-in local hashing embeddings run on CPU.",
                EMBEDDINGS_DEVICE_ENV_VAR,
                requested,
            )
            return "cpu"
        if requested == "cpu":
            return "cpu"
        LOGGER.warning(
            "Unsupported %s=%r. Supported values: auto, cpu. Using CPU.",
            EMBEDDINGS_DEVICE_ENV_VAR,
            requested,
        )
        return "cpu"

    def _normalize_llm_backend(self, llm_backend: Optional[str]) -> str:
        raw_backend = (
            llm_backend
            if llm_backend is not None
            else os.getenv(LLM_BACKEND_ENV_VAR, "auto")
        )
        backend = raw_backend.strip().lower().replace("_", "-")
        if backend in {"openai", "openai-compatible-chat"}:
            backend = "openai-compatible"
        if backend not in SUPPORTED_LLM_BACKENDS:
            raise RuntimeError(
                f"Unsupported {LLM_BACKEND_ENV_VAR}={raw_backend!r}. "
                "Supported values: "
                f"{', '.join(sorted(SUPPORTED_LLM_BACKENDS))}."
            )
        return backend

    def _select_llm_backend(self) -> str:
        if self.llm_backend != "auto":
            return self.llm_backend
        return "ollama"

    def _loaded_model_label(self, model_id: str, backend: str) -> str:
        if backend == "ollama":
            return f"Ollama ({model_id})"
        if backend == "openai-compatible":
            return f"OpenAI-compatible ({model_id or 'unconfigured'})"
        return model_id

    def _load_ollama_model(self, model_id: str) -> OllamaLLM:
        base_url = normalize_ollama_base_url(self.ollama_base_url)
        LOGGER.info("Configuring Ollama model %s at %s", model_id, base_url)
        llm = OllamaLLM(
            model=model_id,
            base_url=base_url,
            timeout=self.ollama_timeout,
            options={
                "temperature": 0,
                "num_predict": self.profile["max_new_tokens"],
            },
        )
        llm.validate_model_available()
        return llm

    def _load_ollama_embeddings(self, model_id: str) -> OllamaEmbeddings:
        if not model_id:
            raise RuntimeError(
                f"{EMBEDDINGS_MODEL_ENV_VAR} or {OLLAMA_EMBEDDINGS_MODEL_ENV_VAR} "
                "is required for Ollama embeddings."
            )
        base_url = normalize_ollama_base_url(self.ollama_base_url)
        LOGGER.info(
            "Configuring Ollama embedding model %s at %s",
            model_id,
            base_url,
        )
        embeddings = OllamaEmbeddings(
            model=model_id,
            base_url=base_url,
            timeout=self.ollama_timeout,
        )
        embeddings.validate_model_available()
        return embeddings

    def _load_openai_compatible_model(self, model_id: str) -> OpenAICompatibleLLM:
        base_url = normalize_openai_compatible_base_url(self.openai_compat_base_url)
        if not model_id:
            raise RuntimeError(
                f"{LLM_MODEL_ENV_VAR} or {OPENAI_COMPAT_MODEL_ENV_VAR} is required for "
                "LLM_BACKEND=openai-compatible."
            )
        LOGGER.info(
            "Configuring OpenAI-compatible model %s at %s",
            model_id,
            safe_openai_compatible_base_url_for_error(base_url),
        )
        llm = OpenAICompatibleLLM(
            model=model_id,
            base_url=base_url,
            api_key=self.openai_compat_api_key,
            timeout=self.openai_compat_timeout,
            max_tokens=self.profile["max_new_tokens"],
        )
        llm.validate_model_available()
        return llm

    def _load_openai_compatible_embeddings(
        self, model_id: str
    ) -> OpenAICompatibleEmbeddings:
        base_url = normalize_openai_compatible_base_url(self.openai_compat_base_url)
        if not model_id:
            raise RuntimeError(
                f"{EMBEDDINGS_MODEL_ENV_VAR} or "
                f"{OPENAI_COMPAT_EMBEDDINGS_MODEL_ENV_VAR} is required for "
                "LLM_BACKEND=openai-compatible document embeddings."
            )
        LOGGER.info(
            "Configuring OpenAI-compatible embedding model %s at %s",
            model_id,
            safe_openai_compatible_base_url_for_error(base_url),
        )
        embeddings = OpenAICompatibleEmbeddings(
            model=model_id,
            base_url=base_url,
            api_key=self.openai_compat_api_key,
            timeout=self.openai_compat_timeout,
        )
        embeddings.validate_model_available()
        return embeddings

    def _initialize_llm(self) -> None:
        requested_backend = self._select_llm_backend()
        if self.llm_backend == "mock":
            LOGGER.warning("Mock LLM backend selected. Real model inference is disabled.")
            self.active_llm_backend = "mock"
            self.llm = MockLLM()
            return

        if requested_backend == "ollama":
            try:
                self.llm = self._load_ollama_model(self.ollama_model)
                self.active_llm_backend = "ollama"
                self.loaded_model_id = self.ollama_model
                self.loaded_model_label = self._loaded_model_label(
                    self.ollama_model, "ollama"
                )
                LOGGER.info("Using Ollama model %s", self.ollama_model)
                return
            except Exception as exc:
                self.llm = None
                self.active_llm_backend = None
                backend_label = (
                    "auto-selected ollama"
                    if self.llm_backend == "auto"
                    else "ollama"
                )
                safe_base_url = safe_ollama_base_url_for_error(
                    self.ollama_base_url
                )
                raise RuntimeError(
                    f"Unable to initialize {backend_label} LLM "
                    f"`{self.ollama_model}` at {safe_base_url}. "
                    "Start Ollama and pull the configured model, choose "
                    "LLM_BACKEND=openai-compatible for a gateway, or set "
                    "LLM_BACKEND=mock only for explicit demo/test mode. "
                    f"Last error: {exc}"
                ) from exc

        if requested_backend == "openai-compatible":
            try:
                self.llm = self._load_openai_compatible_model(
                    self.openai_compat_model
                )
                self.active_llm_backend = "openai-compatible"
                self.loaded_model_id = self.openai_compat_model
                self.loaded_model_label = self._loaded_model_label(
                    self.openai_compat_model, "openai-compatible"
                )
                LOGGER.info(
                    "Using OpenAI-compatible model %s", self.openai_compat_model
                )
                return
            except Exception as exc:
                self.llm = None
                self.active_llm_backend = None
                safe_base_url = safe_openai_compatible_base_url_for_error(
                    self.openai_compat_base_url
                )
                raise RuntimeError(
                    "Unable to initialize openai-compatible LLM "
                    f"`{self.openai_compat_model}` at "
                    f"{safe_base_url}. Last error: {exc}"
                ) from exc

        raise RuntimeError(
            f"Unsupported initialized LLM backend: {requested_backend}."
        )

    def _ensure_llm_initialized(self) -> None:
        with self._state_lock:
            if self.llm is None:
                self._initialize_llm()

    def _snapshot_active_document_state(self) -> ActiveDocumentState:
        with self._state_lock:
            return self._active_document_state

    def _normalize_session_id(self, session_id: Optional[str]) -> str:
        return str(session_id or "default").strip() or "default"

    def _record_loop_report(self, loop_report: Optional[LoopReport]) -> None:
        if loop_report is None:
            return
        session_id = self._normalize_session_id(loop_report.run.session_id)
        with self._session_lock:
            session = self.loop_sessions.get(
                session_id, LoopSession(session_id=session_id)
            )
            self.loop_sessions[session_id] = session.add_report(
                loop_report,
                max_reports=self.max_session_reports,
            )

    def loop_session(self, session_id: str = "default") -> LoopSession:
        normalized_session_id = self._normalize_session_id(session_id)
        with self._session_lock:
            return self.loop_sessions.get(
                normalized_session_id,
                LoopSession(session_id=normalized_session_id),
            )

    def loop_sessions_snapshot(self) -> Dict[str, LoopSession]:
        with self._session_lock:
            return dict(self.loop_sessions)

    def clear_loop_session(self, session_id: Optional[str] = "default") -> None:
        with self._session_lock:
            if session_id is None:
                self.loop_sessions.clear()
                return
            self.loop_sessions.pop(self._normalize_session_id(session_id), None)

    def export_loop_session_jsonl(
        self,
        path: str,
        *,
        session_id: str = "default",
        public: bool = False,
    ) -> str:
        artifact_path = self.loop_session(session_id).write_jsonl(
            path,
            public=public,
        )
        return str(artifact_path)

    def _commit_active_document_state(
        self,
        *,
        document_name: str,
        vector_store: FaissVectorStore,
        retrieval_chain,
        processing_report: DocumentProcessingReport,
    ) -> DocumentQAStatus:
        context_provider = DocumentContextProvider(
            document_name=document_name,
            vector_store=vector_store,
            retrieval_chain=retrieval_chain,
        )
        active_state = ActiveDocumentState(
            document_name=document_name,
            vector_store=vector_store,
            retrieval_chain=retrieval_chain,
            context_provider=context_provider,
        )
        with self._state_lock:
            self._active_document_state = active_state
            # Keep legacy attributes synchronized for existing tests and callers.
            self.current_document_name = active_state.document_name
            self.vector_store = active_state.vector_store
            self.retrieval_chain = active_state.retrieval_chain
            self.latest_processing_report = processing_report
            return self._status_from_locked(active_state, processing_report)

    def _record_processing_failure(
        self,
        *,
        attempted_document_name: Optional[str],
        phase: str,
        file_extension: Optional[str],
        chunk_count: int,
        truncated: bool,
        text_encoding: Optional[str],
        error_message: str,
    ) -> DocumentQAStatus:
        with self._state_lock:
            processing_report = self._processing_report(
                attempted_document_name=attempted_document_name,
                success=False,
                phase=phase,
                file_extension=file_extension,
                chunk_count=chunk_count,
                truncated=truncated,
                text_encoding=text_encoding,
                error_message=error_message,
                active_document_name=self._active_document_state.document_name,
            )
            self.latest_processing_report = processing_report
            return self._status_from_locked(
                self._active_document_state, processing_report
            )

    def _status_from_locked(
        self,
        active_state: ActiveDocumentState,
        processing_report: Optional[DocumentProcessingReport],
    ) -> DocumentQAStatus:
        active_backend = self._active_backend()
        return DocumentQAStatus(
            profile_label="FAST" if self.fast_mode else "QUALITY",
            configured_backend=self.llm_backend,
            active_backend=active_backend,
            active_model_label=self._active_model_label(),
            loaded_model_id=self.loaded_model_id,
            loaded_model_label=self.loaded_model_label,
            embeddings_model=self.embeddings_model,
            embeddings_device=self.embeddings_device,
            device=self.device,
            document_name=active_state.document_name,
            ready_for_queries=(
                self.llm is not None and active_state.retrieval_chain is not None
            ),
            processing_report=processing_report,
        )

    def status(self) -> DocumentQAStatus:
        with self._state_lock:
            active_state = self._active_document_state
            processing_report = self.latest_processing_report
            return self._status_from_locked(active_state, processing_report)

    def _initialize_embeddings(self) -> None:
        if self.embeddings is not None:
            return

        try:
            selected_backend = self._select_llm_backend()
            if selected_backend == "mock":
                self.embeddings = LocalHashingEmbeddings()
                return
            if selected_backend == "ollama":
                self.embeddings = self._load_ollama_embeddings(
                    self.embeddings_model
                )
                return
            if selected_backend == "openai-compatible":
                self.embeddings = self._load_openai_compatible_embeddings(
                    self.embeddings_model
                )
                return
            raise RuntimeError(
                f"Unsupported embeddings backend: {selected_backend}."
            )
        except Exception as exc:
            self.embeddings_error = str(exc)
            self.embeddings = None
            LOGGER.exception(
                "Embeddings initialization failed. Document processing will be unavailable."
            )

    def _active_backend(self) -> str:
        return self.active_llm_backend or self.llm_backend

    def _active_model_label(self) -> str:
        active_model_label = self.loaded_model_label or self.loaded_model_id
        if active_model_label:
            return active_model_label

        active_backend = self._active_backend()
        if active_backend == "mock":
            return "MockLLM (explicit demo)"
        if active_backend == "auto":
            return f"Auto (Ollama {self.ollama_model})"
        if active_backend == "ollama":
            return self._loaded_model_label(self.ollama_model, "ollama")
        if active_backend == "openai-compatible":
            return self._loaded_model_label(
                self.openai_compat_model, "openai-compatible"
            )
        return self.model_id or "unconfigured"

    def _text_encoding_mode(self, text_encoding: Optional[str]) -> str:
        return (text_encoding or "auto").strip().lower() or "auto"

    def _processing_report(
        self,
        *,
        attempted_document_name: Optional[str],
        success: bool,
        phase: str,
        file_extension: Optional[str],
        chunk_count: int,
        truncated: bool,
        text_encoding: Optional[str],
        error_message: Optional[str] = None,
        active_document_name: Optional[str] = None,
    ) -> DocumentProcessingReport:
        active_state = self._snapshot_active_document_state()
        return DocumentProcessingReport(
            attempted_document_name=attempted_document_name,
            active_document_name=(
                active_state.document_name
                if active_document_name is None
                else active_document_name
            ),
            success=success,
            phase=phase,
            file_extension=file_extension,
            chunk_count=chunk_count,
            truncated=truncated,
            max_chunk_limit=self.max_document_chunks,
            text_encoding_mode=self._text_encoding_mode(text_encoding),
            backend=self._active_backend(),
            model_label=self._active_model_label(),
            error_message=error_message,
        )

    def _build_retrieval_chain(
        self, vector_store: FaissVectorStore, document_name: Optional[str]
    ):
        if not self.llm:
            raise RuntimeError("LLM is not initialized.")
        if not vector_store:
            raise RuntimeError("Vector store is not initialized.")

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.profile["retrieval_k"],
                "fetch_k": self.profile["retrieval_fetch_k"],
                "lambda_mult": self.profile["retrieval_lambda_mult"],
            },
        )
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful AI assistant. Answer using only the provided context. "
            "Cite relevant sources inline with bracketed numbers like [1]. "
            "If the answer is not in the context, say that clearly and do not invent citations.\n\n"
            "Uploaded document name: {document_name}\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "{self_check_instruction}\n\n"
            "Answer:"
        )

        return DocumentRetrievalChain(
            retriever=retriever,
            prompt=prompt,
            llm=self.llm,
            document_name=document_name,
            profile=self.profile,
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
    ) -> DocumentQAStatus:
        """Process a document and build a vector store."""
        with self._upload_lock:
            attempted_document_name = (
                os.path.basename(document_path) if document_path else None
            )
            file_extension = (
                os.path.splitext(document_path)[1].lower() if document_path else None
            )
            phase = "validate"
            chunk_count = 0
            truncated = False

            try:
                file_extension = self._validate_document(document_path)

                phase = "initialize_llm"
                self._ensure_llm_initialized()

                phase = "initialize_embeddings"
                if not self.embeddings:
                    self._initialize_embeddings()
                if not self.embeddings:
                    error_detail = (
                        f" Last error: {self.embeddings_error}"
                        if self.embeddings_error
                        else ""
                    )
                    raise RuntimeError(
                        "Embedding backend is unavailable. "
                        "Check the configured embedding model and provider runtime."
                        f"{error_detail}"
                    )

                phase = "load"
                documents = self._load_documents(
                    document_path, file_extension, text_encoding=text_encoding
                )
                if not documents:
                    raise ValueError("No readable content found in the uploaded document.")

                phase = "split"
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.profile["splitter_chunk_size"],
                    chunk_overlap=self.profile["splitter_chunk_overlap"],
                    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
                )
                chunks = text_splitter.split_documents(documents)
                if not chunks:
                    raise ValueError("No text chunks were generated from the document.")

                if len(chunks) > self.max_document_chunks:
                    truncated = True
                    LOGGER.warning(
                        "Document produced %s chunks; truncating to %s to keep memory bounded.",
                        len(chunks),
                        self.max_document_chunks,
                    )
                    chunks = chunks[: self.max_document_chunks]
                chunk_count = len(chunks)
                for chunk_index, chunk in enumerate(chunks):
                    chunk.metadata = dict(chunk.metadata or {})
                    chunk.metadata["chunk_index"] = chunk_index

                phase = "index"
                vector_store = FaissVectorStore.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                )

                phase = "chain"
                retrieval_chain = self._build_retrieval_chain(
                    vector_store, attempted_document_name
                )

                processing_report = self._processing_report(
                    attempted_document_name=attempted_document_name,
                    active_document_name=attempted_document_name,
                    success=True,
                    phase="complete",
                    file_extension=file_extension,
                    chunk_count=chunk_count,
                    truncated=truncated,
                    text_encoding=text_encoding,
                )
                return self._commit_active_document_state(
                    document_name=attempted_document_name,
                    vector_store=vector_store,
                    retrieval_chain=retrieval_chain,
                    processing_report=processing_report,
                )
            except Exception as exc:
                status = self._record_processing_failure(
                    attempted_document_name=attempted_document_name,
                    phase=phase,
                    file_extension=file_extension,
                    chunk_count=chunk_count,
                    truncated=truncated,
                    text_encoding=text_encoding,
                    error_message=str(exc),
                )
                raise DocumentProcessingError(
                    f"Error loading document: {exc}", status
                ) from exc

    def _is_document_identity_question(self, prompt: str) -> bool:
        lowered = prompt.lower()
        return any(hint in lowered for hint in DOCUMENT_IDENTITY_QUESTION_HINTS)

    def _answer_is_refusal(self, answer: str) -> bool:
        lowered = answer.lower()
        refusal_markers = (
            "could not find",
            "not in the context",
            "not enough relevant information",
            "not provided in the context",
            "cannot answer",
            "can't answer",
            "insufficient information",
        )
        return any(marker in lowered for marker in refusal_markers)

    def _inline_citation_ids(self, answer: str) -> List[int]:
        citation_ids = []
        for match in re.findall(r"\[(\d+)\]", answer):
            try:
                citation_ids.append(int(match))
            except ValueError:
                continue
        return citation_ids

    def _citation_ids_are_valid(
        self, inline_citation_ids: List[int], citations: List[AnswerCitation]
    ) -> bool:
        valid_citation_ids = {citation.citation_id for citation in citations}
        return set(inline_citation_ids).issubset(valid_citation_ids)

    def _cited_citations_for_answer(
        self, answer: str, citations: List[AnswerCitation]
    ) -> List[AnswerCitation]:
        inline_citation_ids = set(self._inline_citation_ids(answer))
        return [
            citation for citation in citations if citation.citation_id in inline_citation_ids
        ]

    def _support_token(self, token: str) -> str:
        if len(token) > 4 and token.endswith("ies"):
            return f"{token[:-3]}y"
        if len(token) > 4 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def _support_tokens(self, text: str) -> set:
        normalized_text = re.sub(r"\[\d+\]", " ", text.lower())
        tokens = re.findall(r"\w+", normalized_text, flags=re.UNICODE)
        return {
            self._support_token(token)
            for token in tokens
            if (not token.isascii() or len(token) > 2)
            and token not in ANSWER_SUPPORT_STOPWORDS
        }

    def _normalize_support_text(self, text: str) -> str:
        without_citations = re.sub(r"\[\d+\]", " ", text.lower())
        normalized_chars = []
        for char in without_citations:
            if char.isalnum():
                normalized_chars.append(char)
            else:
                normalized_chars.append(" ")
        return " ".join("".join(normalized_chars).split())

    def _matched_claim_is_denied(
        self,
        normalized_answer: str,
        normalized_evidence: str,
        raw_evidence: str = "",
    ) -> bool:
        if not normalized_answer:
            return False

        match_starts = []
        search_start = 0
        while True:
            start = normalized_evidence.find(normalized_answer, search_start)
            if start < 0:
                break
            match_starts.append(start)
            search_start = start + len(normalized_answer)

        if not match_starts:
            return False

        if raw_evidence:
            claim_pattern = r"\b" + r"\W+".join(
                re.escape(token) for token in normalized_answer.split()
            )
            qa_denial_pattern = (
                rf"{claim_pattern}\b\s*(?:\?|\:|[-–—])\s*"
                r"(?:no|nope)\b(?=\s*(?:[.!?,;:]|$))"
            )
            if re.search(qa_denial_pattern, raw_evidence, flags=re.IGNORECASE):
                return True

        prefix_denial_markers = (
            "it is false that",
            "it is not true that",
            "it is incorrect that",
            "it is wrong that",
            "it was false that",
            "it was not true that",
            "it was incorrect that",
            "it was wrong that",
            "false that",
            "not true that",
            "incorrect that",
            "wrong that",
            "denied that",
            "refuted that",
        )

        denial_markers = (
            "is false",
            "is not true",
            "is incorrect",
            "is wrong",
            "is denied",
            "is refuted",
            "is rejected",
            "is debunked",
            "is contradicted",
            "is untrue",
            "is unsupported",
            "is not supported",
            "is disputed",
            "is inaccurate",
            "is baseless",
            "is unfounded",
            "is disproven",
            "is disproved",
            "has been denied",
            "has been refuted",
            "has been rejected",
            "has been debunked",
            "has been contradicted",
            "has been unsupported",
            "has been disputed",
            "has been disproven",
            "has been disproved",
            "have been denied",
            "have been refuted",
            "have been rejected",
            "have been debunked",
            "have been contradicted",
            "have been unsupported",
            "have been disputed",
            "have been disproven",
            "have been disproved",
            "had been denied",
            "had been refuted",
            "had been rejected",
            "had been debunked",
            "had been contradicted",
            "had been unsupported",
            "had been disputed",
            "had been disproven",
            "had been disproved",
            "was false",
            "was not true",
            "was incorrect",
            "was wrong",
            "was denied",
            "was refuted",
            "was rejected",
            "was debunked",
            "was contradicted",
            "was untrue",
            "was unsupported",
            "was not supported",
            "was disputed",
            "was inaccurate",
            "was baseless",
            "was unfounded",
            "was disproven",
            "was disproved",
            "are false",
            "are not true",
            "are incorrect",
            "are wrong",
            "are denied",
            "are refuted",
            "are rejected",
            "are debunked",
            "are contradicted",
            "are untrue",
            "are unsupported",
            "are not supported",
            "are disputed",
            "are inaccurate",
            "are baseless",
            "are unfounded",
            "are disproven",
            "are disproved",
            "were false",
            "were not true",
            "were incorrect",
            "were wrong",
            "were denied",
            "were refuted",
            "were rejected",
            "were debunked",
            "were contradicted",
            "were untrue",
            "were unsupported",
            "were not supported",
            "were disputed",
            "were inaccurate",
            "were baseless",
            "were unfounded",
            "were disproven",
            "were disproved",
            "not true",
            "not supported",
            "false",
            "incorrect",
            "wrong",
            "denied",
            "refuted",
            "rejected",
            "debunked",
            "contradicted",
            "untrue",
            "unsupported",
            "disputed",
            "inaccurate",
            "baseless",
            "unfounded",
            "disproven",
            "disproved",
        )

        def has_denial_marker(tokens: List[str]) -> bool:
            text = " ".join(tokens[:8])
            return any(
                text == marker or text.startswith(f"{marker} ")
                for marker in denial_markers
            )

        referent_tokens = {"it", "that", "this", "they", "these", "those"}
        referential_determiners = {"a", "an", "the"}
        referential_nouns = {
            "answer",
            "assertion",
            "assertions",
            "claim",
            "claims",
            "idea",
            "ideas",
            "premise",
            "premises",
            "report",
            "reports",
            "statement",
            "statements",
        }
        referential_modifiers = {"above", "previous", "prior", "same"}

        def has_referential_denial_marker(tokens: List[str]) -> bool:
            if has_denial_marker(tokens):
                return True

            if tokens and tokens[0] in referent_tokens:
                if has_denial_marker(tokens[1:]):
                    return True
                if len(tokens) > 1 and tokens[1] in referential_nouns:
                    return has_denial_marker(tokens[2:])

            if tokens and tokens[0] in referential_determiners:
                noun_tokens = tokens[1:]
                if noun_tokens and noun_tokens[0] in referential_modifiers:
                    noun_tokens = noun_tokens[1:]
                if noun_tokens and noun_tokens[0] in referential_nouns:
                    return has_denial_marker(noun_tokens[1:])

            return False

        discourse_connectors = {
            "although",
            "but",
            "however",
            "that",
            "this",
            "though",
            "which",
            "yet",
        }

        for start in match_starts:
            before_match = normalized_evidence[:start]
            previous_tokens = before_match.split()[-6:]
            previous_text = " ".join(previous_tokens)
            if any(
                previous_text == marker or previous_text.endswith(f" {marker}")
                for marker in prefix_denial_markers
            ):
                return True

            after_match = normalized_evidence[start + len(normalized_answer) :]
            following_tokens = after_match.split()[:8]
            following_text = " ".join(following_tokens)
            if following_tokens and following_tokens[0] in {"no", "nope"}:
                if len(following_tokens) == 1:
                    return True
                if following_tokens[1] in {"it", "this", "that", "instead", "rather"}:
                    return True
                if following_text.startswith(
                    ("no not true", "no false", "no incorrect")
                ):
                    return True

            if has_referential_denial_marker(following_tokens):
                return True

            if following_tokens and following_tokens[0] in discourse_connectors:
                connector_tail = following_tokens[1:]
                if has_referential_denial_marker(connector_tail):
                    return True

        return False

    def _citation_text_refutes_answer(
        self, answer: str, citations: List[AnswerCitation], question: str
    ) -> bool:
        cited_citations = self._cited_citations_for_answer(answer, citations)
        if not cited_citations:
            return False

        evidence_text_parts = []
        for citation in cited_citations:
            evidence_text_parts.append(citation.excerpt)

        answer_tokens = self._support_tokens(answer)
        if not answer_tokens:
            return False

        normalized_answer = self._normalize_support_text(answer)
        normalized_evidence = self._normalize_support_text(" ".join(evidence_text_parts))
        if not normalized_answer or normalized_answer not in normalized_evidence:
            return False
        return self._matched_claim_is_denied(
            normalized_answer,
            normalized_evidence,
            " ".join(evidence_text_parts),
        )

    def _mechanical_self_check_answer(
        self,
        answer: str,
        citations: List[AnswerCitation],
        *,
        question: str = "",
        retry_attempted: bool = False,
    ) -> AnswerSelfCheck:
        clean_answer = answer.strip()
        if len(clean_answer) < 3:
            return AnswerSelfCheck(
                outcome="needs_retry",
                reasons=["answer_too_short"],
                retry_attempted=retry_attempted,
            )

        if not citations:
            if self._answer_is_refusal(clean_answer):
                return AnswerSelfCheck(
                    outcome="not_verified",
                    reasons=["refused_without_prompt_evidence"],
                    retry_attempted=retry_attempted,
                )
            return AnswerSelfCheck(
                outcome="needs_refusal",
                reasons=["no_prompt_evidence"],
                retry_attempted=retry_attempted,
            )

        if self._answer_is_refusal(clean_answer):
            return AnswerSelfCheck(
                outcome="needs_retry",
                reasons=["answer_refused_despite_prompt_evidence"],
                retry_attempted=retry_attempted,
            )

        inline_citation_ids = self._inline_citation_ids(clean_answer)
        if not inline_citation_ids:
            return AnswerSelfCheck(
                outcome="needs_retry",
                reasons=["missing_inline_citation"],
                retry_attempted=retry_attempted,
            )

        if not self._citation_ids_are_valid(inline_citation_ids, citations):
            return AnswerSelfCheck(
                outcome="needs_retry",
                reasons=["invalid_inline_citation"],
                retry_attempted=retry_attempted,
            )

        if self._citation_text_refutes_answer(clean_answer, citations, question):
            return AnswerSelfCheck(
                outcome="needs_refusal",
                reasons=[
                    "citation_text_does_not_support_answer",
                    "deterministic_refutation_detected",
                ],
                retry_attempted=retry_attempted,
            )

        return AnswerSelfCheck(
            outcome="mechanical_checks_passed",
            reasons=["mechanical_checks_passed"],
            retry_attempted=retry_attempted,
        )

    def _loop_decision_for_self_check(self, self_check: AnswerSelfCheck) -> LoopDecision:
        if self_check.outcome == "mechanical_checks_passed":
            return LoopDecision.CONTINUE
        if self_check.outcome == "supported":
            return LoopDecision.SUPPORTED
        if self_check.outcome == "not_verified":
            return LoopDecision.NOT_VERIFIED
        if self_check.outcome == "needs_retry":
            return LoopDecision.RETRY
        if self_check.outcome == "needs_refusal":
            return LoopDecision.REFUSE
        return LoopDecision.ERROR

    def _verification_result_for_self_check(
        self, self_check: AnswerSelfCheck
    ) -> VerificationResult:
        reasons = tuple(self_check.reasons)
        if self_check.outcome == "supported":
            outcome = VerificationOutcome.SUPPORTED
        elif self_check.outcome == "not_verified":
            outcome = VerificationOutcome.NOT_VERIFIED
        elif "llm_verifier_unsupported" in reasons:
            outcome = VerificationOutcome.UNSUPPORTED
        elif "llm_verifier_insufficient" in reasons:
            outcome = VerificationOutcome.INSUFFICIENT
        else:
            outcome = VerificationOutcome.ERROR

        return VerificationResult(
            outcome=outcome,
            reasons=reasons,
            verifier=self._active_backend(),
            metadata={
                "retry_attempted": self_check.retry_attempted,
                "self_check_outcome": self_check.outcome,
            },
        )

    def _loop_step(
        self,
        phase: LoopPhase,
        *,
        decision: LoopDecision = LoopDecision.CONTINUE,
        name: Optional[str] = None,
        input_summary: Optional[str] = None,
        output_summary: Optional[str] = None,
        retry_count: int = 0,
        error_message: Optional[str] = None,
        verification: Optional[VerificationResult] = None,
        human_review=None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LoopStep:
        return self._planned_loop_step(
            phase,
            decision=decision,
            name=name,
            input_summary=input_summary,
            retry_count=retry_count,
            error_message=error_message,
            verification=verification,
            human_review=human_review,
            metadata=metadata,
        ).complete(output_summary=output_summary, error_message=error_message)

    def _planned_loop_step(
        self,
        phase: LoopPhase,
        *,
        decision: LoopDecision = LoopDecision.CONTINUE,
        name: Optional[str] = None,
        input_summary: Optional[str] = None,
        retry_count: int = 0,
        error_message: Optional[str] = None,
        verification: Optional[VerificationResult] = None,
        human_review=None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LoopStep:
        return LoopStep(
            phase=phase,
            decision=decision,
            name=name,
            input_summary=input_summary,
            backend=self._active_backend(),
            model_label=self._active_model_label(),
            retry_count=retry_count,
            error_message=error_message,
            verification=verification,
            human_review=human_review,
            metadata=metadata or {},
        )

    def _self_check_answer_with_loop_steps(
        self,
        answer: str,
        citations: List[AnswerCitation],
        *,
        question: str = "",
        retry_attempted: bool = False,
    ) -> Tuple[AnswerSelfCheck, List[LoopStep]]:
        mechanical_check = self._mechanical_self_check_answer(
            answer,
            citations,
            question=question,
            retry_attempted=retry_attempted,
        )
        retry_count = 1 if retry_attempted else 0
        mechanical_step = self._loop_step(
            LoopPhase.MECHANICAL_CHECK,
            decision=self._loop_decision_for_self_check(mechanical_check),
            name="Mechanical answer checks",
            input_summary="answer plus prompt citations",
            output_summary=mechanical_check.outcome,
            retry_count=retry_count,
            metadata={
                "reasons": list(mechanical_check.reasons),
                "citation_count": len(citations),
                "inline_citation_ids": self._inline_citation_ids(answer),
                "retry_attempted": retry_attempted,
            },
        )
        if mechanical_check.outcome != "mechanical_checks_passed":
            return mechanical_check, [mechanical_step]

        if self._active_backend() == "mock":
            self_check = AnswerSelfCheck(
                outcome="not_verified",
                reasons=[
                    "mechanical_checks_passed",
                    "verifier_unavailable_mock_backend",
                ],
                retry_attempted=retry_attempted,
            )
        else:
            self_check = self._verify_answer_with_llm(
                question=question,
                answer=answer,
                citations=citations,
                retry_attempted=retry_attempted,
            )

        verify_step = self._loop_step(
            LoopPhase.VERIFY,
            decision=self._loop_decision_for_self_check(self_check),
            name="Answer verifier",
            input_summary="answer plus cited excerpts",
            output_summary=self_check.outcome,
            retry_count=retry_count,
            verification=self._verification_result_for_self_check(self_check),
            metadata={
                "reasons": list(self_check.reasons),
                "citation_count": len(citations),
                "retry_attempted": retry_attempted,
            },
        )
        return self_check, [mechanical_step, verify_step]

    def _self_check_answer(
        self,
        answer: str,
        citations: List[AnswerCitation],
        *,
        question: str = "",
        retry_attempted: bool = False,
    ) -> AnswerSelfCheck:
        self_check, _steps = self._self_check_answer_with_loop_steps(
            answer,
            citations,
            question=question,
            retry_attempted=retry_attempted,
        )
        return self_check

    def _verifier_prompt(
        self, *, question: str, answer: str, citations: List[AnswerCitation]
    ) -> str:
        cited_citations = self._cited_citations_for_answer(answer, citations)
        cited_excerpts = "\n\n".join(
            f"[{citation.citation_id}] {citation.excerpt.strip()}"
            for citation in cited_citations
        )
        return (
            "You are a strict citation verifier for a document QA system.\n"
            "Use only the cited excerpts. Do not use outside knowledge.\n"
            "Decide whether every factual claim in the answer is directly supported "
            "by the cited excerpts.\n"
            "Return only JSON with this schema: "
            '{"outcome":"supported|unsupported|insufficient","reason":"short reason"}.\n'
            "Use unsupported when the cited excerpts contradict any answer claim.\n"
            "Use insufficient when the cited excerpts do not contain enough evidence.\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Cited excerpts:\n{cited_excerpts}"
        )

    def _parse_verifier_response(self, raw_response: str) -> Tuple[Optional[str], str]:
        json_match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
        if not json_match:
            return None, "missing_json"
        try:
            payload = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None, "invalid_json"

        outcome = str(payload.get("outcome", "")).strip().lower()
        if outcome not in VERIFIER_OUTCOMES:
            return None, "invalid_outcome"
        reason = str(payload.get("reason", "")).strip()
        return outcome, reason

    def _verify_answer_with_llm(
        self,
        *,
        question: str,
        answer: str,
        citations: List[AnswerCitation],
        retry_attempted: bool,
    ) -> AnswerSelfCheck:
        if self.llm is None:
            return AnswerSelfCheck(
                outcome="needs_refusal",
                reasons=["llm_verifier_unavailable"],
                retry_attempted=retry_attempted,
            )

        prompt = self._verifier_prompt(
            question=question,
            answer=answer,
            citations=citations,
        )
        try:
            raw_response = str(self.llm.invoke(prompt)).strip()
        except Exception as exc:
            LOGGER.warning("LLM verifier failed: %s", exc)
            return AnswerSelfCheck(
                outcome="needs_refusal",
                reasons=["llm_verifier_error"],
                retry_attempted=retry_attempted,
            )

        verifier_outcome, parse_reason = self._parse_verifier_response(raw_response)
        if verifier_outcome is None:
            return AnswerSelfCheck(
                outcome="needs_refusal",
                reasons=["llm_verifier_parse_failed", parse_reason],
                retry_attempted=retry_attempted,
            )

        if verifier_outcome == "supported":
            return AnswerSelfCheck(
                outcome="supported",
                reasons=["mechanical_checks_passed", "llm_verifier_supported"],
                retry_attempted=retry_attempted,
            )

        return AnswerSelfCheck(
            outcome="needs_refusal",
            reasons=[f"llm_verifier_{verifier_outcome}"],
            retry_attempted=retry_attempted,
        )

    def _fail_closed_self_check(self, self_check: AnswerSelfCheck) -> AnswerSelfCheck:
        reasons = list(self_check.reasons)
        if "self_check_failed_closed" not in reasons:
            reasons.insert(0, "self_check_failed_closed")
        return AnswerSelfCheck(
            outcome="needs_refusal",
            reasons=reasons,
            retry_attempted=self_check.retry_attempted,
        )

    def _self_check_retry_instruction(self, self_check: AnswerSelfCheck) -> str:
        reasons = ", ".join(self_check.reasons)
        return (
            "Self-check retry instruction: the previous answer failed checks "
            f"({reasons}). Answer again using only the context above. Include at least "
            "one bracketed citation like [1] for supported claims. If the context does "
            "not contain the answer, say that clearly."
        )

    def _query_result(
        self,
        *,
        answer: str,
        question: str,
        active_state: ActiveDocumentState,
        retrieved_chunk_count: int = 0,
        citations: Optional[List[AnswerCitation]] = None,
        self_check: Optional[AnswerSelfCheck] = None,
        error_message: Optional[str] = None,
        loop_report: Optional[LoopReport] = None,
    ) -> QueryResult:
        return QueryResult(
            answer=answer,
            trace=AnswerTrace(
                question=question,
                document_name=active_state.document_name,
                backend=self._active_backend(),
                model_label=self._active_model_label(),
                retrieved_chunk_count=retrieved_chunk_count,
                citations=citations or [],
                self_check=self_check,
                error_message=error_message,
            ),
            loop_report=loop_report,
        )

    def _start_loop_run(
        self,
        *,
        prompt: str,
        session_id: str,
        active_state: ActiveDocumentState,
    ) -> LoopRun:
        context_provider = active_state.context_provider
        context_provider_type = (
            context_provider.provider_type if context_provider else "document"
        )
        return LoopRun(
            user_input=prompt,
            context_provider=context_provider_type,
            backend=self._active_backend(),
            model_label=self._active_model_label(),
            session_id=session_id,
            policy=LoopPolicy(max_retries=1),
            metadata={
                "document_name": active_state.document_name,
                "context_provider": context_provider_type,
                "context_provider_name": (
                    context_provider.display_name if context_provider else None
                ),
                "profile": "FAST" if self.fast_mode else "QUALITY",
                "allow_tool_calls": False,
                "untrusted_inputs": [
                    "document_text",
                    "retrieved_chunks",
                    "model_output",
                    "future_tool_output",
                ],
            },
        )

    def _coerce_guardrail_decision(self, decision) -> Optional[GuardrailDecision]:
        if decision is None:
            return None
        if isinstance(decision, GuardrailDecision):
            return decision
        return GuardrailDecision(decision=LoopDecision(decision))

    def _run_loop_middleware(
        self, hook_name: str, *args
    ) -> Optional[GuardrailDecision]:
        for middleware in self.loop_middlewares:
            hook = getattr(middleware, hook_name, None)
            if not callable(hook):
                continue
            try:
                decision = self._coerce_guardrail_decision(hook(*args))
            except Exception as exc:
                LOGGER.exception("Loop middleware %s failed in %s.", middleware, hook_name)
                return GuardrailDecision(
                    decision=LoopDecision.BLOCK,
                    reason=f"middleware_{hook_name}_error",
                    metadata={
                        "middleware": middleware.__class__.__name__,
                        "error": str(exc),
                    },
                )
            if decision and not decision.can_continue:
                return decision
        return None

    def _append_loop_step(
        self, run: LoopRun, step: LoopStep
    ) -> Tuple[LoopRun, Optional[GuardrailDecision]]:
        guardrail_decision = self._prepare_loop_step(run, step)
        if guardrail_decision:
            return run, guardrail_decision

        return self._record_loop_step(run, step)

    def _prepare_loop_step(
        self, run: LoopRun, step: LoopStep
    ) -> Optional[GuardrailDecision]:
        return self._run_loop_middleware("before_step", run, step)

    def _record_loop_step(
        self, run: LoopRun, step: LoopStep
    ) -> Tuple[LoopRun, Optional[GuardrailDecision]]:
        next_run = run.with_step(step)
        guardrail_decision = self._run_loop_middleware(
            "after_step", next_run, step
        )
        if guardrail_decision:
            return next_run, guardrail_decision
        return next_run, None

    def _append_loop_steps(
        self, run: LoopRun, steps: List[LoopStep]
    ) -> Tuple[LoopRun, Optional[GuardrailDecision]]:
        for step in steps:
            run, guardrail_decision = self._append_loop_step(run, step)
            if guardrail_decision:
                return run, guardrail_decision
        return run, None

    def _run_self_check_with_loop(
        self,
        *,
        run: LoopRun,
        answer: str,
        citations: List[AnswerCitation],
        question: str,
        retry_attempted: bool = False,
    ) -> Tuple[LoopRun, Optional[GuardrailDecision], Optional[AnswerSelfCheck]]:
        retry_count = 1 if retry_attempted else 0
        planned_mechanical_step = self._planned_loop_step(
            LoopPhase.MECHANICAL_CHECK,
            name="Mechanical answer checks",
            input_summary="answer plus prompt citations",
            retry_count=retry_count,
        )
        guardrail_decision = self._prepare_loop_step(run, planned_mechanical_step)
        if guardrail_decision:
            return run, guardrail_decision, None

        mechanical_check = self._mechanical_self_check_answer(
            answer,
            citations,
            question=question,
            retry_attempted=retry_attempted,
        )
        mechanical_step = self._loop_step(
            LoopPhase.MECHANICAL_CHECK,
            decision=self._loop_decision_for_self_check(mechanical_check),
            name="Mechanical answer checks",
            input_summary="answer plus prompt citations",
            output_summary=mechanical_check.outcome,
            retry_count=retry_count,
            metadata={
                "reasons": list(mechanical_check.reasons),
                "citation_count": len(citations),
                "inline_citation_ids": self._inline_citation_ids(answer),
                "retry_attempted": retry_attempted,
            },
        )
        run, guardrail_decision = self._record_loop_step(run, mechanical_step)
        if guardrail_decision or mechanical_check.outcome != "mechanical_checks_passed":
            return run, guardrail_decision, mechanical_check

        planned_verify_step = self._planned_loop_step(
            LoopPhase.VERIFY,
            name="Answer verifier",
            input_summary="answer plus cited excerpts",
            retry_count=retry_count,
        )
        guardrail_decision = self._prepare_loop_step(run, planned_verify_step)
        if guardrail_decision:
            return run, guardrail_decision, None

        if self._active_backend() == "mock":
            self_check = AnswerSelfCheck(
                outcome="not_verified",
                reasons=[
                    "mechanical_checks_passed",
                    "verifier_unavailable_mock_backend",
                ],
                retry_attempted=retry_attempted,
            )
        else:
            self_check = self._verify_answer_with_llm(
                question=question,
                answer=answer,
                citations=citations,
                retry_attempted=retry_attempted,
            )

        verify_step = self._loop_step(
            LoopPhase.VERIFY,
            decision=self._loop_decision_for_self_check(self_check),
            name="Answer verifier",
            input_summary="answer plus cited excerpts",
            output_summary=self_check.outcome,
            retry_count=retry_count,
            verification=self._verification_result_for_self_check(self_check),
            metadata={
                "reasons": list(self_check.reasons),
                "citation_count": len(citations),
                "retry_attempted": retry_attempted,
            },
        )
        run, guardrail_decision = self._record_loop_step(run, verify_step)
        return run, guardrail_decision, self_check

    def _guardrail_answer(self, decision: GuardrailDecision) -> str:
        if decision.decision == LoopDecision.REQUIRES_REVIEW:
            return "This query requires human review before the loop can continue."
        if decision.decision == LoopDecision.REFUSE:
            return "A loop guardrail refused this query before it could safely complete."
        if decision.decision == LoopDecision.RETRY:
            return "A loop guardrail requested a retry, but no safe retry path is available for this step."
        return "A loop guardrail blocked this query before it could complete."

    def _terminal_decision_for_guardrail(
        self, decision: GuardrailDecision
    ) -> LoopDecision:
        if decision.decision == LoopDecision.RETRY:
            return LoopDecision.BLOCK
        return decision.decision

    def _error_message_for_guardrail(self, decision: GuardrailDecision) -> str:
        if decision.decision == LoopDecision.RETRY:
            return "guardrail_retry_unavailable"
        return decision.reason or decision.decision.value

    def _guardrail_step(self, decision: GuardrailDecision) -> LoopStep:
        return self._loop_step(
            LoopPhase.ERROR,
            decision=decision.decision,
            name="Guardrail decision",
            output_summary=decision.reason or decision.decision.value,
            error_message=decision.reason,
            human_review=decision.human_review,
            metadata={
                "guardrail_decision": decision.decision.value,
                "guardrail_reason": decision.reason,
                **dict(decision.metadata),
            },
        )

    def _finish_loop_report(
        self,
        *,
        run: LoopRun,
        answer: str,
        final_decision: LoopDecision,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[
        LoopReport,
        str,
        LoopDecision,
        Optional[str],
        Optional[GuardrailDecision],
    ]:
        applied_guardrail_decision = None
        final_step = self._loop_step(
            LoopPhase.FINAL,
            decision=final_decision,
            name="Final answer",
            output_summary=final_decision.value,
            error_message=error_message,
            metadata=metadata or {},
        )
        run, guardrail_decision = self._append_loop_step(run, final_step)
        if guardrail_decision:
            applied_guardrail_decision = guardrail_decision
            answer = self._guardrail_answer(guardrail_decision)
            final_decision = self._terminal_decision_for_guardrail(
                guardrail_decision
            )
            error_message = self._error_message_for_guardrail(guardrail_decision)
            run = run.with_step(self._guardrail_step(guardrail_decision))

        completed_run = run.complete(
            final_decision=final_decision,
            final_answer=answer,
            error_message=error_message,
            metadata=metadata or {},
        )
        after_run_decision = self._run_loop_middleware("after_run", completed_run)
        if after_run_decision:
            applied_guardrail_decision = after_run_decision
            answer = self._guardrail_answer(after_run_decision)
            final_decision = self._terminal_decision_for_guardrail(
                after_run_decision
            )
            error_message = self._error_message_for_guardrail(after_run_decision)
            completed_run = completed_run.with_step(
                self._guardrail_step(after_run_decision)
            ).complete(
                final_decision=final_decision,
                final_answer=answer,
                error_message=error_message,
                metadata={
                    **dict(metadata or {}),
                    "after_run_guardrail": True,
                },
            )
        return (
            LoopReport(run=completed_run),
            answer,
            final_decision,
            error_message,
            applied_guardrail_decision,
        )

    def _finish_query_result(
        self,
        *,
        answer: str,
        question: str,
        active_state: ActiveDocumentState,
        run: LoopRun,
        final_decision: LoopDecision,
        retrieved_chunk_count: int = 0,
        citations: Optional[List[AnswerCitation]] = None,
        self_check: Optional[AnswerSelfCheck] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        (
            loop_report,
            answer,
            final_decision,
            error_message,
            applied_guardrail_decision,
        ) = self._finish_loop_report(
            run=run,
            answer=answer,
            final_decision=final_decision,
            error_message=error_message,
            metadata=metadata,
        )
        if applied_guardrail_decision:
            retrieved_chunk_count = 0
            citations = []
            self_check = None
        result = self._query_result(
            answer=answer,
            question=question,
            active_state=active_state,
            retrieved_chunk_count=retrieved_chunk_count,
            citations=citations,
            self_check=self_check,
            error_message=error_message,
            loop_report=loop_report,
        )
        self._record_loop_report(loop_report)
        return result

    def _finish_guardrail_query_result(
        self,
        *,
        decision: GuardrailDecision,
        question: str,
        active_state: ActiveDocumentState,
        run: LoopRun,
    ) -> QueryResult:
        run = run.with_step(self._guardrail_step(decision))
        return self._finish_query_result(
            answer=self._guardrail_answer(decision),
            question=question,
            active_state=active_state,
            run=run,
            final_decision=self._terminal_decision_for_guardrail(decision),
            error_message=self._error_message_for_guardrail(decision),
            metadata={"guardrail_decision": decision.decision.value},
        )

    def query_with_trace(self, prompt: str, session_id: str = "default") -> QueryResult:
        """Answer a user query and return the retrieved evidence used."""
        active_state = self._snapshot_active_document_state()
        session_id = self._normalize_session_id(session_id)
        clean_prompt = (prompt or "").strip()
        run = self._start_loop_run(
            prompt=clean_prompt,
            session_id=session_id,
            active_state=active_state,
        )

        guardrail_decision = self._run_loop_middleware("before_run", run)
        if guardrail_decision:
            return self._finish_guardrail_query_result(
                decision=guardrail_decision,
                question=clean_prompt,
                active_state=active_state,
                run=run,
            )

        if not clean_prompt:
            run, guardrail_decision = self._append_loop_step(
                run,
                self._loop_step(
                    LoopPhase.INPUT,
                    decision=LoopDecision.BLOCK,
                    name="Input validation",
                    output_summary="empty_question",
                    error_message="empty_question",
                ),
            )
            if guardrail_decision:
                return self._finish_guardrail_query_result(
                    decision=guardrail_decision,
                    question=clean_prompt,
                    active_state=active_state,
                    run=run,
                )
            return self._finish_query_result(
                answer="Please provide a question.",
                question=clean_prompt,
                active_state=active_state,
                run=run,
                final_decision=LoopDecision.BLOCK,
                error_message="empty_question",
            )

        if not active_state.retrieval_chain:
            run, guardrail_decision = self._append_loop_step(
                run,
                self._loop_step(
                    LoopPhase.CONTEXT_SELECT,
                    decision=LoopDecision.BLOCK,
                    name="Select document context",
                    output_summary="document_not_loaded",
                    error_message="document_not_loaded",
                    metadata={"document_name": active_state.document_name},
                ),
            )
            if guardrail_decision:
                return self._finish_guardrail_query_result(
                    decision=guardrail_decision,
                    question=clean_prompt,
                    active_state=active_state,
                    run=run,
                )
            return self._finish_query_result(
                answer="Please upload and process a document first.",
                question=clean_prompt,
                active_state=active_state,
                run=run,
                final_decision=LoopDecision.BLOCK,
                error_message="document_not_loaded",
            )

        run, guardrail_decision = self._append_loop_step(
            run,
            self._loop_step(
                LoopPhase.CONTEXT_SELECT,
                decision=LoopDecision.CONTINUE,
                name="Select document context",
                output_summary=active_state.document_name,
                metadata={"document_name": active_state.document_name},
            ),
        )
        if guardrail_decision:
            return self._finish_guardrail_query_result(
                decision=guardrail_decision,
                question=clean_prompt,
                active_state=active_state,
                run=run,
            )

        if not self.llm:
            run, guardrail_decision = self._append_loop_step(
                run,
                self._loop_step(
                    LoopPhase.ERROR,
                    decision=LoopDecision.ERROR,
                    name="LLM readiness",
                    output_summary="llm_not_initialized",
                    error_message="llm_not_initialized",
                ),
            )
            if guardrail_decision:
                return self._finish_guardrail_query_result(
                    decision=guardrail_decision,
                    question=clean_prompt,
                    active_state=active_state,
                    run=run,
                )
            return self._finish_query_result(
                answer=(
                    "Language model is not initialized. Please check your LLM "
                    "backend setup."
                ),
                question=clean_prompt,
                active_state=active_state,
                run=run,
                final_decision=LoopDecision.ERROR,
                error_message="llm_not_initialized",
            )

        if active_state.document_name and self._is_document_identity_question(clean_prompt):
            return self._finish_query_result(
                answer=f"The uploaded document is `{active_state.document_name}`.",
                question=clean_prompt,
                active_state=active_state,
                run=run,
                final_decision=LoopDecision.FINAL,
                metadata={"identity_answer": True},
            )

        try:
            if hasattr(active_state.retrieval_chain, "invoke_with_trace"):
                supports_split_trace = (
                    hasattr(active_state.retrieval_chain, "retrieve_with_trace")
                    and hasattr(active_state.retrieval_chain, "draft_with_trace")
                )
                planned_retrieve_step = self._planned_loop_step(
                    LoopPhase.RETRIEVE,
                    name="Retrieve prompt evidence",
                    input_summary=clean_prompt,
                )
                guardrail_decision = self._prepare_loop_step(
                    run, planned_retrieve_step
                )
                if guardrail_decision:
                    return self._finish_guardrail_query_result(
                        decision=guardrail_decision,
                        question=clean_prompt,
                        active_state=active_state,
                        run=run,
                    )
                if supports_split_trace:
                    retrieved_context = active_state.retrieval_chain.retrieve_with_trace(
                        clean_prompt
                    )
                    retrieved_chunk_count = retrieved_context.retrieved_chunk_count
                    citations = retrieved_context.citations
                    run, guardrail_decision = self._record_loop_step(
                        run,
                        self._loop_step(
                            LoopPhase.RETRIEVE,
                            decision=LoopDecision.CONTINUE,
                            name="Retrieve prompt evidence",
                            input_summary=clean_prompt,
                            output_summary=(f"{retrieved_chunk_count} prompt chunks"),
                            metadata={
                                "retrieved_chunk_count": retrieved_chunk_count,
                                "citation_ids": [
                                    citation.citation_id for citation in citations
                                ],
                                "context_chars": len(retrieved_context.context),
                            },
                        ),
                    )
                    if guardrail_decision:
                        return self._finish_guardrail_query_result(
                            decision=guardrail_decision,
                            question=clean_prompt,
                            active_state=active_state,
                            run=run,
                        )
                    planned_draft_step = self._planned_loop_step(
                        LoopPhase.DRAFT,
                        name="Draft answer",
                        input_summary=clean_prompt,
                    )
                    guardrail_decision = self._prepare_loop_step(
                        run, planned_draft_step
                    )
                    if guardrail_decision:
                        return self._finish_guardrail_query_result(
                            decision=guardrail_decision,
                            question=clean_prompt,
                            active_state=active_state,
                            run=run,
                        )
                    chain_result = active_state.retrieval_chain.draft_with_trace(
                        clean_prompt,
                        retrieved_context,
                    )
                else:
                    planned_draft_step = self._planned_loop_step(
                        LoopPhase.DRAFT,
                        name="Draft answer",
                        input_summary=clean_prompt,
                    )
                    guardrail_decision = self._prepare_loop_step(
                        run, planned_draft_step
                    )
                    if guardrail_decision:
                        return self._finish_guardrail_query_result(
                            decision=guardrail_decision,
                            question=clean_prompt,
                            active_state=active_state,
                            run=run,
                        )
                    chain_result = active_state.retrieval_chain.invoke_with_trace(
                        clean_prompt
                    )
                    retrieved_chunk_count = chain_result.retrieved_chunk_count
                    citations = chain_result.citations

                response = chain_result.answer
                retrieved_chunk_count = chain_result.retrieved_chunk_count
                citations = chain_result.citations
                draft_metadata = {
                    "answer_chars": len(str(response).strip()),
                    "inline_citation_ids": self._inline_citation_ids(str(response)),
                }
                if not supports_split_trace:
                    draft_metadata.update(
                        {
                            "combined_retrieve_and_draft": True,
                            "retrieved_chunk_count": retrieved_chunk_count,
                            "citation_ids": [
                                citation.citation_id for citation in citations
                            ],
                            "context_chars": len(
                                getattr(chain_result, "context", "") or ""
                            ),
                        }
                    )
                run, guardrail_decision = self._record_loop_step(
                    run,
                    self._loop_step(
                        LoopPhase.DRAFT,
                        decision=LoopDecision.CONTINUE,
                        name="Draft answer",
                        input_summary=clean_prompt,
                        output_summary=str(response).strip()[:500],
                        metadata=draft_metadata,
                    ),
                )
                if guardrail_decision:
                    return self._finish_guardrail_query_result(
                        decision=guardrail_decision,
                        question=clean_prompt,
                        active_state=active_state,
                        run=run,
                    )
                run, guardrail_decision, self_check = self._run_self_check_with_loop(
                    run=run,
                    answer=response,
                    citations=citations,
                    question=clean_prompt,
                )
                if guardrail_decision:
                    return self._finish_guardrail_query_result(
                        decision=guardrail_decision,
                        question=clean_prompt,
                        active_state=active_state,
                        run=run,
                    )
                if self_check is None:
                    self_check = AnswerSelfCheck(
                        outcome="needs_refusal",
                        reasons=["self_check_interrupted"],
                    )
                if self_check.outcome == "needs_retry" and hasattr(
                    active_state.retrieval_chain, "retry_with_trace"
                ):
                    run, guardrail_decision = self._append_loop_step(
                        run,
                        self._loop_step(
                            LoopPhase.RETRY,
                            decision=LoopDecision.RETRY,
                            name="Retry answer",
                            output_summary="retrying after self-check failure",
                            metadata={"reasons": list(self_check.reasons)},
                        ),
                    )
                    if guardrail_decision:
                        return self._finish_guardrail_query_result(
                            decision=guardrail_decision,
                            question=clean_prompt,
                            active_state=active_state,
                            run=run,
                        )
                    planned_retry_draft_step = self._planned_loop_step(
                        LoopPhase.DRAFT,
                        name="Draft retry answer",
                        input_summary=clean_prompt,
                        retry_count=1,
                    )
                    guardrail_decision = self._prepare_loop_step(
                        run, planned_retry_draft_step
                    )
                    if guardrail_decision:
                        return self._finish_guardrail_query_result(
                            decision=guardrail_decision,
                            question=clean_prompt,
                            active_state=active_state,
                            run=run,
                        )
                    retry_result = active_state.retrieval_chain.retry_with_trace(
                        clean_prompt,
                        chain_result,
                        self_check_instruction=self._self_check_retry_instruction(
                            self_check
                        ),
                    )
                    response = retry_result.answer
                    retrieved_chunk_count = retry_result.retrieved_chunk_count
                    citations = retry_result.citations
                    run, guardrail_decision = self._record_loop_step(
                        run,
                        self._loop_step(
                            LoopPhase.DRAFT,
                            decision=LoopDecision.CONTINUE,
                            name="Draft retry answer",
                            input_summary=clean_prompt,
                            output_summary=str(response).strip()[:500],
                            retry_count=1,
                            metadata={
                                "answer_chars": len(str(response).strip()),
                                "inline_citation_ids": self._inline_citation_ids(
                                    str(response)
                                ),
                            },
                        ),
                    )
                    if guardrail_decision:
                        return self._finish_guardrail_query_result(
                            decision=guardrail_decision,
                            question=clean_prompt,
                            active_state=active_state,
                            run=run,
                        )
                    (
                        run,
                        guardrail_decision,
                        self_check,
                    ) = self._run_self_check_with_loop(
                        run=run,
                        answer=response,
                        citations=citations,
                        question=clean_prompt,
                        retry_attempted=True,
                    )
                    if guardrail_decision:
                        return self._finish_guardrail_query_result(
                            decision=guardrail_decision,
                            question=clean_prompt,
                            active_state=active_state,
                            run=run,
                        )
                    if self_check is None:
                        self_check = AnswerSelfCheck(
                            outcome="needs_refusal",
                            reasons=["self_check_interrupted"],
                            retry_attempted=True,
                        )
            else:
                planned_draft_step = self._planned_loop_step(
                    LoopPhase.DRAFT,
                    name="Draft answer",
                    input_summary=clean_prompt,
                )
                guardrail_decision = self._prepare_loop_step(run, planned_draft_step)
                if guardrail_decision:
                    return self._finish_guardrail_query_result(
                        decision=guardrail_decision,
                        question=clean_prompt,
                        active_state=active_state,
                        run=run,
                    )
                response = active_state.retrieval_chain.invoke(clean_prompt)
                retrieved_chunk_count = 0
                citations = []
                self_check = None
                run, guardrail_decision = self._record_loop_step(
                    run,
                    self._loop_step(
                        LoopPhase.DRAFT,
                        decision=LoopDecision.CONTINUE,
                        name="Draft answer",
                        input_summary=clean_prompt,
                        output_summary=str(response).strip()[:500],
                        metadata={"trace_available": False},
                    ),
                )
                if guardrail_decision:
                    return self._finish_guardrail_query_result(
                        decision=guardrail_decision,
                        question=clean_prompt,
                        active_state=active_state,
                        run=run,
                    )

            response = str(response).strip()
            if len(response) < 3:
                response = SELF_CHECK_REFUSAL_ANSWER
                run, guardrail_decision, self_check = self._run_self_check_with_loop(
                    run=run,
                    answer=response,
                    citations=citations,
                    question=clean_prompt,
                )
                if guardrail_decision:
                    return self._finish_guardrail_query_result(
                        decision=guardrail_decision,
                        question=clean_prompt,
                        active_state=active_state,
                        run=run,
                    )
                if self_check is None:
                    self_check = AnswerSelfCheck(
                        outcome="needs_refusal",
                        reasons=["self_check_interrupted"],
                    )

            if self_check and self_check.outcome not in SELF_CHECK_PASS_OUTCOMES:
                response = SELF_CHECK_REFUSAL_ANSWER
                if self_check.outcome != "needs_refusal":
                    self_check = self._fail_closed_self_check(self_check)
                run, guardrail_decision = self._append_loop_step(
                    run,
                    self._loop_step(
                        LoopPhase.REFUSE,
                        decision=LoopDecision.REFUSE,
                        name="Refuse unsupported answer",
                        output_summary=self_check.outcome,
                        metadata={
                            "reasons": list(self_check.reasons),
                            "retry_attempted": self_check.retry_attempted,
                        },
                    ),
                )
                if guardrail_decision:
                    return self._finish_guardrail_query_result(
                        decision=guardrail_decision,
                        question=clean_prompt,
                        active_state=active_state,
                        run=run,
                    )

            final_decision = (
                self._loop_decision_for_self_check(self_check)
                if self_check
                else LoopDecision.FINAL
            )
            if final_decision == LoopDecision.CONTINUE:
                final_decision = LoopDecision.FINAL
            result = self._finish_query_result(
                answer=response,
                question=clean_prompt,
                active_state=active_state,
                run=run,
                final_decision=final_decision,
                retrieved_chunk_count=retrieved_chunk_count,
                citations=citations,
                self_check=self_check,
                metadata={
                    "self_check_outcome": self_check.outcome if self_check else None,
                    "retry_attempted": (
                        self_check.retry_attempted if self_check else False
                    ),
                },
            )

            self.chat_history.append(
                {
                    "session_id": session_id,
                    "question": clean_prompt,
                    "answer": result.answer,
                    "citations": [
                        {
                            "id": citation.citation_id,
                            "source_name": citation.source_name,
                            "page": citation.page,
                            "chunk_index": citation.chunk_index,
                            "excerpt": citation.excerpt,
                        }
                        for citation in result.trace.citations
                    ],
                    "self_check": (
                        {
                            "outcome": result.trace.self_check.outcome,
                            "reasons": result.trace.self_check.reasons,
                            "retry_attempted": result.trace.self_check.retry_attempted,
                        }
                        if result.trace.self_check
                        else None
                    ),
                }
            )
            return result
        except Exception as exc:
            LOGGER.exception("Error while processing query.")
            error_decision = self._run_loop_middleware("on_error", run, exc)
            if error_decision:
                return self._finish_guardrail_query_result(
                    decision=error_decision,
                    question=clean_prompt,
                    active_state=active_state,
                    run=run,
                )
            run = run.with_step(
                self._loop_step(
                    LoopPhase.ERROR,
                    decision=LoopDecision.ERROR,
                    name="Query error",
                    output_summary="query_failed",
                    error_message="query_failed",
                    metadata={"error_type": exc.__class__.__name__},
                )
            )
            return self._finish_query_result(
                answer="I hit an internal error while processing your question. Please try again.",
                question=clean_prompt,
                active_state=active_state,
                run=run,
                final_decision=LoopDecision.ERROR,
                error_message="query_failed",
            )

    def query(self, prompt: str, session_id: str = "default") -> str:
        """Answer a user query using retrieved document context."""
        return self.query_with_trace(prompt, session_id=session_id).answer


# Backward-compatible class name. New code should import AILoopEngine from
# src.ai_loop_engine, but existing callers may still use DocumentQA.
DocumentQA = AILoopEngine
