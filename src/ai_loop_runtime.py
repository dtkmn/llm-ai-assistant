from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

try:
    from .native_runtime import apply_native_runtime_defaults
except ImportError:
    from native_runtime import apply_native_runtime_defaults


apply_native_runtime_defaults()

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM

try:
    from .context_providers import (
        ActiveDocumentState,
        ContextProvider,
        DocumentContextProvider,
    )
except ImportError:
    from context_providers import (
        ActiveDocumentState,
        ContextProvider,
        DocumentContextProvider,
    )

try:
    from .document_config import (
        ALLOWED_TEXT_CONTROL_CHARACTERS,
        COMMON_WESTERN_PUNCTUATION,
        COMMON_WESTERN_SYMBOLS,
        CONFLICTING_LEGACY_ENCODINGS,
        DETECTED_ENCODING_CHAOS_GAP,
        DETECTED_ENCODING_MAX_CHAOS,
        DETECTED_ENCODING_MIN_COHERENCE,
        MAX_DOCUMENT_BYTES,
        MAX_DOCUMENT_CHUNKS,
        MAX_SHORT_WESTERN_FALLBACK_BYTES,
        SUPPORTED_EXTENSIONS,
        TEXT_ENCODING_ALIASES,
        TEXT_ENCODING_BOMS,
        TEXT_ENCODING_FALLBACKS,
        UTF_NUL_FAMILY_ENCODINGS,
        UTF_NUL_PATTERN_THRESHOLD,
        has_binary_control_characters,
        is_latin_letter,
        latin_token_is_suspicious,
        normalize_encoding_name,
        nul_ratio,
        validate_document,
        western_text_penalty,
    )
except ImportError:
    from document_config import (
        ALLOWED_TEXT_CONTROL_CHARACTERS,
        COMMON_WESTERN_PUNCTUATION,
        COMMON_WESTERN_SYMBOLS,
        CONFLICTING_LEGACY_ENCODINGS,
        DETECTED_ENCODING_CHAOS_GAP,
        DETECTED_ENCODING_MAX_CHAOS,
        DETECTED_ENCODING_MIN_COHERENCE,
        MAX_DOCUMENT_BYTES,
        MAX_DOCUMENT_CHUNKS,
        MAX_SHORT_WESTERN_FALLBACK_BYTES,
        SUPPORTED_EXTENSIONS,
        TEXT_ENCODING_ALIASES,
        TEXT_ENCODING_BOMS,
        TEXT_ENCODING_FALLBACKS,
        UTF_NUL_FAMILY_ENCODINGS,
        UTF_NUL_PATTERN_THRESHOLD,
        has_binary_control_characters,
        is_latin_letter,
        latin_token_is_suspicious,
        normalize_encoding_name,
        nul_ratio,
        validate_document,
        western_text_penalty,
    )

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
        MODEL_THINKING_ENV_VAR,
        OLLAMA_BASE_URL_ENV_VAR,
        OLLAMA_EMBEDDINGS_MODEL_ENV_VAR,
        OLLAMA_MODEL_ENV_VAR,
        OLLAMA_THINK_LEVEL_ENV_VAR,
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
        normalize_ollama_base_url,
        normalize_ollama_think_level,
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
        MODEL_THINKING_ENV_VAR,
        OLLAMA_BASE_URL_ENV_VAR,
        OLLAMA_EMBEDDINGS_MODEL_ENV_VAR,
        OLLAMA_MODEL_ENV_VAR,
        OLLAMA_THINK_LEVEL_ENV_VAR,
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
        normalize_ollama_base_url,
        normalize_ollama_think_level,
        normalize_openai_compatible_base_url,
        safe_ollama_base_url_for_error,
        safe_openai_compatible_base_url_for_error,
    )

try:
    from . import answer_loop as _answer_loop
    from .answer_loop import (
        ANSWER_SUPPORT_STOPWORDS,
        SELF_CHECK_PASS_OUTCOMES,
        SELF_CHECK_REFUSAL_ANSWER,
        VERIFIER_OUTCOMES,
        AnswerSelfCheck,
    )
except ImportError:
    import answer_loop as _answer_loop
    from answer_loop import (
        ANSWER_SUPPORT_STOPWORDS,
        SELF_CHECK_PASS_OUTCOMES,
        SELF_CHECK_REFUSAL_ANSWER,
        VERIFIER_OUTCOMES,
        AnswerSelfCheck,
    )

try:
    from .model_adapters import (
        MockLLM,
        OLLAMA_NO_PROXY_OPENER,
        OPENAI_COMPAT_NO_PROXY_OPENER,
        OllamaEmbeddings,
        OllamaLLM,
        OpenAICompatibleEmbeddings,
        OpenAICompatibleLLM,
        open_ollama_request_no_proxy,
        open_openai_compatible_request,
    )
except ImportError:
    from model_adapters import (
        MockLLM,
        OLLAMA_NO_PROXY_OPENER,
        OPENAI_COMPAT_NO_PROXY_OPENER,
        OllamaEmbeddings,
        OllamaLLM,
        OpenAICompatibleEmbeddings,
        OpenAICompatibleLLM,
        open_ollama_request_no_proxy,
        open_openai_compatible_request,
    )

try:
    from .retrieval_types import AnswerCitation, RetrievalChainResult, RetrievedContext
except ImportError:
    from retrieval_types import AnswerCitation, RetrievalChainResult, RetrievedContext

LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_SESSION_REPORTS = 200
_RETRIEVAL_EXPORTS = {
    "DocumentRetrievalChain",
    "FaissRetriever",
    "FaissVectorStore",
    "build_document_retrieval_chain",
}
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
SELF_CHECK_REFUSAL_ANSWER = _answer_loop.SELF_CHECK_REFUSAL_ANSWER
NO_CONTEXT_SELF_CHECK_REASONS = [
    "no_context_provider",
    "verifier_requires_prompt_evidence",
]
DIRECT_STANDALONE_CITATION_MARKER_PATTERN = re.compile(
    r"(?<!\S)\[(\d+)\](?=$|[\s.,;:!?)])"
)


def _document_ingestion_module():
    try:
        from . import document_ingestion
    except ImportError:
        import document_ingestion

    return document_ingestion


def _document_text_module():
    try:
        from . import document_text
    except ImportError:
        import document_text

    return document_text


def _retrieval_module():
    try:
        from . import retrieval
    except ImportError:
        import retrieval

    return retrieval


def _retrieval_export(name: str):
    if name in globals():
        return globals()[name]
    return getattr(_retrieval_module(), name)


def __getattr__(name):
    if name in _RETRIEVAL_EXPORTS:
        value = _retrieval_export(name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def decode_text_file(
    document_path: str, text_encoding: Optional[str] = None
) -> str:
    return _document_text_module().decode_text_file(document_path, text_encoding)


def decode_supported_text(raw_content: bytes, encoding: str) -> str:
    return _document_text_module().decode_supported_text(raw_content, encoding)


def decode_utf8_or_western_text(raw_content: bytes) -> str:
    return _document_text_module().decode_utf8_or_western_text(raw_content)


def decode_confident_detected_text(raw_content: bytes, matches) -> Optional[str]:
    return _document_text_module().decode_confident_detected_text(raw_content, matches)


def fallback_encoding_is_plausible(
    raw_content: bytes, matches, encoding: str, fallback_text: str
) -> bool:
    return _document_text_module().fallback_encoding_is_plausible(
        raw_content, matches, encoding, fallback_text
    )


def detected_encoding_is_confident(matches, index: int) -> bool:
    return _document_text_module().detected_encoding_is_confident(matches, index)


def decode_nul_pattern_text(raw_content: bytes) -> Optional[str]:
    return _document_text_module().decode_nul_pattern_text(raw_content)


def nul_pattern_encodings(raw_content: bytes) -> List[str]:
    return _document_text_module().nul_pattern_encodings(raw_content)


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
class AnswerTrace:
    question: str
    document_name: Optional[str]
    backend: str
    model_label: str
    retrieved_chunk_count: int
    citations: List[AnswerCitation]
    self_check: Optional["AnswerSelfCheck"] = None
    error_message: Optional[str] = None
    model_thinking: Optional[str] = None


@dataclass(frozen=True)
class QueryResult:
    answer: str
    trace: AnswerTrace
    loop_report: Optional[LoopReport] = None


class DocumentProcessingError(RuntimeError):
    def __init__(self, message: str, status: DocumentQAStatus):
        super().__init__(message)
        self.status = status


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


class EngineTextDecoder:
    def __init__(self, engine: "AILoopEngine"):
        self.engine = engine

    def decode_supported_text(self, raw_content: bytes, encoding: str) -> str:
        return self.engine._decode_supported_text(raw_content, encoding)

    def decode_utf8_or_western_text(self, raw_content: bytes) -> str:
        return self.engine._decode_utf8_or_western_text(raw_content)

    def decode_confident_detected_text(
        self, raw_content: bytes, matches
    ) -> Optional[str]:
        return self.engine._decode_confident_detected_text(raw_content, matches)

    def fallback_encoding_is_plausible(
        self, raw_content: bytes, matches, encoding: str, fallback_text: str
    ) -> bool:
        return self.engine._fallback_encoding_is_plausible(
            raw_content, matches, encoding, fallback_text
        )

    def detected_encoding_is_confident(self, matches, index: int) -> bool:
        return self.engine._detected_encoding_is_confident(matches, index)

    def decode_nul_pattern_text(self, raw_content: bytes) -> Optional[str]:
        return self.engine._decode_nul_pattern_text(raw_content)

    def nul_pattern_encodings(self, raw_content: bytes) -> List[str]:
        return self.engine._nul_pattern_encodings(raw_content)


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
        self.model_thinking_enabled = env_flag(MODEL_THINKING_ENV_VAR, True)
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
        think_level = (
            normalize_ollama_think_level(
                os.getenv(OLLAMA_THINK_LEVEL_ENV_VAR),
                model=model_id,
            )
            if self.model_thinking_enabled
            else None
        )
        llm = OllamaLLM(
            model=model_id,
            base_url=base_url,
            timeout=self.ollama_timeout,
            enable_thinking=self.model_thinking_enabled,
            think_level=think_level,
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
        vector_store: Any,
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
        self, vector_store: Any, document_name: Optional[str]
    ):
        if not self.llm:
            raise RuntimeError("LLM is not initialized.")
        if not vector_store:
            raise RuntimeError("Vector store is not initialized.")

        return _retrieval_export("build_document_retrieval_chain")(
            vector_store=vector_store,
            llm=self.llm,
            document_name=document_name,
            profile=self.profile,
        )

    def _validate_document(self, document_path: str) -> str:
        return validate_document(document_path, self.max_document_bytes)

    def _load_documents(
        self,
        document_path: str,
        file_extension: str,
        text_encoding: Optional[str] = None,
    ) -> List[Any]:
        return _document_ingestion_module().load_documents(
            document_path,
            file_extension,
            text_encoding=text_encoding,
            decode_text=self._decode_text_file,
        )

    def _decode_text_file(
        self, document_path: str, text_encoding: Optional[str] = None
    ) -> str:
        return _document_text_module().DefaultTextDecoder.decode_text_file(
            EngineTextDecoder(self), document_path, text_encoding
        )

    def _decode_supported_text(self, raw_content: bytes, encoding: str) -> str:
        return decode_supported_text(raw_content, encoding)

    def _decode_utf8_or_western_text(self, raw_content: bytes) -> str:
        return _document_text_module().DefaultTextDecoder.decode_utf8_or_western_text(
            EngineTextDecoder(self), raw_content
        )

    def _decode_confident_detected_text(
        self, raw_content: bytes, matches
    ) -> Optional[str]:
        return _document_text_module().DefaultTextDecoder.decode_confident_detected_text(
            EngineTextDecoder(self), raw_content, matches
        )

    def _fallback_encoding_is_plausible(
        self, raw_content: bytes, matches, encoding: str, fallback_text: str
    ) -> bool:
        return _document_text_module().DefaultTextDecoder.fallback_encoding_is_plausible(
            EngineTextDecoder(self), raw_content, matches, encoding, fallback_text
        )

    def _detected_encoding_is_confident(self, matches, index: int) -> bool:
        return detected_encoding_is_confident(matches, index)

    def _decode_nul_pattern_text(self, raw_content: bytes) -> Optional[str]:
        return _document_text_module().DefaultTextDecoder.decode_nul_pattern_text(
            EngineTextDecoder(self), raw_content
        )

    def _nul_pattern_encodings(self, raw_content: bytes) -> List[str]:
        return nul_pattern_encodings(raw_content)

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
                split_result = _document_ingestion_module().split_document_chunks(
                    documents,
                    self.profile,
                    self.max_document_chunks,
                )
                chunks = split_result.chunks
                truncated = split_result.truncated
                chunk_count = len(chunks)

                phase = "index"
                vector_store = _retrieval_export("FaissVectorStore").from_documents(
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
        return _answer_loop.answer_is_refusal(answer)

    def _inline_citation_ids(self, answer: str) -> List[int]:
        return _answer_loop.inline_citation_ids(answer)

    def _citation_ids_are_valid(
        self, inline_citation_ids: List[int], citations: List[AnswerCitation]
    ) -> bool:
        return _answer_loop.citation_ids_are_valid(inline_citation_ids, citations)

    def _cited_citations_for_answer(
        self, answer: str, citations: List[AnswerCitation]
    ) -> List[AnswerCitation]:
        return _answer_loop.cited_citations_for_answer(answer, citations)

    def _support_token(self, token: str) -> str:
        return _answer_loop.support_token(token)

    def _support_tokens(self, text: str) -> set:
        return _answer_loop.support_tokens(text)

    def _normalize_support_text(self, text: str) -> str:
        return _answer_loop.normalize_support_text(text)

    def _matched_claim_is_denied(
        self,
        normalized_answer: str,
        normalized_evidence: str,
        raw_evidence: str = "",
    ) -> bool:
        return _answer_loop.matched_claim_is_denied(
            normalized_answer, normalized_evidence, raw_evidence
        )

    def _citation_text_refutes_answer(
        self, answer: str, citations: List[AnswerCitation], question: str
    ) -> bool:
        return _answer_loop.citation_text_refutes_answer(answer, citations, question)

    def _mechanical_self_check_answer(
        self,
        answer: str,
        citations: List[AnswerCitation],
        *,
        question: str = "",
        retry_attempted: bool = False,
    ) -> AnswerSelfCheck:
        return _answer_loop.mechanical_self_check_answer(
            answer,
            citations,
            question=question,
            retry_attempted=retry_attempted,
        )

    def _loop_decision_for_self_check(self, self_check: AnswerSelfCheck) -> LoopDecision:
        return _answer_loop.loop_decision_for_self_check(self_check)

    def _verification_result_for_self_check(
        self, self_check: AnswerSelfCheck
    ) -> VerificationResult:
        return _answer_loop.verification_result_for_self_check(
            self_check, verifier=self._active_backend()
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
        return _answer_loop.verifier_prompt(
            question=question, answer=answer, citations=citations
        )

    def _parse_verifier_response(self, raw_response: str) -> Tuple[Optional[str], str]:
        return _answer_loop.parse_verifier_response(raw_response)

    def _verify_answer_with_llm(
        self,
        *,
        question: str,
        answer: str,
        citations: List[AnswerCitation],
        retry_attempted: bool,
    ) -> AnswerSelfCheck:
        return _answer_loop.verify_answer_with_llm(
            llm=self.llm,
            question=question,
            answer=answer,
            citations=citations,
            retry_attempted=retry_attempted,
            logger=LOGGER,
        )

    def _fail_closed_self_check(self, self_check: AnswerSelfCheck) -> AnswerSelfCheck:
        return _answer_loop.fail_closed_self_check(self_check)

    def _self_check_retry_instruction(self, self_check: AnswerSelfCheck) -> str:
        return _answer_loop.self_check_retry_instruction(self_check)

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
        model_thinking: Optional[str] = None,
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
                model_thinking=model_thinking,
            ),
            loop_report=loop_report,
        )

    def _active_context_provider_for_run(
        self, active_state: ActiveDocumentState
    ) -> Optional[ContextProvider]:
        context_provider = active_state.context_provider
        if (
            active_state.retrieval_chain
            and context_provider
            and getattr(context_provider, "ready", bool(active_state.retrieval_chain))
        ):
            return context_provider
        return None

    def _context_provider_type_for_run(
        self, active_state: ActiveDocumentState
    ) -> str:
        context_provider = self._active_context_provider_for_run(active_state)
        return context_provider.provider_type if context_provider else "none"

    def _direct_answer_prompt(self, question: str) -> str:
        return (
            "You are AI Loop Engine running without an external context provider. "
            "Answer the user's question directly. Be concise, do not invent "
            "citations, and say when you are unsure.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def _strip_direct_answer_citations(self, answer: str) -> str:
        cleaned = DIRECT_STANDALONE_CITATION_MARKER_PATTERN.sub(
            "", answer or ""
        )
        cleaned = cleaned.strip()
        cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
        return re.sub(r"\s{2,}", " ", cleaned).strip()

    def _direct_citation_marker_ids(self, answer: str) -> List[int]:
        standalone_ids = [
            int(match.group(1))
            for match in DIRECT_STANDALONE_CITATION_MARKER_PATTERN.finditer(
                answer or ""
            )
        ]
        return standalone_ids

    def _last_model_thinking(self) -> Optional[str]:
        llm_thinking = getattr(self.llm, "last_thinking", None)
        return (
            llm_thinking.strip()
            if isinstance(llm_thinking, str) and llm_thinking.strip()
            else None
        )

    def _refresh_run_model_identity(self, run: LoopRun) -> LoopRun:
        return replace(
            run,
            backend=self._active_backend(),
            model_label=self._active_model_label(),
        )

    def _record_chat_history_entry(
        self, *, session_id: str, question: str, result: QueryResult
    ) -> None:
        self.chat_history.append(
            {
                "session_id": session_id,
                "question": question,
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

    def _start_loop_run(
        self,
        *,
        prompt: str,
        session_id: str,
        active_state: ActiveDocumentState,
    ) -> LoopRun:
        context_provider = self._active_context_provider_for_run(active_state)
        context_provider_type = self._context_provider_type_for_run(active_state)
        untrusted_inputs = ["model_output", "future_tool_output"]
        if context_provider_type == "document":
            untrusted_inputs = ["document_text", "retrieved_chunks", *untrusted_inputs]
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
                "untrusted_inputs": untrusted_inputs,
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
        model_thinking: Optional[str] = None,
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
            model_thinking = None
        result = self._query_result(
            answer=answer,
            question=question,
            active_state=active_state,
            retrieved_chunk_count=retrieved_chunk_count,
            citations=citations,
            self_check=self_check,
            error_message=error_message,
            loop_report=loop_report,
            model_thinking=model_thinking,
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

        if not self.llm:
            try:
                self._ensure_llm_initialized()
            except Exception as exc:
                LOGGER.exception("LLM initialization failed during query.")
                run, guardrail_decision = self._append_loop_step(
                    run,
                    self._loop_step(
                        LoopPhase.ERROR,
                        decision=LoopDecision.ERROR,
                        name="LLM readiness",
                        output_summary="llm_initialization_failed",
                        error_message="llm_initialization_failed",
                        metadata={"error_type": exc.__class__.__name__},
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
                        "Language model could not be initialized. Please check "
                        "your LLM backend setup."
                    ),
                    question=clean_prompt,
                    active_state=active_state,
                    run=run,
                    final_decision=LoopDecision.ERROR,
                    error_message="llm_initialization_failed",
                )
            run = self._refresh_run_model_identity(run)

        context_provider_type = self._context_provider_type_for_run(active_state)
        context_provider = self._active_context_provider_for_run(active_state)
        context_output = (
            active_state.document_name
            if context_provider_type == "document"
            else "no_context_provider"
        )
        run, guardrail_decision = self._append_loop_step(
            run,
            self._loop_step(
                LoopPhase.CONTEXT_SELECT,
                decision=LoopDecision.CONTINUE,
                name="Select context provider",
                output_summary=context_output,
                metadata={
                    "document_name": active_state.document_name,
                    "context_provider": context_provider_type,
                    "context_provider_name": (
                        context_provider.display_name if context_provider else None
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

        model_thinking = None
        direct_context = not active_state.retrieval_chain
        try:
            if direct_context:
                planned_draft_step = self._planned_loop_step(
                    LoopPhase.DRAFT,
                    name="Draft direct answer",
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
                raw_response = str(
                    self.llm.invoke(self._direct_answer_prompt(clean_prompt))
                ).strip()
                model_thinking = self._last_model_thinking()
                removed_inline_citation_ids = self._direct_citation_marker_ids(
                    raw_response
                )
                response = self._strip_direct_answer_citations(raw_response)
                if len(response) < 3:
                    model_thinking = None
                    error_metadata = {
                        "context_provider": "none",
                        "raw_answer_chars": len(raw_response),
                        "removed_inline_citation_ids": removed_inline_citation_ids,
                    }
                    run, guardrail_decision = self._record_loop_step(
                        run,
                        self._loop_step(
                            LoopPhase.DRAFT,
                            decision=LoopDecision.ERROR,
                            name="Draft direct answer",
                            input_summary=clean_prompt,
                            output_summary="empty_direct_answer",
                            error_message="empty_direct_answer",
                            metadata=error_metadata,
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
                            "The model returned an empty answer. Please try again "
                            "or check your LLM backend."
                        ),
                        question=clean_prompt,
                        active_state=active_state,
                        run=run,
                        final_decision=LoopDecision.ERROR,
                        error_message="empty_direct_answer",
                    )
                retrieved_chunk_count = 0
                citations = []
                self_check = AnswerSelfCheck(
                    outcome="not_verified",
                    reasons=list(NO_CONTEXT_SELF_CHECK_REASONS),
                    retry_attempted=False,
                )
                draft_metadata = {
                    "answer_chars": len(response),
                    "context_provider": "none",
                    "inline_citation_ids": self._direct_citation_marker_ids(response),
                    "model_thinking_available": bool(model_thinking),
                    "removed_inline_citation_ids": removed_inline_citation_ids,
                    "trace_available": True,
                }
                if model_thinking:
                    draft_metadata["model_thinking_chars"] = len(model_thinking)
                run, guardrail_decision = self._record_loop_step(
                    run,
                    self._loop_step(
                        LoopPhase.DRAFT,
                        decision=LoopDecision.CONTINUE,
                        name="Draft direct answer",
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
                verify_step = self._loop_step(
                    LoopPhase.VERIFY,
                    decision=LoopDecision.NOT_VERIFIED,
                    name="No-context verification boundary",
                    input_summary="answer without prompt evidence",
                    output_summary=self_check.outcome,
                    verification=self._verification_result_for_self_check(
                        self_check
                    ),
                    metadata={
                        "reasons": list(self_check.reasons),
                        "citation_count": 0,
                        "retry_attempted": False,
                        "verifier_skipped": True,
                    },
                )
                run, guardrail_decision = self._record_loop_step(run, verify_step)
                if guardrail_decision:
                    return self._finish_guardrail_query_result(
                        decision=guardrail_decision,
                        question=clean_prompt,
                        active_state=active_state,
                        run=run,
                    )
            elif hasattr(active_state.retrieval_chain, "invoke_with_trace"):
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
                model_thinking = getattr(chain_result, "model_thinking", None)
                draft_metadata = {
                    "answer_chars": len(str(response).strip()),
                    "inline_citation_ids": self._inline_citation_ids(str(response)),
                    "model_thinking_available": bool(model_thinking),
                }
                if model_thinking:
                    draft_metadata["model_thinking_chars"] = len(model_thinking)
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
                    model_thinking = getattr(retry_result, "model_thinking", None)
                    retry_draft_metadata = {
                        "answer_chars": len(str(response).strip()),
                        "inline_citation_ids": self._inline_citation_ids(
                            str(response)
                        ),
                        "model_thinking_available": bool(model_thinking),
                    }
                    if model_thinking:
                        retry_draft_metadata["model_thinking_chars"] = len(
                            model_thinking
                        )
                    run, guardrail_decision = self._record_loop_step(
                        run,
                        self._loop_step(
                            LoopPhase.DRAFT,
                            decision=LoopDecision.CONTINUE,
                            name="Draft retry answer",
                            input_summary=clean_prompt,
                            output_summary=str(response).strip()[:500],
                            retry_count=1,
                            metadata=retry_draft_metadata,
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
                model_thinking = self._last_model_thinking()
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
            if len(response) < 3 and not direct_context:
                response = SELF_CHECK_REFUSAL_ANSWER
                model_thinking = None
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
                model_thinking = None
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
                model_thinking=model_thinking,
                metadata={
                    "context_provider": run.context_provider,
                    "self_check_outcome": self_check.outcome if self_check else None,
                    "retry_attempted": (
                        self_check.retry_attempted if self_check else False
                    ),
                },
            )

            self._record_chat_history_entry(
                session_id=session_id,
                question=clean_prompt,
                result=result,
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

__all__ = sorted(
    {
        name
        for name in globals()
        if not name.startswith("_")
    }
    | _RETRIEVAL_EXPORTS
)
