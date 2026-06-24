"""Canonical public runtime API for AI Loop Engine.

The historical runtime lived behind ``src.DocumentQA``. Keep that module as a
compatibility shim while new code imports this public boundary.
"""

try:
    from .DocumentQA import (
        DEFAULT_OLLAMA_BASE_URL,
        DEFAULT_OLLAMA_MODEL,
        LLM_MODEL_ENV_VAR,
        MAX_DOCUMENT_CHUNKS,
        OLLAMA_BASE_URL_ENV_VAR,
        OLLAMA_MODEL_ENV_VAR,
        SELF_CHECK_REFUSAL_ANSWER,
        AILoopEngine,
        DocumentProcessingError,
        DocumentProcessingReport,
        DocumentQA,
        DocumentQAStatus,
        QueryResult,
    )
except ImportError:
    from DocumentQA import (
        DEFAULT_OLLAMA_BASE_URL,
        DEFAULT_OLLAMA_MODEL,
        LLM_MODEL_ENV_VAR,
        MAX_DOCUMENT_CHUNKS,
        OLLAMA_BASE_URL_ENV_VAR,
        OLLAMA_MODEL_ENV_VAR,
        SELF_CHECK_REFUSAL_ANSWER,
        AILoopEngine,
        DocumentProcessingError,
        DocumentProcessingReport,
        DocumentQA,
        DocumentQAStatus,
        QueryResult,
    )


__all__ = [
    "AILoopEngine",
    "DocumentQA",
    "DocumentProcessingError",
    "DocumentProcessingReport",
    "DocumentQAStatus",
    "MAX_DOCUMENT_CHUNKS",
    "QueryResult",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "LLM_MODEL_ENV_VAR",
    "OLLAMA_BASE_URL_ENV_VAR",
    "OLLAMA_MODEL_ENV_VAR",
    "SELF_CHECK_REFUSAL_ANSWER",
]
