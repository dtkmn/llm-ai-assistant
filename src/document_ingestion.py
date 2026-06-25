import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

try:
    from .native_runtime import apply_native_runtime_defaults
except ImportError:
    from native_runtime import apply_native_runtime_defaults


apply_native_runtime_defaults()

import docx2txt
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

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
    from .document_text import (
        DEFAULT_TEXT_DECODER,
        DefaultTextDecoder,
        TextDecoder,
        decode_confident_detected_text,
        decode_nul_pattern_text,
        decode_supported_text,
        decode_text_file,
        decode_utf8_or_western_text,
        detected_encoding_is_confident,
        fallback_encoding_is_plausible,
        nul_pattern_encodings,
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
    from document_text import (
        DEFAULT_TEXT_DECODER,
        DefaultTextDecoder,
        TextDecoder,
        decode_confident_detected_text,
        decode_nul_pattern_text,
        decode_supported_text,
        decode_text_file,
        decode_utf8_or_western_text,
        detected_encoding_is_confident,
        fallback_encoding_is_plausible,
        nul_pattern_encodings,
    )

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitDocumentResult:
    chunks: List[Document]
    truncated: bool


def load_documents(
    document_path: str,
    file_extension: str,
    text_encoding: Optional[str] = None,
    decode_text: Optional[Callable[[str, Optional[str]], str]] = None,
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
        text_decoder = decode_text or decode_text_file
        content = text_decoder(document_path, text_encoding)
    except (LookupError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not decode text document: {exc}") from exc
    return [Document(page_content=content, metadata={"source": document_path})]


def split_document_chunks(
    documents: List[Document],
    profile: Dict,
    max_document_chunks: int,
) -> SplitDocumentResult:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=profile["splitter_chunk_size"],
        chunk_overlap=profile["splitter_chunk_overlap"],
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks were generated from the document.")

    truncated = False
    if len(chunks) > max_document_chunks:
        truncated = True
        LOGGER.warning(
            "Document produced %s chunks; truncating to %s to keep memory bounded.",
            len(chunks),
            max_document_chunks,
        )
        chunks = chunks[:max_document_chunks]

    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata = dict(chunk.metadata or {})
        chunk.metadata["chunk_index"] = chunk_index

    return SplitDocumentResult(chunks=chunks, truncated=truncated)
