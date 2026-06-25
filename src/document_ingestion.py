import logging
import os
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol

try:
    from .native_runtime import apply_native_runtime_defaults
except ImportError:
    from native_runtime import apply_native_runtime_defaults


apply_native_runtime_defaults()

import docx2txt
from charset_normalizer import from_bytes
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_DOCUMENT_BYTES = 25 * 1024 * 1024
MAX_DOCUMENT_CHUNKS = 2_000
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


@dataclass(frozen=True)
class SplitDocumentResult:
    chunks: List[Document]
    truncated: bool


class TextDecoder(Protocol):
    def decode_text_file(
        self, document_path: str, text_encoding: Optional[str] = None
    ) -> str:
        ...

    def decode_supported_text(self, raw_content: bytes, encoding: str) -> str:
        ...

    def decode_utf8_or_western_text(self, raw_content: bytes) -> str:
        ...

    def decode_confident_detected_text(self, raw_content: bytes, matches) -> Optional[str]:
        ...

    def fallback_encoding_is_plausible(
        self, raw_content: bytes, matches, encoding: str, fallback_text: str
    ) -> bool:
        ...

    def detected_encoding_is_confident(self, matches, index: int) -> bool:
        ...

    def decode_nul_pattern_text(self, raw_content: bytes) -> Optional[str]:
        ...

    def nul_pattern_encodings(self, raw_content: bytes) -> List[str]:
        ...


class DefaultTextDecoder:
    def decode_text_file(
        self, document_path: str, text_encoding: Optional[str] = None
    ) -> str:
        with open(document_path, "rb") as file:
            raw_content = file.read()

        if text_encoding:
            normalized_text_encoding = text_encoding.strip().lower()
            if normalized_text_encoding == "utf-8-or-western":
                return self.decode_utf8_or_western_text(raw_content)
            if normalized_text_encoding != "auto":
                return self.decode_supported_text(raw_content, text_encoding.strip())

        for bom, encoding in TEXT_ENCODING_BOMS:
            if raw_content.startswith(bom):
                return self.decode_supported_text(raw_content, encoding)

        nul_pattern_text = self.decode_nul_pattern_text(raw_content)
        if nul_pattern_text is not None:
            return nul_pattern_text

        try:
            return self.decode_supported_text(raw_content, "utf-8")
        except UnicodeDecodeError:
            pass

        detected_matches = list(from_bytes(raw_content))
        detected_text = self.decode_confident_detected_text(
            raw_content, detected_matches
        )
        if detected_text is not None:
            return detected_text

        for encoding in TEXT_ENCODING_FALLBACKS:
            try:
                fallback_text = self.decode_supported_text(raw_content, encoding)
            except (LookupError, UnicodeDecodeError):
                continue
            if not self.fallback_encoding_is_plausible(
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

    def decode_supported_text(self, raw_content: bytes, encoding: str) -> str:
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

    def decode_utf8_or_western_text(self, raw_content: bytes) -> str:
        for bom, encoding in TEXT_ENCODING_BOMS:
            if raw_content.startswith(bom):
                return self.decode_supported_text(raw_content, encoding)

        try:
            return self.decode_supported_text(raw_content, "utf-8")
        except UnicodeDecodeError:
            pass

        for encoding in TEXT_ENCODING_FALLBACKS:
            try:
                return self.decode_supported_text(raw_content, encoding)
            except (LookupError, UnicodeDecodeError):
                continue

        raise UnicodeDecodeError(
            "text-file",
            raw_content,
            0,
            len(raw_content),
            "unable to decode as UTF-8 or Western text",
        )

    def decode_confident_detected_text(
        self, raw_content: bytes, matches
    ) -> Optional[str]:
        for index, match in enumerate(matches):
            if not self.detected_encoding_is_confident(matches, index):
                continue
            candidates = [match.encoding, *(match.could_be_from_charset or [])]
            for candidate in candidates:
                if not candidate:
                    continue
                normalized = normalize_encoding_name(candidate)
                if normalized in UTF_NUL_FAMILY_ENCODINGS and b"\x00" not in raw_content:
                    continue
                try:
                    return self.decode_supported_text(raw_content, candidate)
                except (LookupError, UnicodeDecodeError):
                    continue
        return None

    def fallback_encoding_is_plausible(
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
                    candidate_text = self.decode_supported_text(raw_content, candidate)
                except (LookupError, UnicodeDecodeError):
                    continue
                if candidate_text != fallback_text:
                    return False

        # Auto mode must fail closed on ambiguous legacy candidates. The UI
        # defaults to Auto; Western and other legacy codecs are explicit opt-ins.
        if len(raw_content) <= MAX_SHORT_WESTERN_FALLBACK_BYTES:
            return fallback_penalty == 0

        return fallback_listed_as_low_chaos and fallback_penalty == 0

    def detected_encoding_is_confident(self, matches, index: int) -> bool:
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

    def decode_nul_pattern_text(self, raw_content: bytes) -> Optional[str]:
        if b"\x00" not in raw_content:
            return None

        for encoding in self.nul_pattern_encodings(raw_content):
            try:
                return self.decode_supported_text(raw_content, encoding)
            except (LookupError, UnicodeDecodeError):
                continue
        return None

    def nul_pattern_encodings(self, raw_content: bytes) -> List[str]:
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


DEFAULT_TEXT_DECODER = DefaultTextDecoder()


def validate_document(document_path: str, max_document_bytes: int) -> str:
    if not os.path.exists(document_path):
        raise ValueError("Uploaded file could not be found.")

    file_extension = os.path.splitext(document_path)[1].lower()
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file_extension}")

    file_size = os.path.getsize(document_path)
    if file_size > max_document_bytes:
        limit_mb = max_document_bytes // (1024 * 1024)
        raise ValueError(
            f"File is too large ({file_size} bytes). Maximum supported size is {limit_mb} MB."
        )

    return file_extension


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


def decode_text_file(
    document_path: str, text_encoding: Optional[str] = None
) -> str:
    return DEFAULT_TEXT_DECODER.decode_text_file(document_path, text_encoding)


def decode_supported_text(raw_content: bytes, encoding: str) -> str:
    return DEFAULT_TEXT_DECODER.decode_supported_text(raw_content, encoding)


def decode_utf8_or_western_text(raw_content: bytes) -> str:
    return DEFAULT_TEXT_DECODER.decode_utf8_or_western_text(raw_content)


def decode_confident_detected_text(raw_content: bytes, matches) -> Optional[str]:
    return DEFAULT_TEXT_DECODER.decode_confident_detected_text(raw_content, matches)


def fallback_encoding_is_plausible(
    raw_content: bytes, matches, encoding: str, fallback_text: str
) -> bool:
    return DEFAULT_TEXT_DECODER.fallback_encoding_is_plausible(
        raw_content, matches, encoding, fallback_text
    )


def detected_encoding_is_confident(matches, index: int) -> bool:
    return DEFAULT_TEXT_DECODER.detected_encoding_is_confident(matches, index)


def decode_nul_pattern_text(raw_content: bytes) -> Optional[str]:
    return DEFAULT_TEXT_DECODER.decode_nul_pattern_text(raw_content)


def nul_pattern_encodings(raw_content: bytes) -> List[str]:
    return DEFAULT_TEXT_DECODER.nul_pattern_encodings(raw_content)


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
