import os
import unicodedata
from typing import Optional

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
