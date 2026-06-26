from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

try:
    from .env_file import load_local_env_files
except ImportError:
    from env_file import load_local_env_files

try:
    from .native_runtime import apply_native_runtime_defaults
except ImportError:
    from native_runtime import apply_native_runtime_defaults


load_local_env_files()
apply_native_runtime_defaults()

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

try:
    from .ai_loop_engine import AILoopEngine, DocumentProcessingError
    from .document_config import MAX_DOCUMENT_BYTES
    from .web_contract import (
        APP_TITLE,
        TEXT_ENCODING_OPTIONS,
        empty_query_response_dict,
        env_flag,
        normalize_text_encoding,
        query_response_dict,
        runtime_status_dict,
        status_with_unexpected_upload_error,
        upload_status_message,
    )
except ImportError:
    from ai_loop_engine import AILoopEngine, DocumentProcessingError
    from document_config import MAX_DOCUMENT_BYTES
    from web_contract import (
        APP_TITLE,
        TEXT_ENCODING_OPTIONS,
        empty_query_response_dict,
        env_flag,
        normalize_text_encoding,
        query_response_dict,
        runtime_status_dict,
        status_with_unexpected_upload_error,
        upload_status_message,
    )


logging.basicConfig(
    level=logging.DEBUG if env_flag("APP_DEBUG", False) else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "web_static"
MAX_UPLOAD_FILENAME_LENGTH = 180
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_SESSION_ID_LENGTH = 96
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
qa_system: Optional[AILoopEngine] = None


class QueryRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ClearChatRequest(BaseModel):
    session_id: Optional[str] = None


def get_engine() -> AILoopEngine:
    global qa_system
    if qa_system is None:
        qa_system = AILoopEngine(fast_mode=env_flag("FAST_MODE", False))
    return qa_system


def safe_upload_name(filename: str | None) -> str:
    raw_name = filename or ""
    raw_basename = os.path.basename(raw_name)
    if any(ord(char) < 32 or ord(char) == 127 for char in raw_basename):
        raise HTTPException(status_code=400, detail="Invalid upload filename.")
    name = raw_basename.strip()
    if raw_name and not name:
        raise HTTPException(status_code=400, detail="Invalid upload filename.")
    if name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid upload filename.")
    if len(name.encode("utf-8")) > MAX_UPLOAD_FILENAME_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid upload filename.")
    return name or "upload"


def safe_session_id(session_id: str | None) -> str:
    value = str(session_id or "default").strip()
    if not value:
        return "default"
    if len(value.encode("utf-8")) > MAX_SESSION_ID_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid session id.")
    if not SESSION_ID_PATTERN.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid session id.")
    return value


def clear_chat_history_for_session(engine: AILoopEngine, session_id: str) -> None:
    history = getattr(engine, "chat_history", None)
    if not isinstance(history, list):
        return

    def entry_session_id(entry: object) -> str:
        if isinstance(entry, dict):
            return str(entry.get("session_id") or "default").strip() or "default"
        return "default"

    history[:] = [
        entry for entry in history if entry_session_id(entry) != session_id
    ]


async def write_upload_file(upload: UploadFile, upload_path: Path) -> None:
    bytes_written = 0
    with upload_path.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > MAX_DOCUMENT_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Uploaded document exceeds the 25 MB limit.",
                )
            output.write(chunk)


def create_app(engine: Optional[AILoopEngine] = None) -> FastAPI:
    api = FastAPI(title=APP_TITLE)
    api.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    def runtime() -> AILoopEngine:
        return engine if engine is not None else get_engine()

    @api.middleware("http")
    async def reject_oversized_upload_request(request: Request, call_next):
        if request.method.upper() == "POST" and request.url.path == "/api/documents":
            content_length = request.headers.get("content-length")
            if not content_length:
                return JSONResponse(
                    {"detail": "Content-Length is required for document uploads."},
                    status_code=411,
                )
            try:
                request_bytes = int(content_length)
            except ValueError:
                return JSONResponse(
                    {"detail": "Invalid Content-Length."},
                    status_code=400,
                )
            if request_bytes > MAX_DOCUMENT_BYTES + MAX_MULTIPART_OVERHEAD_BYTES:
                return JSONResponse(
                    {"detail": "Uploaded document exceeds the 25 MB limit."},
                    status_code=413,
                )
        return await call_next(request)

    @api.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @api.get("/api/health")
    def health() -> dict:
        return {"ok": True, "app": APP_TITLE}

    @api.get("/api/config")
    def config() -> dict:
        return {
            "title": APP_TITLE,
            "text_encodings": [
                {"label": label, "value": value}
                for label, value in TEXT_ENCODING_OPTIONS.items()
            ],
        }

    @api.get("/api/status")
    def status() -> dict:
        return runtime_status_dict(runtime().status())

    @api.post("/api/documents")
    async def upload_document(
        file: UploadFile = File(...),
        text_encoding: str = Form("auto"),
    ) -> JSONResponse:
        uploaded_name = safe_upload_name(file.filename)
        selected_encoding = normalize_text_encoding(text_encoding)
        if selected_encoding is None:
            raise HTTPException(status_code=400, detail="Unsupported text encoding.")

        with TemporaryDirectory(prefix="ai-loop-upload-") as temp_dir:
            upload_path = Path(temp_dir) / uploaded_name
            await write_upload_file(file, upload_path)
            current_engine = runtime()
            pre_upload_status = current_engine.status()
            try:
                qa_status = current_engine.process_document(
                    str(upload_path),
                    text_encoding=selected_encoding,
                )
                return JSONResponse(
                    {
                        "message": upload_status_message(uploaded_name, qa_status),
                        "status": runtime_status_dict(qa_status),
                    }
                )
            except DocumentProcessingError as exc:
                LOGGER.warning("Document processing failed: %s", exc)
                qa_status = exc.status
                return JSONResponse(
                    {
                        "message": upload_status_message(uploaded_name, qa_status),
                        "status": runtime_status_dict(qa_status),
                    },
                    status_code=400,
                )
            except RuntimeError as exc:
                LOGGER.exception("Unexpected document processing failure: %s", exc)
                qa_status = status_with_unexpected_upload_error(
                    pre_upload_status,
                    uploaded_name,
                    selected_encoding,
                    exc,
                )
                return JSONResponse(
                    {
                        "message": upload_status_message(uploaded_name, qa_status),
                        "status": runtime_status_dict(qa_status),
                    },
                    status_code=500,
                )

    @api.post("/api/query")
    def query(request: QueryRequest) -> dict:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required.")
        session_id = safe_session_id(request.session_id)
        return query_response_dict(
            runtime().query_with_trace(message, session_id=session_id)
        )

    @api.post("/api/chat/clear")
    def clear_chat(request: Optional[ClearChatRequest] = Body(default=None)) -> dict:
        session_id = safe_session_id(request.session_id if request else None)
        current_engine = runtime()
        if hasattr(current_engine, "clear_loop_session"):
            current_engine.clear_loop_session(session_id)
        clear_chat_history_for_session(current_engine, session_id)
        return empty_query_response_dict()

    return api


app = create_app()


def main() -> None:
    host = os.getenv("WEB_HOST", os.getenv("HOST", "0.0.0.0"))
    port = int(os.getenv("PORT", os.getenv("WEB_PORT", "7860")))
    log_level = "debug" if env_flag("APP_DEBUG") else "info"
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
