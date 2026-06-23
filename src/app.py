import json
import logging
import os
import sys
from dataclasses import replace
from typing import Optional

import gradio as gr

# Add parent directory to path to handle both local and Docker execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .DocumentQA import (
        MAX_DOCUMENT_CHUNKS,
        DocumentProcessingError,
        DocumentProcessingReport,
        DocumentQA,
        DocumentQAStatus,
        QueryResult,
    )
except ImportError:
    from DocumentQA import (
        MAX_DOCUMENT_CHUNKS,
        DocumentProcessingError,
        DocumentProcessingReport,
        DocumentQA,
        DocumentQAStatus,
        QueryResult,
    )


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


logging.basicConfig(
    level=logging.DEBUG if env_flag("APP_DEBUG", False) else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)
APP_TITLE = "AI Loop Engine"
TEXT_ENCODING_OPTIONS = {
    "Auto": "auto",
    "UTF-8 / Western": "utf-8-or-western",
    "UTF-8": "utf-8",
    "Western (Windows-1252)": "cp1252",
    "Latin-1 (ISO-8859-1)": "latin-1",
    "Central European (Windows-1250)": "cp1250",
    "Cyrillic (Windows-1251)": "cp1251",
    "Turkish (Windows-1254)": "cp1254",
    "Baltic (Windows-1257)": "cp1257",
}

# Initialize the document context loop system
qa_system = DocumentQA(
    allow_interactive_token=False, fast_mode=env_flag("FAST_MODE", False)
)


def format_upload_status(uploaded_name: str, qa_status: DocumentQAStatus) -> str:
    report = qa_status.processing_report
    document_name = (
        report.attempted_document_name
        if report and report.attempted_document_name
        else uploaded_name
    )

    if report and not report.success:
        active_message = (
            f"Active document remains `{report.active_document_name}`."
            if report.active_document_name
            else "No active document is loaded."
        )
        return (
            f"Document context `{document_name}` failed during `{report.phase}`. "
            f"{active_message} Error: {report.error_message}"
        )

    chunk_message = ""
    if report:
        chunk_message = f" Chunks: `{report.chunk_count}`"
        if report.truncated:
            chunk_message += f" (truncated at `{report.max_chunk_limit}`)."
        else:
            chunk_message += "."

    if qa_status.mock_mode:
        return (
            f"Document context `{document_name}` processed in mock mode. "
            f"Profile: `{qa_status.profile_label}`. "
            f"Active model: `{qa_status.active_model_label}`. "
            f"{chunk_message} "
            "Answers will be demonstration responses until a real LLM backend is configured."
        )
    return (
        f"Document context `{document_name}` indexed. "
        f"Profile: `{qa_status.profile_label}`. "
        f"Backend: `{qa_status.active_backend}`. "
        f"Active model: `{qa_status.active_model_label}`. "
        f"{chunk_message} "
        "Inference will be validated on the first question."
    )


def format_runtime_status(qa_status: DocumentQAStatus) -> str:
    report = qa_status.processing_report
    runtime_status = {
        "active_document": qa_status.document_name,
        "last_attempted_document": (
            report.attempted_document_name if report else None
        ),
        "backend": qa_status.active_backend,
        "model": qa_status.active_model_label,
        "profile": qa_status.profile_label,
        "ready_for_queries": qa_status.ready_for_queries,
        "readiness_scope": "retrieval_pipeline",
        "inference_validated": False,
        "last_success": report.success if report else None,
        "phase": report.phase if report else None,
        "file_extension": report.file_extension if report else None,
        "chunk_count": report.chunk_count if report else 0,
        "truncated": report.truncated if report else False,
        "max_chunk_limit": report.max_chunk_limit if report else None,
        "text_encoding_mode": report.text_encoding_mode if report else None,
        "last_error": report.error_message if report else None,
    }
    return json.dumps(runtime_status, indent=2)


def status_with_unexpected_upload_error(
    qa_status: DocumentQAStatus,
    uploaded_name: str,
    selected_encoding: str,
    exc: Exception,
) -> DocumentQAStatus:
    previous_report = qa_status.processing_report
    max_chunk_limit = (
        previous_report.max_chunk_limit if previous_report else MAX_DOCUMENT_CHUNKS
    )
    failure_report = DocumentProcessingReport(
        attempted_document_name=uploaded_name,
        active_document_name=qa_status.document_name,
        success=False,
        phase="unexpected",
        file_extension=os.path.splitext(uploaded_name)[1].lower() or None,
        chunk_count=0,
        truncated=False,
        max_chunk_limit=max_chunk_limit,
        text_encoding_mode=selected_encoding or "auto",
        backend=qa_status.active_backend,
        model_label=qa_status.active_model_label,
        error_message=str(exc),
    )
    return replace(qa_status, processing_report=failure_report)


def public_trace_error(
    query_result: QueryResult,
    public_loop_report: Optional[dict],
) -> Optional[str]:
    redaction = (public_loop_report or {}).get("public_redaction") or {}
    if redaction.get("applied"):
        return "terminal_guardrail_decision"
    return query_result.trace.error_message


def format_answer_trace(query_result: Optional[QueryResult]) -> str:
    if query_result is None:
        return json.dumps(
            {
                "question": None,
                "answer": None,
                "document": None,
                "backend": None,
                "model": None,
                "retrieved_chunk_count": 0,
                "citations": [],
                "self_check": None,
                "loop_report": None,
                "error": None,
            },
            indent=2,
        )

    trace = query_result.trace
    self_check = trace.self_check
    public_loop_report = (
        query_result.loop_report.to_public_dict()
        if query_result.loop_report
        else None
    )
    return json.dumps(
        {
            "question": trace.question,
            "answer": query_result.answer,
            "document": trace.document_name,
            "backend": trace.backend,
            "model": trace.model_label,
            "retrieved_chunk_count": trace.retrieved_chunk_count,
            "citations": [
                {
                    "id": citation.citation_id,
                    "source": citation.source_name,
                    "page": citation.page,
                    "chunk": (
                        citation.chunk_index + 1
                        if citation.chunk_index is not None
                        else None
                    ),
                    "excerpt": citation.excerpt,
                }
                for citation in trace.citations
            ],
            "self_check": (
                {
                    "outcome": self_check.outcome,
                    "reasons": self_check.reasons,
                    "retry_attempted": self_check.retry_attempted,
                }
                if self_check
                else None
            ),
            "loop_report": public_loop_report,
            "error": public_trace_error(query_result, public_loop_report),
        },
        indent=2,
    )


def process_document(file, text_encoding="Auto"):
    """Process the uploaded document."""
    if file is None or not getattr(file, "name", None):
        return "No document uploaded.", format_runtime_status(qa_system.status())

    uploaded_name = os.path.basename(file.name)
    selected_encoding = TEXT_ENCODING_OPTIONS.get(text_encoding, "auto")
    pre_upload_status = qa_system.status()
    try:
        qa_status = qa_system.process_document(
            file.name, text_encoding=selected_encoding
        )
        return format_upload_status(uploaded_name, qa_status), format_runtime_status(
            qa_status
        )
    except DocumentProcessingError as exc:
        LOGGER.warning("Document processing failed: %s", exc)
        qa_status = exc.status
        return format_upload_status(uploaded_name, qa_status), format_runtime_status(
            qa_status
        )
    except RuntimeError as exc:
        LOGGER.exception("Unexpected document processing failure: %s", exc)
        qa_status = status_with_unexpected_upload_error(
            pre_upload_status, uploaded_name, selected_encoding, exc
        )
        return format_upload_status(uploaded_name, qa_status), format_runtime_status(
            qa_status
        )


def chat(message, history):
    """Chat function to interact with the current document context loop."""
    history = history or []
    if message and message.strip():
        query_result = qa_system.query_with_trace(message)
        response = query_result.answer
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, "", format_answer_trace(query_result)
    return history, "", format_answer_trace(None)


def clear_chat():
    """Clear the chat history."""
    qa_system.chat_history.clear()
    return [], format_answer_trace(None)


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(f"# {APP_TITLE}")

    with gr.Row():
        with gr.Column():
            file_upload = gr.File(label="Upload Document Context")
            text_encoding = gr.Dropdown(
                choices=list(TEXT_ENCODING_OPTIONS),
                value="Auto",
                label="Text Encoding",
            )
            upload_button = gr.Button("Index Context")
            upload_status = gr.Textbox(label="Context Status")
            runtime_status = gr.Textbox(
                label="Runtime Status",
                value=format_runtime_status(qa_system.status()),
                lines=12,
                interactive=False,
            )

        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Ask a question")
            answer_trace = gr.Textbox(
                label="Loop Trace",
                value=format_answer_trace(None),
                lines=12,
                interactive=False,
            )
            clear = gr.Button("Clear")

    upload_button.click(
        process_document,
        inputs=[file_upload, text_encoding],
        outputs=[upload_status, runtime_status],
    )
    msg.submit(chat, [msg, chatbot], [chatbot, msg, answer_trace])
    clear.click(clear_chat, None, [chatbot, answer_trace], queue=False)


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    demo.launch(
        debug=env_flag("APP_DEBUG", False),
        server_name=server_name,
        server_port=server_port,
        share=False,
    )
