import logging
import os
import sys

import gradio as gr

# Add parent directory to path to handle both local and Docker execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from DocumentQA import DocumentQA
except ImportError:
    from src.DocumentQA import DocumentQA


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

# Initialize the DocumentQA system
qa_system = DocumentQA(
    allow_interactive_token=False, fast_mode=env_flag("FAST_MODE", False)
)


def process_document(file, text_encoding="Auto"):
    """Process the uploaded document."""
    if file is None or not getattr(file, "name", None):
        return "No document uploaded."

    try:
        selected_encoding = TEXT_ENCODING_OPTIONS.get(text_encoding, "auto")
        qa_system.process_document(file.name, text_encoding=selected_encoding)
        uploaded_name = os.path.basename(file.name)
        active_model = (
            getattr(qa_system, "loaded_model_label", None)
            or qa_system.loaded_model_id
            or "MockLLM (fallback)"
        )
        active_backend = qa_system.active_llm_backend or qa_system.llm_backend
        mode_label = "FAST" if qa_system.fast_mode else "QUALITY"
        if active_backend == "mock":
            return (
                f"Document `{uploaded_name}` processed in mock mode. "
                f"Profile: `{mode_label}`. Active model: `{active_model}`. "
                "Answers will be demonstration responses until a real LLM backend is configured."
            )
        return (
            f"Document `{uploaded_name}` indexed. "
            f"Profile: `{mode_label}`. Backend: `{active_backend}`. "
            f"Active model: `{active_model}`. "
            "Inference will be validated on the first question."
        )
    except RuntimeError as exc:
        LOGGER.warning("Document processing failed: %s", exc)
        return str(exc)


def chat(message, history):
    """Chat function to interact with the DocumentQA system."""
    history = history or []
    if message and message.strip():
        response = qa_system.query(message)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
    return history, ""


def clear_chat():
    """Clear the chat history."""
    qa_system.chat_history.clear()
    return []


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# LLM AI Assistant")

    with gr.Row():
        with gr.Column():
            file_upload = gr.File(label="Upload Document")
            text_encoding = gr.Dropdown(
                choices=list(TEXT_ENCODING_OPTIONS),
                value="Auto",
                label="Text Encoding",
            )
            upload_button = gr.Button("Process Document")
            upload_status = gr.Textbox(label="Status")

        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Ask a question")
            clear = gr.Button("Clear")

    upload_button.click(
        process_document, inputs=[file_upload, text_encoding], outputs=upload_status
    )
    msg.submit(chat, [msg, chatbot], [chatbot, msg])
    clear.click(clear_chat, None, chatbot, queue=False)


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    demo.launch(
        debug=env_flag("APP_DEBUG", False),
        server_name=server_name,
        server_port=server_port,
        share=False,
    )
