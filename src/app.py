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

# Initialize the DocumentQA system
qa_system = DocumentQA(
    allow_interactive_token=False, fast_mode=env_flag("FAST_MODE", False)
)


def process_document(file):
    """Process the uploaded document."""
    if file is None or not getattr(file, "name", None):
        return "No document uploaded."

    try:
        qa_system.process_document(file.name)
        uploaded_name = os.path.basename(file.name)
        active_model = qa_system.loaded_model_id or "MockLLM (fallback)"
        mode_label = "FAST" if qa_system.fast_mode else "QUALITY"
        return (
            f"Document `{uploaded_name}` processed successfully. "
            f"Profile: `{mode_label}`. Active model: `{active_model}`. "
            "You can now ask questions about it."
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
            upload_button = gr.Button("Process Document")
            upload_status = gr.Textbox(label="Status")

        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Ask a question")
            clear = gr.Button("Clear")

    upload_button.click(process_document, inputs=file_upload, outputs=upload_status)
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
