import gradio as gr
from DocumentQA import DocumentQA

# Initialize the DocumentQA system
qa_system = DocumentQA()

def process_document(file):
    """
    Process the uploaded document.
    """
    if file is not None:
        qa_system.process_document(file.name)
        return "Thank you for providing your document. I have analyzed it, so now you can ask me any questions regarding it!"
    return "No document uploaded."

def chat(message, history):
    """
    Chat function to interact with the DocumentQA system.
    """
    return qa_system.query(message)

def clear_chat():
    """
    Clear the chat history.
    """
    qa_system.session_store.pop("default", None)
    return None

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
    msg.submit(chat, [msg, chatbot], chatbot)
    clear.click(clear_chat, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(debug=True, server_name="0.0.0.0", server_port=7860)
