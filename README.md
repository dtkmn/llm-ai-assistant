---
title: LLM Powered AI Assistant
emoji: 🏳️‍🌈
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7862
---


# LLM AI Assistant
This project leverages LangChain and Meta's Llama 3.2 to create an AI assistant capable of answering questions based on provided document context. The assistant uses local model execution for better quality and reliability.

https://huggingface.co/spaces/0xdant/llm-ai-assistant

## Features
- **Meta Llama 3.2 Integration:** Uses Meta's Llama-3.2-3B-Instruct model from HuggingFace for high-quality responses
- **Local Model Execution:** Runs models locally for better reliability and control
- **Document Processing:** Supports PDF, DOCX, and text documents with intelligent chunking
- **Vector Search:** Uses FAISS for efficient similarity search with HuggingFace embeddings
- **Gradio Interface:** Modern, user-friendly web interface for document upload and chat
- **GPU/MPS Support:** Automatically utilizes CUDA, Apple Silicon (MPS), or CPU


![LLM-flow.png](https://github.com/dtkmn/llm-ai-assistant/blob/main/LLM-flow.png)

## Installation

### Prerequisites
- Python 3.11 or higher
- HuggingFace account and API token ([get one here](https://huggingface.co/settings/tokens))
- At least 8GB RAM (for Llama 3.2-3B model)

### Local Setup

1. Clone the repository:
    
    ```bash
    git clone https://github.com/dtkmn/llm-ai-assistant.git
    cd llm-ai-assistant
    ``` 

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up your HuggingFace token:    

    ```bash
    export HUGGINGFACEHUB_API_TOKEN=your_token_here
    ```

5. Run the application:

    ```bash
    python src/app.py
    ```
   
## 🐳 Docker Setup

The application is containerized for easy deployment.

### Build the Docker Image

   ```bash
   docker build -t llm-ai-assistant .
   ```

### Run the Container

   ```bash
   docker run -p 7862:7862 \
     -e HUGGINGFACEHUB_API_TOKEN=your_token_here \
     llm-ai-assistant
   ```

**Note:** The first run will download ~6GB for the Llama 3.2 model, which may take a few minutes.


## Usage
1. Open your browser and go to `http://localhost:7862`
2. Upload a document (PDF, DOCX, or TXT)
3. Click "Process Document" to analyze it
4. Ask questions about your document in the chat interface
5. Get detailed, AI-generated answers based on the document content

## Technical Details

### Model
- **LLM:** Meta Llama-3.2-3B-Instruct (locally executed)
- **Embeddings:** sentence-transformers/all-mpnet-base-v2
- **Vector Store:** FAISS for efficient similarity search
- **Framework:** LangChain for orchestration

### Configuration
- **Response Length:** Up to 512 tokens (~350-400 words)
- **Temperature:** 0.3 (balanced between creativity and consistency)
- **Chunk Size:** 800 characters with 100 character overlap
- **Retrieval:** Top 5 most relevant chunks per query

## Contributing
Feel free to submit issues and pull requests. Contributions are welcome!

## License
This project is open source and available under the MIT License.

