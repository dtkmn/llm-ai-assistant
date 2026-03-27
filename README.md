---
title: LLM Powered AI Assistant
emoji: 🏳️‍🌈
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
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

5. (Optional) choose a different local model:

    ```bash
    export LLM_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
    ```

6. (Optional) enable fast mode (lower latency, lower quality):

    ```bash
    export FAST_MODE=true
    ```

7. (Optional) enable debug logs:

    ```bash
    export APP_DEBUG=false
    ```

8. Run the application:

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
   docker run -p 7860:7860 \
     -e HUGGINGFACEHUB_API_TOKEN=your_token_here \
     llm-ai-assistant
   ```

**Note:** The first run downloads model weights (size depends on `LLM_MODEL_ID`; the default 1.5B model is much smaller than 4B/7B models but still takes a few GB).


## Usage
1. Open your browser and go to `http://localhost:7860`
2. Upload a document (PDF, DOCX, TXT, or MD; max 25 MB)
3. Click "Process Document" to analyze it
4. Ask questions about your document in the chat interface
5. Get detailed, AI-generated answers based on the document content

## Technical Details

### Model
- **LLM:** Configurable via `LLM_MODEL_ID`
  - **Quality mode (default):** tries `Qwen/Qwen2.5-1.5B-Instruct`, then `Qwen/Qwen2.5-7B-Instruct`, then `meta-llama/Llama-3.2-3B-Instruct`
  - **Fast mode (`FAST_MODE=true`):** tries `Qwen/Qwen2.5-1.5B-Instruct` first, then `meta-llama/Llama-3.2-3B-Instruct`
- **Embeddings:**
  - **Quality mode:** `Alibaba-NLP/gte-modernbert-base`
  - **Fast mode:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store:** FAISS for efficient similarity search
- **Framework:** LangChain for orchestration

### Configuration
- **Response Length:** 384 new tokens (quality) / 160 new tokens (fast)
- **Generation mode:** Deterministic (`do_sample=False`) for more reliable doc QA
- **Chunk Size:** 1200/200 overlap (quality) / 900/120 overlap (fast)
- **Retrieval:** MMR retrieval with source/page grounding
  - **Quality:** `k=6`, `fetch_k=24`
  - **Fast:** `k=3`, `fetch_k=10`
- **Safety limits:** Max upload size 25 MB, chunk cap 2,000 chunks per document

## Security and Dependency Maintenance
- Dependencies are pinned in `requirements.txt` for reproducible installs.
- Dependabot is enabled weekly (`.github/dependabot.yml`) for dependency updates.
- Current baseline includes Gradio `6.6.0`, LangChain `1.2.10`, LangChain-HuggingFace `1.2.1`, HuggingFace Hub `1.5.0`, and Transformers `5.2.0`.
- Note: `marshmallow` is intentionally pinned to `3.26.2` because `dataclasses-json` currently requires `<4.0.0`.
- Security-sensitive transitive dependencies are explicitly pinned (for example `aiohttp`, `urllib3`, `python-multipart`, and `orjson`) to keep audit results stable.
- Recommended recurring checks:

  ```bash
  pip install pip-audit
  pip-audit -r requirements.txt
  pip check
  ```

## Contributing
Feel free to submit issues and pull requests. Contributions are welcome!

## License
This project is open source and available under the MIT License.
