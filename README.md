---
title: LLM Powered AI Assistant
emoji: 🏳️‍🌈
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
---


# LLM AI Assistant
This project uses LangChain, Hugging Face or Ollama LLM backends, FAISS, and Gradio to answer questions from uploaded document context. It is a compact retrieval-augmented generation (RAG) assistant designed to run locally, in Docker, or on Hugging Face Spaces.

https://huggingface.co/spaces/0xdant/llm-ai-assistant

## Features
- **Local-first LLM Backend:** Recommended local path is Ollama; Hugging Face endpoint/local backends remain available for hosted deployments, gated models, and Hugging Face Spaces
- **Document Processing:** Supports PDF, DOCX, and text documents with intelligent chunking
- **Vector Search:** Uses FAISS for efficient similarity search with HuggingFace embeddings
- **Gradio Interface:** Modern, user-friendly web interface for document upload and chat
- **GPU/MPS Support:** Automatically utilizes CUDA, Apple Silicon (MPS), or CPU


![LLM-flow.png](https://github.com/dtkmn/llm-ai-assistant/blob/main/LLM-flow.png)

## Installation

### Prerequisites
- Python 3.11 or higher
- Ollama installed for the recommended local-first setup
- HuggingFace account and API token for hosted endpoint inference or gated models ([get one here](https://huggingface.co/settings/tokens)); not required for `LLM_BACKEND=ollama`
- At least 8GB RAM if forcing local model execution on CPU; GPU/MPS is strongly preferred for local models

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

4. Run local-first with Ollama:

    Terminal 1, unless the Ollama desktop app/service is already running:

    ```bash
    ollama serve
    ```

    Terminal 2:

    ```bash
    ollama pull nemotron-3-nano:4b
    export LLM_BACKEND=ollama
    export OLLAMA_MODEL=nemotron-3-nano:4b
    export OLLAMA_BASE_URL=http://localhost:11434
    ```

5. (Optional) choose a different backend:

    Supported values:
    - `ollama` uses a local Ollama server. This is the recommended local path.
    - `auto` uses hosted Hugging Face inference on CPU and local weights on CUDA/MPS.
    - `endpoint` always uses hosted Hugging Face inference.
    - `local` always downloads and runs Hugging Face model weights in-process.
    - `mock` disables real inference for demos/tests.

6. (Optional) set up Hugging Face for hosted endpoint inference, CPU `auto`, or gated Hugging Face `local` models:

    ```bash
    export HUGGINGFACEHUB_API_TOKEN=your_token_here
    ```

7. (Optional) choose a different Hugging Face local model:

    ```bash
    export LLM_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
    ```

8. (Optional) enable fast mode (lower latency, lower quality):

    ```bash
    export FAST_MODE=true
    ```

9. (Optional) enable debug logs:

    ```bash
    export APP_DEBUG=false
    ```

10. Run the application:

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

**Note:** With `LLM_BACKEND=auto`, CPU deployments use hosted Hugging Face inference and avoid downloading LLM weights. The `local` backend downloads Hugging Face model weights in-process. The `ollama` backend expects an already-running Ollama server and a pulled model.


## Usage
1. Open your browser and go to `http://localhost:7860`
2. Upload a document (PDF, DOCX, TXT, or MD; max 25 MB)
3. Click "Process Document" to analyze it
4. Ask questions about your document in the chat interface
5. Get detailed, AI-generated answers based on the document content

## Technical Details

### Model
- **LLM backend:** Configurable via `LLM_BACKEND`
  - `ollama`: local Ollama server via `OLLAMA_BASE_URL`; recommended for local use
  - `auto` (default): hosted endpoint on CPU, local model on CUDA/MPS
  - `endpoint`: hosted Hugging Face inference
  - `local`: in-process Transformers pipeline
  - `mock`: deterministic demo/test fallback
- **LLM model:** Configurable via `LLM_MODEL_ID`
  - **Quality mode (default):** tries `Qwen/Qwen2.5-1.5B-Instruct`, then `Qwen/Qwen2.5-7B-Instruct`, then `meta-llama/Llama-3.2-3B-Instruct`
  - **Fast mode (`FAST_MODE=true`):** tries `Qwen/Qwen2.5-1.5B-Instruct` first, then `meta-llama/Llama-3.2-3B-Instruct`
- **Hosted endpoint:** optionally configurable with `HF_ENDPOINT_URL` and `HF_ENDPOINT_TIMEOUT`
- **Ollama:** optionally configurable with `OLLAMA_MODEL` (default `nemotron-3-nano:4b`), `OLLAMA_BASE_URL` (default `http://localhost:11434`), and `OLLAMA_TIMEOUT`
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

## Local-First Direction
- The product direction is **download and run locally first**. Ollama is the
  recommended path for Mac and workstation use because it keeps model setup
  outside the Python dependency graph and avoids requiring cloud credentials.
- Hugging Face support should stay, but as an optional provider path: it is still
  useful for Hugging Face Spaces, hosted endpoint deployments, gated model
  experiments, and environments where a local model server is not practical.
- New features should not require Hugging Face tokens for the happy path. If a
  feature works locally, document the Ollama path first and the hosted path
  second.

## Loop Engineering Pattern
This repo is intentionally built around two loops:

- **Runtime document loop:** upload -> validate/decode -> split -> embed/index ->
  retrieve prompt chunks -> answer with inline citations -> run mechanical
  checks -> verify cited claims with the active real backend -> retry once or
  fail closed -> return trace/status.
- **Engineering loop:** change one contract -> add focused regressions -> run
  golden document evals -> run broad validation -> ask for review -> stage only
  intentional files.

Golden evals live in `tests/test_golden_document_eval.py`. They are provider-free
CI checks that exercise the full document QA loop with a deterministic fake LLM:

```bash
python -m pytest tests/test_golden_document_eval.py -q
python -m pytest
```

Use these before adding planner loops, tools, multi-document memory, or more
agent-like behavior. Blunt rule: if the boring document loop is not measurably
honest, agent features will only make the failure harder to see.

### Optional Live Ollama Model Eval

CI stays provider-free. When you want to compare pulled local Ollama models,
run the live eval command manually. Each case performs answer and verifier
calls, so start small on memory-constrained Macs. The live eval command only
accepts loopback Ollama URLs such as `http://localhost:11434` or
`http://127.0.0.1:11434`; it is not an arbitrary remote model benchmark tool.

```bash
python -m src.ollama_model_eval \
  --models nemotron-3-nano:4b \
  --case launch_date \
  --timeout 30 \
  --no-fail
```

Then run the full golden set for one model:

```bash
python -m src.ollama_model_eval \
  --models nemotron-3-nano:4b \
  --timeout 60 \
  --no-fail
```

The command refuses multiple models by default so a comparison run does not
accidentally overload a local Mac. Prefer one model per command. Only use the
override when you have enough free unified memory and are comfortable watching
resource pressure:

```bash
python -m src.ollama_model_eval \
  --models nemotron-3-nano:4b qwen3:8b \
  --allow-multi-model \
  --timeout 60 \
  --no-fail
```

The command asks Ollama to unload each model after its run by default. If your
machine still feels memory pressure, stop the run and inspect resident models:

```bash
ollama ps
ollama stop nemotron-3-nano:4b
ollama stop qwen3:8b
```

## Security and Dependency Maintenance
- Dependencies are pinned in `requirements.txt` for reproducible installs.
- Dependabot is enabled weekly (`.github/dependabot.yml`) for dependency updates.
- Current baseline includes Gradio `6.15.2`, LangChain `1.3.10`, LangChain-HuggingFace `1.2.1`, HuggingFace Hub `1.5.0`, Transformers `5.4.0`, and Torch `2.12.1`.
- Note: `marshmallow` is intentionally pinned to `3.26.2` because `dataclasses-json` currently requires `<4.0.0`.
- Security-sensitive transitive dependencies are explicitly pinned (for example `aiohttp`, `urllib3`, `python-multipart`, and `orjson`) to keep audit results stable.
- Recommended recurring checks:

  ```bash
  pip install -r requirements-dev.txt
  pytest tests/test_golden_document_eval.py -q
  pytest tests/test_ollama_model_eval.py -q
  pytest
  python -m pip_audit -r requirements.txt --strict
  python -m pip check
  ```

## Agent-Assisted Development
- `AGENTS.md` contains repo-level instructions for coding agents: setup commands,
  validation expectations, backend honesty rules, encoding policy, and release
  guardrails.
- `.agents/skills/document-qa/SKILL.md` defines the focused document-QA
  engineering skill for changes to ingestion, retrieval, model routing, UI
  status, and CI publishing.
- Use the documented loop for non-trivial changes: explore, plan, act, observe,
  verify, review, and ship.

## Contributing
Feel free to submit issues and pull requests. Contributions are welcome!

## License
This project is open source and available under the MIT License.
