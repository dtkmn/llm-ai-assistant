---
title: LLM Powered AI Assistant
emoji: 🏳️‍🌈
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
---


# LLM AI Assistant
This project leverages LangChain and LLM to create an AI assistant capable of answering questions based on provided document context and chat history.

https://huggingface.co/spaces/0xdant/llm-ai-assistant

## Features
- **Prompt Template Integration:** Utilizes custom chat prompt templates for context-aware responses.
- **Chat History Management:** Maintains session-specific chat histories to enhance the assistant's understanding and coherence.
- **Document Processing:** Processes PDF documents to extract relevant information using PyPDFLoader and FAISS.
- **GPU Support:** Automatically utilizes available GPU for faster computation.


![LLM-flow.png](https://github.com/dtkmn/llm-ai-assistant/blob/main/LLM-flow.png)

## Installation

1. Clone the repository:
    
    ```bash
    git clone https://github.com/dtkmn/llm-ai-assistant.git
    cd llm-ai-assistant
    ``` 

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment variables (create a .env file):    

    ```bash
    export HUGGINGFACEHUB_API_TOKEN=your_token_here
    ```

4. Run the application:

    ```bash
    python src/app.py
    ```
   
## 🐳 Docker Setup (Optional)

### Build and Run the Container

   ```bash
   docker build -t ai-assistant:latest .
   docker run -p 7860:7860 -e HUGGINGFACEHUB_API_TOKEN=your-token-here ai-assistant:latest
   ```


## Usage
- Open your browser and go to `http://localhost:7860`
- Upload a PDF document and interact with the chat interface.

## Contributing
Feel free to submit issues and pull requests. Contributions are welcome!

