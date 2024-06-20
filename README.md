# LLM Powered AI Assistant
This project is a personal data assistant using a Language Learning Model (LLM) to process and answer questions related to uploaded PDF documents.


---
title: LLM Powered AI Assistant
emoji: 🏳️‍🌈
colorFrom: gray
colorTo: green
sdk: docker
app_file: app.py
app_port: 7860
pinned: true
---


## Features
- Upload PDF documents for analysis
- Interactive chat interface
- Toggle between light and dark modes


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

## Usage
- Open your browser and go to `http://localhost:5000`
- Upload a PDF document and interact with the chat interface.

## Contributing
Feel free to submit issues and pull requests. Contributions are welcome!

