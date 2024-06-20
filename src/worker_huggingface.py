import os
import torch
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub


# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

# Function to initialize the language model and its embeddings
def init_llm():
    global llm_hub, embeddings

    # Hugging Face API token
    # Setup environment variable HUGGINGFACEHUB_API_TOKEN

    # repo name for the model
    # model_id = "facebook/blenderbot-400M-distill"
    # model_id = "tiiuae/falcon-7b-instruct"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Define the model parameters
    model_kwargs = {
        "temperature": 0.1,  # Lower temperature for more focused outputs
        "top_k": 10,         # Use top-k sampling
        "top_p": 0.9,        # Use top-p (nucleus) sampling
        "max_length": 512,   # Limit the length of the response
        "repetition_penalty": 1.2  # Penalize repeated phrases
    }
    # load the model into the HuggingFaceHub
    llm_hub = HuggingFaceHub(repo_id=model_id, model_kwargs=model_kwargs)

    # #Initialize embeddings using a pre-trained model to represent the text data.
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': DEVICE}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


# Function to process a PDF document
def process_document(document_path):
    global conversation_retrieval_chain

    # Load the document
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    texts = text_splitter.split_documents(documents)
    
    # Create an embeddings database using Chroma from the split text chunks.
    # text_embeddings = [embeddings.encode(text.content) for text in texts]
    db = Chroma.from_documents(documents=texts, embedding=embeddings)

    # --> Build the QA chain, which utilizes the LLM and retriever for answering questions. 
    # By default, the vectorstore retriever uses similarity search. 
    # If the underlying vectorstore support maximum marginal relevance search, you can specify that as the search type (search_type="mmr").
    # You can also specify search kwargs like k to use when doing retrieval. k represent how many search results send to llm
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key = "question"
     #   chain_type_kwargs={"prompt": prompt} # if you are using prompt template, you need to uncomment this part
    )


# Function to process a user prompt
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history
    
    improved_prompt = f"Use the given prompt to answer the question. If you don't know the answer, say you don't know. Use three sentence maximum and keep the answer concise. Prompt: {prompt}"

    # Query the model
    output = conversation_retrieval_chain.invoke({"question": improved_prompt, "chat_history": chat_history})
    answer = output["result"]
    print(output)

    # Extract the 'Helpful Answer:' part
    helpful_answer_index = answer.find("Helpful Answer:")
    if helpful_answer_index != -1:
        helpful_answer = answer[helpful_answer_index + len("Helpful Answer:"):].strip()
    else:
        helpful_answer = answer

    # Update the chat history
    chat_history.append((prompt, helpful_answer))
    
    # Return the model's response
    return helpful_answer

# Initialize the language model
init_llm()