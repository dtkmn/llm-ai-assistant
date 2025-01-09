import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import MessagesPlaceholder

from langchain.chains import create_history_aware_retriever

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None
tokenizer = None

# Function to initialize the language model and its embeddings
def init_llm():
    global llm_hub, embeddings

    # Hugging Face API token
    # Setup environment variable HUGGINGFACEHUB_API_TOKEN

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    llm_hub = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
        temperature=0.1,
    )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Function to process a PDF document
def process_document(document_path):
    global conversation_retrieval_chain

    # Load the document
    loader = PyPDFLoader(document_path)
    try:
        documents = loader.load()
    except Exception as e:
        print(f"Error loading document: {e}")
        return None    
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    texts = text_splitter.split_documents(documents)
    
    # Create an embeddings database using FAISS from the split text chunks.
    db = FAISS.from_documents(documents=texts, embedding=embeddings)

    system_prompt = """
    <|start_header_id|>user<|end_header_id|>
    You are an assistant for answering questions using provided context.
    You are given the extracted parts of a long document, previous chat_history and a question. Provide a conversational answer.
    If you don't know the answer, just say "I do not know." Don't make up an answer.
    Question: {input}
    Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    retriever=db.as_retriever(search_type="similarity", search_kwargs={'k': 3, 'lambda_mult': 0.25})
    question_answer_chain = create_stuff_documents_chain(llm_hub, prompt)    
    # conversation_retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm_hub, retriever, contextualize_q_prompt
    )

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversation_retrieval_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


# Function to process a user prompt
def process_prompt(prompt):
    # global conversation_retrieval_chain
    global chat_history   
    
    # Query the model with history    
    output = conversation_retrieval_chain.invoke(
        {"input": prompt},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )
    answer = output["answer"]
    print(output)
    
    # Return the model's response
    return answer

# Initialize the language model
init_llm()