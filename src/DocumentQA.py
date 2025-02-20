import torch
import os
from getpass import getpass
from typing import Optional, Dict

from langchain_huggingface import HuggingFaceEmbeddings
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


class DocumentQA:
    def __init__(
        self,
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        hf_token = None,
        device: Optional[str] = None
    ):
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.embeddings_model = embeddings_model
        self.hf_token = hf_token or self._get_hf_token()

        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.retrieval_chain = None
        self.session_store: Dict[str, BaseChatMessageHistory] = {}
        self._initialize_models()

    def _get_hf_token(self):
        if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass("Enter HF token: ")
        return os.environ["HUGGINGFACEHUB_API_TOKEN"]

    def _initialize_models(self):
        self.llm = HuggingFaceEndpoint(
            repo_id=self.model_id,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            return_full_text=False,
            temperature=0.01,
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embeddings_model,
            model_kwargs={"device": self.device}
        )

    def _create_retrieval_chain(self):
        """Create the conversation chain with history handling"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._get_contextualize_prompt()),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 3, 'lambda_mult': 0.25}
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_qa_system_prompt()),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        self.retrieval_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )


    def process_document(self, document_path: str):
        """Process a PDF document and create vector store"""
        try:
            loader = PyPDFLoader(document_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=128,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  # Add empty string as separator
            )
            texts = text_splitter.split_documents(documents)
            print(f"Number of text chunks: {len(texts)}")  # Debug

            self.vector_store = FAISS.from_documents(
                documents=texts,
                embedding=self.embeddings
            )
            self._create_retrieval_chain()
        except Exception as e:
            raise RuntimeError(f"Error loading document: {e}") from e

    def query(self, prompt: str, session_id: str = "default") -> str:
        """Process a user query with conversation history"""
        if not self.retrieval_chain:
            raise RuntimeError("No document processed. Call process_document() first.")

        response = self.retrieval_chain.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}}
        )
        print(response)  # Debug

        return response["answer"]

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for a session"""
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
        return self.session_store[session_id]

    @staticmethod
    def _get_qa_system_prompt() -> str:
        return """
            You are a helpful assistant that can answer questions based on a PDF's content. 
            You have access to relevant excerpts of the PDF and the conversation so far. 
            Use these excerpts to answer the user's questions in a clear and concise manner.
            
            If the user asks a question not answered by the provided context or the conversation, 
            simply respond with "I do not know." 
            Do not reveal or invent any details that are not supported by the PDF context.
            
            Maintain a neutral, professional tone. 
            When you do not have enough context, say "I do not know." 
            Do not provide chain-of-thought or internal reasoning to the user.

            Question: {input}
            Context: {context}
            """

    @staticmethod
    def _get_contextualize_prompt() -> str:
        return """Given a chat history and the latest user question 
            which might reference context in the chat history, 
            formulate a standalone question which can be understood 
            without the chat history. Do NOT answer the question, 
            just reformulate it if needed and otherwise return it as is."""
