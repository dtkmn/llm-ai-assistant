import torch
import os
from getpass import getpass
from typing import Optional, Dict

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent


class DocumentQA:
    def __init__(
        self,
        model_id="google/gemma-7b-it",
        embeddings_model="sentence-transformers/all-mpnet-base-v2",
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_id = model_id
        self.embeddings_model = embeddings_model
        self.hf_token = hf_token or self._get_hf_token()

        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.agent_executor = None
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
            model_name=self.embeddings_model, model_kwargs={"device": self.device}
        )

    def _create_agent(self):
        """Create the agent with a retrieval tool"""
        retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3, "lambda_mult": 0.25}
        )

        # Create a tool for the retriever
        tools = [
            create_retrieval_chain(
                retriever,
                create_stuff_documents_chain(
                    self.llm,
                    ChatPromptTemplate.from_messages(
                        [
                            ("system", self._get_qa_system_prompt()),
                            ("human", "{input}"),
                        ]
                    ),
                ),
            )
        ]

        # Create the agent
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._get_contextualize_prompt()),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    def process_document(self, document_path: str):
        """Process a document and create a vector store"""
        try:
            file_extension = os.path.splitext(document_path)[1].lower()
            if file_extension == ".pdf":
                loader = PyPDFLoader(document_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(document_path)
            elif file_extension in [".txt", ".md"]:
                loader = TextLoader(document_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=128,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            )
            texts = text_splitter.split_documents(documents)

            self.vector_store = FAISS.from_documents(
                documents=texts, embedding=self.embeddings
            )
            self._create_agent()
        except Exception as e:
            raise RuntimeError(f"Error loading document: {e}") from e

    def query(self, prompt: str, session_id: str = "default") -> str:
        """Process a user query with conversation history"""
        if not self.agent_executor:
            return "Please upload a document first."

        history = self.get_session_history(session_id)

        response = self.agent_executor.invoke(
            {
                "input": prompt,
                "chat_history": history.messages,
            }
        )

        history.add_user_message(prompt)
        history.add_ai_message(response["output"])

        return response["output"]

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for a session"""
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
        return self.session_store[session_id]

    @staticmethod
    def _get_qa_system_prompt() -> str:
        return """
            You are a helpful assistant that can answer questions based on a document's content.
            You have access to relevant excerpts of the document and the conversation so far.
            Use these excerpts to answer the user's questions in a clear and concise manner.
            
            If the user asks a question not answered by the provided context or the conversation, 
            simply respond with "I do not know." 
            Do not reveal or invent any details that are not supported by the document context.
            
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
