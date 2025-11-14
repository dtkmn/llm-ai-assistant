import torch
import os
from getpass import getpass
from typing import Optional, Dict, List

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM


class MockLLM(LLM):
    """A simple mock LLM for demonstration when HuggingFace is not available"""
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Mock response based on the context"""
        if "context:" in prompt.lower():
            # Extract context and question from the prompt
            parts = prompt.split("Question:")
            if len(parts) > 1:
                question = parts[1].strip()
                context_parts = prompt.split("Context:")
                if len(context_parts) > 1:
                    context = context_parts[1].split("Question:")[0].strip()
                    return f"Based on the provided context, I can see information about {question.lower()}. However, this is a demonstration response since no actual language model is configured. Please provide a valid HuggingFace API token to get real AI-powered responses."
        
        return "This is a mock response. Please configure a proper HuggingFace API token to get actual AI responses."


class DocumentQA:
    def __init__(
        self,
        model_id="microsoft/DialoGPT-medium",
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
        self.retrieval_chain = None
        self.chat_history: List[Dict[str, str]] = []
        self._initialize_models()

    def _get_hf_token(self):
        if "HUGGINGFACEHUB_API_TOKEN" in os.environ:
            return os.environ["HUGGINGFACEHUB_API_TOKEN"]
        try:
            token = getpass("Enter HF token (or press Enter to skip): ")
            if token.strip():
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
                return token
        except KeyboardInterrupt:
            print("\nNo HuggingFace token provided. Some features may not work.")
        return None

    def _initialize_models(self):
        try:
            # Try with a valid token first, then fallback to a simple mock
            if self.hf_token and self.hf_token != "dummy":
                # Use local pipeline approach to avoid Hub endpoint issues
                try:
                    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                    import torch
                    
                    print("Attempting to load model locally...")
                    
                    # Use Llama 3.2 3B from HuggingFace - much better quality
                    model_name = "meta-llama/Llama-3.2-3B-Instruct"
                    
                    print(f"Loading {model_name} - high-quality Llama model from HuggingFace...")
                    print("This will download ~6GB on first run. Please wait...")
                    
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        token=self.hf_token
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16,
                        token=self.hf_token
                    )
                    
                    # Set up padding
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    # Create pipeline optimized for Q&A
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=512,  # Increased from 150 to allow longer, more detailed responses
                        temperature=0.3,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        return_full_text=False
                    )
                    
                    self.llm = HuggingFacePipeline(pipeline=pipe)
                    print(f"Successfully initialized local {model_name} model")
                    
                except Exception as e1:
                    print(f"Local model loading failed: {e1}")
                    print("Falling back to mock LLM for demonstration.")
                    self.llm = MockLLM()
            else:
                print("Warning: No valid HuggingFace token provided. Using mock LLM for demonstration.")
                self.llm = MockLLM()
        except Exception as e:
            print(f"Warning: Could not initialize any model: {e}")
            print("Falling back to mock LLM for demonstration.")
            self.llm = MockLLM()

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embeddings_model, model_kwargs={"device": self.device}
        )

    def _create_retrieval_chain(self):
        """Create a simple retrieval chain"""
        if not self.llm:
            print("Warning: Cannot create retrieval chain without initialized LLM")
            return
            
        retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}  # Get more chunks to find better content
        )
        
        prompt = ChatPromptTemplate.from_template(
            """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant. Answer questions based on the provided context accurately and concisely.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        )
        
        def format_docs(docs):
            # Filter and format docs, prefer non-code content
            filtered_docs = []
            for doc in docs:
                content = doc.page_content
                # Skip if it's mostly code (lots of semicolons, brackets)
                code_chars = content.count(';') + content.count('{') + content.count('}')
                if code_chars < len(content) * 0.1:  # Less than 10% code characters
                    filtered_docs.append(content[:400])
            
            # If all docs are code, use them anyway but limit heavily
            if not filtered_docs:
                filtered_docs = [doc.page_content[:300] for doc in docs[:2]]
            
            formatted = "\n\n".join(filtered_docs[:3])  # Max 3 docs
            return formatted[:900]  # Keep under 512 tokens
        
        self.retrieval_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def process_document(self, document_path: str):
        """Process a document and create a vector store"""
        try:
            file_extension = os.path.splitext(document_path)[1].lower()
            if file_extension == ".pdf":
                loader = PyPDFLoader(document_path)
            elif file_extension == ".docx":
                # Also ensuring my previous typo is GONE
                loader = Docx2txtLoader(document_path)
            elif file_extension in [".txt", ".md"]:
                loader = TextLoader(document_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,       # Smaller chunks for better retrieval
                chunk_overlap=100,     # Less overlap
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
            )
            texts = text_splitter.split_documents(documents)

            self.vector_store = FAISS.from_documents(
                documents=texts, embedding=self.embeddings
            )
            self._create_retrieval_chain()
        except Exception as e:
            raise RuntimeError(f"Error loading document: {e}") from e

    def query(self, prompt: str, session_id: str = "default") -> str:
        """Process a user query"""
        if not self.llm:
            return "Error: Language model not initialized. Please check your HuggingFace API token."
        
        if not self.retrieval_chain:
            return "Please upload a document first."

        try:
            print(f"Processing query: {prompt}")  # Debug log
            response = self.retrieval_chain.invoke(prompt)
            print(f"Raw response: {repr(response)}")  # Debug log
            
            # FLAN-T5 gives clean responses, minimal cleaning needed
            if response:
                response = response.strip()
                
                # Just ensure it's not empty
                if not response or len(response) < 3:
                    response = "I couldn't find relevant information in the document to answer that question."
                
            # Handle empty or None response
            if not response or response.strip() == "":
                response = "I couldn't generate a response. Please try rephrasing your question."
            
            # Store conversation history
            self.chat_history.append({"question": prompt, "answer": response})
            
            return response
        except Exception as e:
            print(f"Full error details: {type(e).__name__}: {str(e)}")  # Debug log
            import traceback
            traceback.print_exc()  # Print full traceback
            return f"Error processing query: {type(e).__name__}: {str(e)}"
