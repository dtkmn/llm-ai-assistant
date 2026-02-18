import logging
import os
import sys
from getpass import getpass
from typing import Dict, List, Optional

import torch
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

LOGGER = logging.getLogger(__name__)
HF_TOKEN_ENV_VAR = "HUGGINGFACEHUB_API_TOKEN"
FAST_MODE_ENV_VAR = "FAST_MODE"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_DOCUMENT_BYTES = 25 * 1024 * 1024
MAX_DOCUMENT_CHUNKS = 2_000
DEFAULT_QUALITY_EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_FAST_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_QUALITY_MODELS = (
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
)
DEFAULT_FAST_MODELS = (
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
)
QUALITY_PROFILE = {
    "max_new_tokens": 384,
    "retrieval_k": 6,
    "retrieval_fetch_k": 24,
    "retrieval_lambda_mult": 0.7,
    "context_chunks": 6,
    "context_chars_per_chunk": 700,
    "context_total_chars": 4200,
    "splitter_chunk_size": 1200,
    "splitter_chunk_overlap": 200,
}
FAST_PROFILE = {
    "max_new_tokens": 160,
    "retrieval_k": 3,
    "retrieval_fetch_k": 10,
    "retrieval_lambda_mult": 0.8,
    "context_chunks": 3,
    "context_chars_per_chunk": 450,
    "context_total_chars": 1800,
    "splitter_chunk_size": 900,
    "splitter_chunk_overlap": 120,
}
DOCUMENT_IDENTITY_QUESTION_HINTS = (
    "uploaded document",
    "which document",
    "what document",
    "document name",
    "file name",
    "filename",
    "what file",
    "which file",
)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class MockLLM(LLM):
    """Fallback LLM when no HuggingFace token/model is available."""

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        if "context:" in prompt.lower():
            parts = prompt.split("Question:")
            if len(parts) > 1:
                question = parts[1].strip()
                return (
                    "Based on the provided context, I can identify information related to "
                    f"{question.lower()}. This is a demonstration response because no real "
                    "language model is currently configured."
                )
        return (
            "This is a mock response. Configure a valid HuggingFace token to enable real "
            "AI-powered answers."
        )


class DocumentQA:
    def __init__(
        self,
        model_id: Optional[str] = None,
        embeddings_model: Optional[str] = None,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        fast_mode: Optional[bool] = None,
        allow_interactive_token: bool = False,
        max_document_bytes: int = MAX_DOCUMENT_BYTES,
        max_document_chunks: int = MAX_DOCUMENT_CHUNKS,
    ):
        self.device = device or self._detect_device()
        self.fast_mode = env_flag(FAST_MODE_ENV_VAR, False) if fast_mode is None else fast_mode
        default_models = DEFAULT_FAST_MODELS if self.fast_mode else DEFAULT_QUALITY_MODELS
        default_embeddings_model = (
            DEFAULT_FAST_EMBEDDINGS_MODEL
            if self.fast_mode
            else DEFAULT_QUALITY_EMBEDDINGS_MODEL
        )
        self.model_id = model_id or os.getenv("LLM_MODEL_ID", default_models[0])
        self.embeddings_model = embeddings_model or default_embeddings_model
        self.hf_token = hf_token or self._get_hf_token(allow_interactive_token)
        self.max_document_bytes = max_document_bytes
        self.max_document_chunks = max_document_chunks
        self.profile = FAST_PROFILE if self.fast_mode else QUALITY_PROFILE

        self.llm = None
        self.loaded_model_id: Optional[str] = None
        self.embeddings = None
        self.vector_store = None
        self.retrieval_chain = None
        self.embeddings_error: Optional[str] = None
        self.current_document_name: Optional[str] = None
        self.chat_history: List[Dict[str, str]] = []
        self._initialize_llm()

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_hf_token(self, allow_interactive_token: bool) -> Optional[str]:
        token = os.getenv(HF_TOKEN_ENV_VAR, "").strip()
        if token:
            return token

        if not allow_interactive_token:
            return None

        if not sys.stdin or not sys.stdin.isatty():
            LOGGER.warning(
                "Interactive token prompt requested, but no TTY is available. "
                "Set %s in the environment instead.",
                HF_TOKEN_ENV_VAR,
            )
            return None

        try:
            token = getpass("Enter HF token (or press Enter to skip): ").strip()
        except (KeyboardInterrupt, EOFError):
            LOGGER.warning("No HuggingFace token provided.")
            return None

        if token:
            os.environ[HF_TOKEN_ENV_VAR] = token
            return token
        return None

    def _model_dtype(self) -> torch.dtype:
        if self.device in {"cuda", "mps"}:
            return torch.float16
        return torch.float32

    def _load_local_model(self, model_id: str) -> HuggingFacePipeline:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        LOGGER.info("Loading model %s on %s", model_id, self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.hf_token)

        model_kwargs = {
            "dtype": self._model_dtype(),
            "token": self.hf_token,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        if self.device == "mps":
            model = model.to("mps")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.profile["max_new_tokens"],
            do_sample=False,
            repetition_penalty=1.05,
            return_full_text=False,
        )
        return HuggingFacePipeline(pipeline=text_pipeline)

    def _candidate_models(self) -> List[str]:
        candidates = [self.model_id]
        default_models = DEFAULT_FAST_MODELS if self.fast_mode else DEFAULT_QUALITY_MODELS
        for fallback_model in default_models:
            if fallback_model not in candidates:
                candidates.append(fallback_model)
        return candidates

    def _initialize_llm(self) -> None:
        if self.hf_token == "dummy":
            LOGGER.warning("Dummy HuggingFace token detected. Falling back to MockLLM.")
            self.llm = MockLLM()
            return

        last_error = None
        for candidate_model in self._candidate_models():
            try:
                self.llm = self._load_local_model(candidate_model)
                self.loaded_model_id = candidate_model
                self.model_id = candidate_model
                LOGGER.info("Using model %s", candidate_model)
                return
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "Failed to load model %s: %s. Trying next fallback.",
                    candidate_model,
                    exc,
                )

        if last_error:
            LOGGER.exception("All model candidates failed. Falling back to MockLLM.")
        else:
            LOGGER.warning(
                "No model candidates available. Falling back to MockLLM. "
                "Set %s to enable the local model.",
                HF_TOKEN_ENV_VAR,
            )
        self.llm = MockLLM()

    def _initialize_embeddings(self) -> None:
        if self.embeddings is not None:
            return

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embeddings_model,
                model_kwargs={"device": self.device},
            )
        except Exception as exc:
            self.embeddings_error = str(exc)
            self.embeddings = None
            LOGGER.exception(
                "Embeddings initialization failed. Document processing will be unavailable."
            )

    def _create_retrieval_chain(self) -> None:
        if not self.llm:
            raise RuntimeError("LLM is not initialized.")
        if not self.vector_store:
            raise RuntimeError("Vector store is not initialized.")

        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.profile["retrieval_k"],
                "fetch_k": self.profile["retrieval_fetch_k"],
                "lambda_mult": self.profile["retrieval_lambda_mult"],
            },
        )
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful AI assistant. Answer using only the provided context. "
            "If the answer is not in the context, say that clearly.\n\n"
            "Uploaded document name: {document_name}\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        def format_docs(docs):
            filtered_chunks = []
            for doc in docs:
                content = doc.page_content.strip()
                if not content:
                    continue
                metadata = doc.metadata or {}
                source = metadata.get("source")
                source_name = (
                    os.path.basename(str(source))
                    if source
                    else (self.current_document_name or "uploaded document")
                )
                source_label = f"Source: {source_name}"
                page = metadata.get("page")
                if isinstance(page, int):
                    source_label = f"{source_label} (page {page + 1})"
                filtered_chunks.append(
                    f"{source_label}\n{content[: self.profile['context_chars_per_chunk']]}"
                )
            return "\n\n".join(filtered_chunks[: self.profile["context_chunks"]])[
                : self.profile["context_total_chars"]
            ]

        self.retrieval_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "document_name": lambda _: self.current_document_name or "unknown",
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _validate_document(self, document_path: str) -> str:
        if not os.path.exists(document_path):
            raise ValueError("Uploaded file could not be found.")

        file_extension = os.path.splitext(document_path)[1].lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_extension}")

        file_size = os.path.getsize(document_path)
        if file_size > self.max_document_bytes:
            limit_mb = self.max_document_bytes // (1024 * 1024)
            raise ValueError(
                f"File is too large ({file_size} bytes). Maximum supported size is {limit_mb} MB."
            )

        return file_extension

    def process_document(self, document_path: str) -> None:
        """Process a document and build a vector store."""
        try:
            file_extension = self._validate_document(document_path)
            self.current_document_name = os.path.basename(document_path)
            if not self.embeddings:
                self._initialize_embeddings()
            if not self.embeddings:
                raise RuntimeError(
                    "Embedding model is unavailable. Check network access/model cache and retry."
                )

            if file_extension == ".pdf":
                loader = PyPDFLoader(document_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(document_path)
            else:
                loader = TextLoader(document_path, encoding="utf-8", autodetect_encoding=True)

            documents = loader.load()
            if not documents:
                raise ValueError("No readable content found in the uploaded document.")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.profile["splitter_chunk_size"],
                chunk_overlap=self.profile["splitter_chunk_overlap"],
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
            )
            chunks = text_splitter.split_documents(documents)
            if not chunks:
                raise ValueError("No text chunks were generated from the document.")

            if len(chunks) > self.max_document_chunks:
                LOGGER.warning(
                    "Document produced %s chunks; truncating to %s to keep memory bounded.",
                    len(chunks),
                    self.max_document_chunks,
                )
                chunks = chunks[: self.max_document_chunks]

            self.vector_store = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
            self._create_retrieval_chain()
        except Exception as exc:
            raise RuntimeError(f"Error loading document: {exc}") from exc

    def _is_document_identity_question(self, prompt: str) -> bool:
        lowered = prompt.lower()
        return any(hint in lowered for hint in DOCUMENT_IDENTITY_QUESTION_HINTS)

    def query(self, prompt: str, session_id: str = "default") -> str:
        """Answer a user query using retrieved document context."""
        if not self.llm:
            return "Language model is not initialized. Please check your HuggingFace token setup."

        if not self.retrieval_chain:
            return "Please upload and process a document first."

        clean_prompt = prompt.strip()
        if not clean_prompt:
            return "Please provide a question."

        if self.current_document_name and self._is_document_identity_question(clean_prompt):
            return f"The uploaded document is `{self.current_document_name}`."

        try:
            response = self.retrieval_chain.invoke(clean_prompt)
            if not isinstance(response, str):
                response = str(response)
            response = response.strip()
            if len(response) < 3:
                response = (
                    "I could not find enough relevant information in the document to answer that."
                )

            self.chat_history.append(
                {"session_id": session_id, "question": clean_prompt, "answer": response}
            )
            return response
        except Exception:
            LOGGER.exception("Error while processing query.")
            return "I hit an internal error while processing your question. Please try again."
