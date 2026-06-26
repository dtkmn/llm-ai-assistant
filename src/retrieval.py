from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

try:
    from .native_runtime import apply_native_runtime_defaults
except ImportError:
    from native_runtime import apply_native_runtime_defaults


apply_native_runtime_defaults()

import faiss
import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

try:
    from .retrieval_types import AnswerCitation, RetrievalChainResult, RetrievedContext
except ImportError:
    from retrieval_types import AnswerCitation, RetrievalChainResult, RetrievedContext


class FaissVectorStore:
    def __init__(self, documents: List[Document], embeddings, vectors: np.ndarray):
        if vectors.ndim != 2:
            raise ValueError("Embedding vectors must be a 2D array.")
        self.documents = documents
        self.embeddings = embeddings
        self.vectors = vectors.astype("float32")
        self.index = faiss.IndexFlatL2(self.vectors.shape[1])
        self.index.add(self.vectors)

    @classmethod
    def from_documents(cls, documents: List[Document], embedding) -> "FaissVectorStore":
        texts = [document.page_content for document in documents]
        vectors = np.asarray(embedding.embed_documents(texts), dtype="float32")
        if len(vectors) != len(documents):
            raise ValueError("Embedding count does not match document count.")
        if len(documents) == 0:
            raise ValueError("Cannot build a vector store from zero documents.")
        return cls(documents, embedding, vectors)

    def as_retriever(
        self, search_type: str = "similarity", search_kwargs: Optional[Dict] = None
    ) -> "FaissRetriever":
        return FaissRetriever(
            vector_store=self,
            search_type=search_type,
            search_kwargs=search_kwargs or {},
        )

    def search(self, query: str, search_type: str, search_kwargs: Dict) -> List[Document]:
        k = int(search_kwargs.get("k", 4))
        if k <= 0:
            return []

        query_vector = np.asarray([self.embeddings.embed_query(query)], dtype="float32")
        if search_type == "mmr":
            fetch_k = min(
                int(search_kwargs.get("fetch_k", max(k, 20))), len(self.documents)
            )
            _, indices = self.index.search(query_vector, fetch_k)
            candidate_indices = [int(index) for index in indices[0] if index >= 0]
            selected_indices = self._maximal_marginal_relevance(
                query_vector[0],
                candidate_indices,
                k,
                float(search_kwargs.get("lambda_mult", 0.5)),
            )
            return [self.documents[index] for index in selected_indices]

        search_k = min(k, len(self.documents))
        _, indices = self.index.search(query_vector, search_k)
        return [self.documents[int(index)] for index in indices[0] if index >= 0]

    def _maximal_marginal_relevance(
        self,
        query_vector: np.ndarray,
        candidate_indices: List[int],
        k: int,
        lambda_mult: float,
    ) -> List[int]:
        if not candidate_indices:
            return []

        selected: List[int] = []
        query_norm = self._normalize(query_vector)
        candidate_vectors = self.vectors[candidate_indices]
        candidate_norms = self._normalize_rows(candidate_vectors)
        query_similarities = candidate_norms @ query_norm

        while candidate_indices and len(selected) < k:
            if not selected:
                best_position = int(np.argmax(query_similarities))
            else:
                selected_vectors = self._normalize_rows(self.vectors[selected])
                diversity_penalties = candidate_norms @ selected_vectors.T
                max_diversity_penalties = diversity_penalties.max(axis=1)
                scores = (
                    lambda_mult * query_similarities
                    - (1.0 - lambda_mult) * max_diversity_penalties
                )
                best_position = int(np.argmax(scores))

            selected.append(candidate_indices.pop(best_position))
            candidate_norms = np.delete(candidate_norms, best_position, axis=0)
            query_similarities = np.delete(query_similarities, best_position, axis=0)

        return selected

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _normalize_rows(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)


class FaissRetriever(BaseRetriever):
    vector_store: FaissVectorStore
    search_type: str = "similarity"
    search_kwargs: Dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.vector_store.search(query, self.search_type, self.search_kwargs)


class DocumentRetrievalChain:
    def __init__(
        self,
        *,
        retriever: FaissRetriever,
        prompt: ChatPromptTemplate,
        llm: LLM,
        document_name: Optional[str],
        profile: Dict,
    ):
        self.retriever = retriever
        self.llm = llm
        self.answer_chain = prompt | llm | StrOutputParser()
        self.document_name = document_name
        self.profile = profile

    def invoke(self, question: str) -> str:
        return self.invoke_with_trace(question).answer

    def invoke_with_trace(
        self, question: str, self_check_instruction: str = ""
    ) -> RetrievalChainResult:
        retrieved_context = self.retrieve_with_trace(question)
        return self.draft_with_trace(
            question,
            retrieved_context,
            self_check_instruction=self_check_instruction,
        )

    def retrieve_with_trace(self, question: str) -> RetrievedContext:
        docs = self.retriever.invoke(question)
        context, citations = self._format_context_and_citations(docs)
        return RetrievedContext(
            retrieved_chunk_count=len(citations),
            citations=citations,
            context=context,
        )

    def draft_with_trace(
        self,
        question: str,
        retrieved_context: RetrievedContext,
        self_check_instruction: str = "",
    ) -> RetrievalChainResult:
        response = self._generate_answer(
            question=question,
            context=retrieved_context.context,
            self_check_instruction=self_check_instruction,
        )
        return RetrievalChainResult(
            answer=response,
            retrieved_chunk_count=retrieved_context.retrieved_chunk_count,
            citations=retrieved_context.citations,
            context=retrieved_context.context,
            model_thinking=self._latest_model_thinking(),
        )

    def retry_with_trace(
        self,
        question: str,
        previous_result: RetrievalChainResult,
        self_check_instruction: str,
    ) -> RetrievalChainResult:
        response = self._generate_answer(
            question=question,
            context=previous_result.context,
            self_check_instruction=self_check_instruction,
        )
        return RetrievalChainResult(
            answer=response,
            retrieved_chunk_count=previous_result.retrieved_chunk_count,
            citations=previous_result.citations,
            context=previous_result.context,
            model_thinking=self._latest_model_thinking(),
        )

    def _latest_model_thinking(self) -> Optional[str]:
        thinking = getattr(self.llm, "last_thinking", None)
        if not isinstance(thinking, str):
            return None
        thinking = thinking.strip()
        return thinking or None

    def _generate_answer(
        self, *, question: str, context: str, self_check_instruction: str = ""
    ) -> str:
        response = self.answer_chain.invoke(
            {
                "context": context,
                "question": question,
                "document_name": self.document_name or "unknown",
                "self_check_instruction": self_check_instruction,
            }
        )
        return str(response).strip()

    def _format_context_and_citations(
        self, docs: List[Document]
    ) -> Tuple[str, List[AnswerCitation]]:
        context_parts = []
        citations = []
        remaining_chars = self.profile["context_total_chars"]

        for doc in docs:
            if len(citations) >= self.profile["context_chunks"] or remaining_chars <= 0:
                break

            content = doc.page_content.strip()
            if not content:
                continue

            citation_id = len(citations) + 1
            metadata = doc.metadata or {}
            source_name = self._source_name(metadata)
            page = metadata.get("page")
            display_page = page + 1 if isinstance(page, int) else None
            chunk_index = metadata.get("chunk_index")
            if not isinstance(chunk_index, int):
                chunk_index = None

            source_label = f"[{citation_id}] Source: {source_name}"
            if display_page is not None:
                source_label = f"{source_label} (page {display_page})"
            if chunk_index is not None:
                source_label = f"{source_label}, chunk {chunk_index + 1}"

            allowed_content_chars = min(
                self.profile["context_chars_per_chunk"],
                max(0, remaining_chars - len(source_label) - 1),
            )
            if allowed_content_chars <= 0:
                break

            excerpt = content[:allowed_content_chars]
            context_entry = f"{source_label}\n{excerpt}"
            context_parts.append(context_entry)
            remaining_chars -= len(context_entry) + 2
            citations.append(
                AnswerCitation(
                    citation_id=citation_id,
                    source_name=source_name,
                    page=display_page,
                    chunk_index=chunk_index,
                    excerpt=" ".join(excerpt.split()),
                )
            )

        return "\n\n".join(context_parts), citations

    def _source_name(self, metadata: Dict) -> str:
        source = metadata.get("source")
        if source:
            return os.path.basename(str(source))
        return self.document_name or "uploaded document"


def build_document_retrieval_chain(
    *,
    vector_store: FaissVectorStore,
    llm: LLM,
    document_name: Optional[str],
    profile: Dict,
) -> DocumentRetrievalChain:
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": profile["retrieval_k"],
            "fetch_k": profile["retrieval_fetch_k"],
            "lambda_mult": profile["retrieval_lambda_mult"],
        },
    )
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful AI assistant. Answer using only the provided context. "
        "Cite relevant sources inline with bracketed numbers like [1]. "
        "If the answer is not in the context, say that clearly and do not invent citations.\n\n"
        "Uploaded document name: {document_name}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "{self_check_instruction}\n\n"
        "Answer:"
    )

    return DocumentRetrievalChain(
        retriever=retriever,
        prompt=prompt,
        llm=llm,
        document_name=document_name,
        profile=profile,
    )
