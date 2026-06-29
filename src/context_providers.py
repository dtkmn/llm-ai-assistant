from dataclasses import dataclass
from typing import Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieval import FaissVectorStore


class ContextProvider(Protocol):
    provider_type: str
    display_name: Optional[str]
    vector_store: Optional["FaissVectorStore"]
    retrieval_chain: Optional[object]


@dataclass(frozen=True)
class DocumentContextProvider:
    document_name: Optional[str]
    vector_store: Optional["FaissVectorStore"]
    retrieval_chain: Optional[object]
    provider_type: str = "document"

    @property
    def display_name(self) -> Optional[str]:
        return self.document_name

    @property
    def ready(self) -> bool:
        return self.retrieval_chain is not None


@dataclass(frozen=True)
class WebSearchContextProvider:
    display_name: Optional[str] = "DuckDuckGo web snippets"
    vector_store: Optional["FaissVectorStore"] = None
    retrieval_chain: Optional[object] = None
    provider_type: str = "web"

    @property
    def ready(self) -> bool:
        return self.retrieval_chain is not None


@dataclass(frozen=True)
class ActiveDocumentState:
    document_name: Optional[str]
    vector_store: Optional["FaissVectorStore"]
    retrieval_chain: Optional[object]
    context_provider: Optional[ContextProvider] = None
