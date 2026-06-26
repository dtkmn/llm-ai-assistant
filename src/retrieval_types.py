from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class AnswerCitation:
    citation_id: int
    source_name: str
    page: Optional[int]
    chunk_index: Optional[int]
    excerpt: str


@dataclass(frozen=True)
class RetrievalChainResult:
    answer: str
    retrieved_chunk_count: int
    citations: List[AnswerCitation]
    context: str = ""
    model_thinking: Optional[str] = None


@dataclass(frozen=True)
class RetrievedContext:
    retrieved_chunk_count: int
    citations: List[AnswerCitation]
    context: str
