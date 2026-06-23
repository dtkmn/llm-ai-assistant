from dataclasses import dataclass
from typing import Tuple

from langchain_core.embeddings import Embeddings


GOLDEN_DOCUMENT_TEXT = """# Project Phoenix Brief

Project Phoenix launches in June 2026.
The approved Project Phoenix budget is $42 million.
Alex Rivera owns the Project Phoenix rollout.
Unsupported claims must be refused instead of invented.
"""


@dataclass(frozen=True)
class GoldenEvalCase:
    case_id: str
    question: str
    expected_terms: Tuple[str, ...] = ()
    expect_refusal: bool = False


GOLDEN_EVAL_CASES = (
    GoldenEvalCase(
        case_id="launch_date",
        question="When does Project Phoenix launch?",
        expected_terms=("June 2026",),
    ),
    GoldenEvalCase(
        case_id="budget",
        question="What is the approved Project Phoenix budget?",
        expected_terms=("$42 million",),
    ),
    GoldenEvalCase(
        case_id="owner",
        question="Who owns the Project Phoenix rollout?",
        expected_terms=("Alex Rivera",),
    ),
    GoldenEvalCase(
        case_id="unsupported_venue",
        question="Where is the Project Phoenix launch venue?",
        expect_refusal=True,
    ),
)


class GoldenEvalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        lower = text.lower()
        return [
            float(len(text)),
            float(lower.count("phoenix")),
            float(lower.count("launch")),
            float(lower.count("budget")),
            float(lower.count("owner") + lower.count("owns")),
            float(lower.count("venue") + lower.count("location")),
        ]
