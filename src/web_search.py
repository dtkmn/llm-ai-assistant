from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:
    from .retrieval_types import AnswerCitation, RetrievalChainResult, RetrievedContext
except ImportError:
    from retrieval_types import AnswerCitation, RetrievalChainResult, RetrievedContext


DUCKDUCKGO_INSTANT_ANSWER_URL = "https://api.duckduckgo.com/"
WEB_SEARCH_USER_AGENT = "AI-Loop-Engine/0.1 (+local-first evidence retrieval)"
MAX_WEB_SEARCH_QUERY_CHARS = 500
MAX_WEB_SEARCH_RESPONSE_BYTES = 1_000_000
MAX_WEB_SEARCH_RESULTS = 5
MAX_WEB_SEARCH_SNIPPET_CHARS = 700


class WebSearchError(RuntimeError):
    """Raised when the configured web search provider cannot return evidence."""


@dataclass(frozen=True)
class WebSearchHit:
    title: str
    url: str
    snippet: str


def _clean_text(value: Any, *, max_chars: int = MAX_WEB_SEARCH_SNIPPET_CHARS) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split()).strip()
    return text[:max_chars]


def _safe_result_url(value: Any) -> str:
    raw_url = str(value or "").strip()
    if not raw_url:
        return ""
    try:
        parsed = urllib.parse.urlsplit(raw_url)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    host = parsed.hostname or ""
    if not host:
        return ""
    netloc = host
    try:
        port = parsed.port
    except ValueError:
        return ""
    if port is not None:
        netloc = f"{netloc}:{port}"
    path = parsed.path or ""
    return urllib.parse.urlunsplit((parsed.scheme, netloc, path, "", ""))


def _title_from_hit(*values: Any, fallback_url: str = "") -> str:
    for value in values:
        text = _clean_text(value, max_chars=100)
        if text:
            return text
    if fallback_url:
        try:
            parsed = urllib.parse.urlsplit(fallback_url)
        except ValueError:
            return "Web result"
        if parsed.hostname:
            return parsed.hostname
    return "Web result"


def _dedupe_hits(hits: Iterable[WebSearchHit], *, limit: int) -> List[WebSearchHit]:
    deduped: List[WebSearchHit] = []
    seen = set()
    for hit in hits:
        key = (hit.url, hit.snippet.casefold())
        if not hit.snippet or key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
        if len(deduped) >= limit:
            break
    return deduped


class DuckDuckGoInstantAnswerSearch:
    def __init__(
        self,
        *,
        timeout: int = 12,
        opener: Optional[Any] = None,
    ):
        self.timeout = max(1, int(timeout))
        self.opener = opener or urllib.request.urlopen

    def search(self, query: str, *, max_results: int = MAX_WEB_SEARCH_RESULTS) -> List[WebSearchHit]:
        clean_query = " ".join(str(query or "").split()).strip()
        if not clean_query:
            return []
        clean_query = clean_query[:MAX_WEB_SEARCH_QUERY_CHARS]
        params = urllib.parse.urlencode(
            {
                "q": clean_query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
        )
        request = urllib.request.Request(
            f"{DUCKDUCKGO_INSTANT_ANSWER_URL}?{params}",
            headers={
                "accept": "application/json",
                "user-agent": WEB_SEARCH_USER_AGENT,
            },
            method="GET",
        )
        try:
            with self.opener(request, timeout=self.timeout) as response:
                body = response.read(MAX_WEB_SEARCH_RESPONSE_BYTES + 1)
        except (OSError, urllib.error.URLError, ValueError) as exc:
            raise WebSearchError("web_search_request_failed") from exc
        if len(body) > MAX_WEB_SEARCH_RESPONSE_BYTES:
            raise WebSearchError("web_search_response_too_large")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebSearchError("web_search_invalid_json") from exc
        if not isinstance(payload, dict):
            raise WebSearchError("web_search_invalid_payload")
        return parse_duckduckgo_instant_answer(payload, max_results=max_results)


def parse_duckduckgo_instant_answer(
    payload: Dict[str, Any], *, max_results: int = MAX_WEB_SEARCH_RESULTS
) -> List[WebSearchHit]:
    hits: List[WebSearchHit] = []
    abstract = _clean_text(payload.get("AbstractText"))
    abstract_url = _safe_result_url(payload.get("AbstractURL"))
    if abstract:
        title = _title_from_hit(
            payload.get("Heading"),
            payload.get("AbstractSource"),
            fallback_url=abstract_url,
        )
        hits.append(WebSearchHit(title=title, url=abstract_url, snippet=abstract))

    answer = _clean_text(payload.get("Answer"))
    if answer:
        title = _title_from_hit(payload.get("AnswerType"), payload.get("Heading"))
        hits.append(WebSearchHit(title=title, url=abstract_url, snippet=answer))

    definition = _clean_text(payload.get("Definition"))
    definition_url = _safe_result_url(payload.get("DefinitionURL")) or abstract_url
    if definition:
        title = _title_from_hit(
            payload.get("DefinitionSource"),
            payload.get("Heading"),
            fallback_url=definition_url,
        )
        hits.append(WebSearchHit(title=title, url=definition_url, snippet=definition))

    def visit_related(values: Iterable[Any]) -> None:
        for value in values:
            if not isinstance(value, dict):
                continue
            nested = value.get("Topics")
            if isinstance(nested, list):
                visit_related(nested)
                continue
            snippet = _clean_text(value.get("Text"))
            url = _safe_result_url(value.get("FirstURL"))
            if snippet:
                title = _title_from_hit(value.get("Name"), fallback_url=url)
                hits.append(WebSearchHit(title=title, url=url, snippet=snippet))

    related_topics = payload.get("RelatedTopics")
    if isinstance(related_topics, list):
        visit_related(related_topics)

    return _dedupe_hits(hits, limit=max(0, int(max_results)))


class WebSearchRetrievalChain:
    def __init__(
        self,
        *,
        search_client: DuckDuckGoInstantAnswerSearch,
        llm,
        profile: Dict[str, int],
        max_results: int = MAX_WEB_SEARCH_RESULTS,
    ):
        self.search_client = search_client
        self.llm = llm
        self.profile = profile
        self.max_results = max(1, min(int(max_results), MAX_WEB_SEARCH_RESULTS))

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
        hits = self.search_client.search(question, max_results=self.max_results)
        context, citations = self._format_context_and_citations(hits)
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
        return thinking.strip() or None

    def _generate_answer(
        self, *, question: str, context: str, self_check_instruction: str = ""
    ) -> str:
        prompt = (
            "You are AI Loop Engine answering with web search evidence. Use only "
            "the provided web search snippets as evidence. Cite supported claims "
            "inline with bracketed numbers like [1]. Web snippets are partial "
            "evidence; if the snippets are insufficient, say that clearly and do "
            "not invent citations. Use clean Markdown for a web chat UI.\n\n"
            f"Web search context:\n{context or '(no web results)'}\n\n"
            f"Question: {question}\n\n"
            f"{self_check_instruction}\n\n"
            "Answer:"
        )
        return str(self.llm.invoke(prompt)).strip()

    def _format_context_and_citations(
        self, hits: List[WebSearchHit]
    ) -> tuple[str, List[AnswerCitation]]:
        context_parts: List[str] = []
        citations: List[AnswerCitation] = []
        remaining_chars = int(self.profile.get("context_total_chars", 4200))
        per_result_chars = int(
            self.profile.get("context_chars_per_chunk", MAX_WEB_SEARCH_SNIPPET_CHARS)
        )

        for hit in hits:
            if len(citations) >= int(self.profile.get("context_chunks", self.max_results)):
                break
            if remaining_chars <= 0:
                break
            snippet = _clean_text(hit.snippet, max_chars=per_result_chars)
            if not snippet:
                continue
            citation_id = len(citations) + 1
            source_name = self._source_name(hit)
            source_label = f"[{citation_id}] Source: {source_name}"
            allowed_chars = min(per_result_chars, max(0, remaining_chars - len(source_label) - 1))
            if allowed_chars <= 0:
                break
            excerpt = snippet[:allowed_chars]
            context_entry = f"{source_label}\n{excerpt}"
            context_parts.append(context_entry)
            remaining_chars -= len(context_entry) + 2
            citations.append(
                AnswerCitation(
                    citation_id=citation_id,
                    source_name=source_name,
                    page=None,
                    chunk_index=None,
                    excerpt=" ".join(excerpt.split()),
                )
            )

        return "\n\n".join(context_parts), citations

    def _source_name(self, hit: WebSearchHit) -> str:
        title = _clean_text(hit.title, max_chars=90) or "Web result"
        url = _safe_result_url(hit.url)
        return f"{title} — {url}" if url else title
