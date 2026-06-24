import json
import threading
import urllib.error
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

import src.DocumentQA as document_qa_module
from src.DocumentQA import (
    DEFAULT_OLLAMA_MODEL,
    AnswerCitation,
    DocumentContextProvider,
    DocumentQA,
    FaissVectorStore,
    MockLLM,
    OllamaLLM,
    OpenAICompatibleLLM,
    normalize_openai_compatible_base_url,
    safe_openai_compatible_base_url_for_error,
)
from src.loop_engine import GuardrailDecision, LoopDecision, LoopPhase, LoopReport


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        lower = text.lower()
        return [
            float(len(text)),
            float(lower.count("project")),
            float(lower.count("phoenix")),
            float(lower.count("launch")),
        ]


def create_processed_mock_qa(tmp_path):
    document = tmp_path / "phoenix.txt"
    document.write_text(
        "Project Phoenix is a document QA assistant. "
        "The launch date is June 2026.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")
    qa.embeddings = FakeEmbeddings()
    qa.process_document(str(document))
    return qa, document


def test_query_before_document_asks_for_upload_first():
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    assert qa.query("What is this?") == "Please upload and process a document first."
    result = qa.query_with_trace("What is this?")
    assert result.answer == "Please upload and process a document first."
    assert result.trace.question == "What is this?"
    assert result.trace.document_name is None
    assert result.trace.retrieved_chunk_count == 0
    assert result.trace.citations == []
    assert result.trace.error_message == "document_not_loaded"


def test_process_text_document_with_mock_llm_and_fake_embeddings(tmp_path):
    document = tmp_path / "phoenix.txt"
    document.write_text(
        "Project Phoenix is a document QA assistant. "
        "The launch date is June 2026.",
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")
    qa.embeddings = FakeEmbeddings()

    qa.process_document(str(document))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is not None
    assert qa.retrieval_chain is not None
    assert qa.active_llm_backend == "mock"
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."
    mock_answer = qa.query("What is Project Phoenix?")
    assert "Project Phoenix" in mock_answer
    assert "[1]" in mock_answer

    status = qa.status()
    assert status.profile_label == "FAST"
    assert status.active_backend == "mock"
    assert status.active_model_label == "MockLLM (explicit demo)"
    assert status.embeddings_device == "cpu"
    assert status.document_name == "phoenix.txt"
    assert status.ready_for_queries is True
    assert status.mock_mode is True
    assert status.processing_report is not None
    assert status.processing_report.attempted_document_name == "phoenix.txt"
    assert status.processing_report.active_document_name == "phoenix.txt"
    assert status.processing_report.success is True
    assert status.processing_report.phase == "complete"
    assert status.processing_report.file_extension == ".txt"
    assert status.processing_report.chunk_count > 0
    assert status.processing_report.truncated is False
    assert status.processing_report.max_chunk_limit == qa.max_document_chunks
    assert status.processing_report.text_encoding_mode == "auto"
    assert status.processing_report.backend == "mock"
    assert status.processing_report.model_label == "MockLLM (explicit demo)"
    assert status.processing_report.error_message is None


def test_query_with_trace_returns_retrieved_citations(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)

    result = qa.query_with_trace("What is Project Phoenix?")

    assert "Project Phoenix" in result.answer
    assert "[1]" in result.answer
    assert result.trace.question == "What is Project Phoenix?"
    assert result.trace.document_name == "phoenix.txt"
    assert result.trace.backend == "mock"
    assert result.trace.model_label == "MockLLM (explicit demo)"
    assert result.trace.retrieved_chunk_count > 0
    assert result.trace.error_message is None
    assert result.trace.self_check.outcome == "not_verified"
    assert result.trace.self_check.reasons == [
        "mechanical_checks_passed",
        "verifier_unavailable_mock_backend",
    ]
    assert result.trace.self_check.retry_attempted is False
    assert len(result.trace.citations) >= 1
    citation = result.trace.citations[0]
    assert citation.citation_id == 1
    assert citation.source_name == "phoenix.txt"
    assert citation.page is None
    assert citation.chunk_index == 0
    assert "Project Phoenix" in citation.excerpt
    assert qa.chat_history[-1]["citations"][0]["source_name"] == "phoenix.txt"


def test_query_trace_counts_only_prompt_included_chunks(tmp_path):
    document = tmp_path / "long-phoenix.txt"
    document.write_text(
        "\n\n".join(
            f"Project Phoenix evidence section {index}. The launch date is June 2026."
            for index in range(80)
        ),
        encoding="utf-8",
    )
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")
    qa.embeddings = FakeEmbeddings()
    qa.process_document(str(document))
    assert len(qa.vector_store.documents) > 1
    qa.profile["context_chunks"] = 1

    result = qa.query_with_trace("What is the launch date?")

    assert result.trace.retrieved_chunk_count == 1
    assert len(result.trace.citations) == 1


def test_query_with_trace_includes_loop_report_for_prompt_evidence(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)

    result = qa.query_with_trace("What is Project Phoenix?", session_id="session_a")

    report = result.loop_report
    assert report is not None
    run = report.run
    assert run.session_id == "session_a"
    assert run.user_input == "What is Project Phoenix?"
    assert run.context_provider == "document"
    assert run.metadata["context_provider"] == "document"
    assert run.metadata["context_provider_name"] == "phoenix.txt"
    assert run.backend == "mock"
    assert run.model_label == "MockLLM (explicit demo)"
    assert run.policy.allow_tool_calls is False
    assert "document_text" in run.metadata["untrusted_inputs"]
    assert run.final_decision == LoopDecision.NOT_VERIFIED
    assert run.final_answer == result.answer

    phases = [step.phase for step in run.steps]
    assert phases == [
        LoopPhase.CONTEXT_SELECT,
        LoopPhase.RETRIEVE,
        LoopPhase.DRAFT,
        LoopPhase.MECHANICAL_CHECK,
        LoopPhase.VERIFY,
        LoopPhase.FINAL,
    ]
    retrieve_step = next(step for step in run.steps if step.phase == LoopPhase.RETRIEVE)
    verify_step = next(step for step in run.steps if step.phase == LoopPhase.VERIFY)
    assert retrieve_step.metadata["retrieved_chunk_count"] == len(result.trace.citations)
    assert retrieve_step.metadata["citation_ids"] == [1]
    assert verify_step.decision == LoopDecision.NOT_VERIFIED
    assert verify_step.verification.outcome.value == "not_verified"
    assert verify_step.verification.reasons == tuple(result.trace.self_check.reasons)


def test_query_records_loop_session_and_exports_jsonl(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)

    first_result = qa.query_with_trace("What is Project Phoenix?", session_id="alpha")
    second_result = qa.query_with_trace("What is the launch date?", session_id="alpha")
    qa.query_with_trace("What is the launch date?", session_id="beta")

    alpha_session = qa.loop_session("alpha")
    beta_session = qa.loop_session("beta")
    artifact_path = tmp_path / "alpha-session.jsonl"
    exported_path = qa.export_loop_session_jsonl(artifact_path, session_id="alpha")
    exported_lines = artifact_path.read_text(encoding="utf-8").splitlines()
    restored_reports = [
        LoopReport.from_dict(json.loads(line)) for line in exported_lines
    ]

    assert exported_path == str(artifact_path)
    assert alpha_session.report_count == 2
    assert beta_session.report_count == 1
    assert [report.run.run_id for report in alpha_session.reports] == [
        first_result.loop_report.run.run_id,
        second_result.loop_report.run.run_id,
    ]
    assert [report.run.session_id for report in restored_reports] == [
        "alpha",
        "alpha",
    ]
    assert restored_reports[0].run.user_input == "What is Project Phoenix?"
    assert restored_reports[1].run.final_decision == LoopDecision.NOT_VERIFIED


def test_loop_session_history_is_bounded(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    qa.max_session_reports = 1

    first_result = qa.query_with_trace("What is Project Phoenix?", session_id="alpha")
    second_result = qa.query_with_trace("What is the launch date?", session_id="alpha")

    alpha_session = qa.loop_session("alpha")

    assert alpha_session.report_count == 1
    assert alpha_session.reports[0].run.run_id == second_result.loop_report.run.run_id
    assert alpha_session.reports[0].run.run_id != first_result.loop_report.run.run_id


def test_blocked_query_is_recorded_for_replay(tmp_path):
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    result = qa.query_with_trace("What is this?", session_id="blocked")

    session = qa.loop_session("blocked")

    assert result.trace.error_message == "document_not_loaded"
    assert session.report_count == 1
    assert session.reports[0].run.final_decision == LoopDecision.BLOCK
    assert session.reports[0].run.error_message == "document_not_loaded"


def test_processed_document_is_wrapped_as_context_provider(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)

    active_state = qa._snapshot_active_document_state()

    assert isinstance(active_state.context_provider, DocumentContextProvider)
    assert active_state.context_provider.provider_type == "document"
    assert active_state.context_provider.display_name == document.name
    assert active_state.context_provider.vector_store is qa.vector_store
    assert active_state.context_provider.retrieval_chain is qa.retrieval_chain
    assert active_state.context_provider.ready is True


def test_loop_middleware_can_block_before_retrieval(tmp_path):
    class BlockRetrieveMiddleware:
        def before_step(self, run, step):
            if step.phase == LoopPhase.RETRIEVE:
                return GuardrailDecision(
                    decision=LoopDecision.BLOCK,
                    reason="retrieval_blocked",
                    metadata={"policy": "test"},
                )
            return None

    document = tmp_path / "phoenix.txt"
    document.write_text(
        "Project Phoenix launches in June 2026.",
        encoding="utf-8",
    )
    qa = DocumentQA(
        fast_mode=True,
        hf_token="dummy",
        llm_backend="mock",
        loop_middlewares=(BlockRetrieveMiddleware(),),
    )
    qa.embeddings = FakeEmbeddings()
    qa.process_document(str(document))

    class CountingRetrievalChain:
        def __init__(self):
            self.calls = 0

        def invoke_with_trace(self, question, self_check_instruction=""):
            self.calls += 1
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026 [1].",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

    retrieval_chain = CountingRetrievalChain()
    replace_retrieval_chain(qa, retrieval_chain)

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert retrieval_chain.calls == 0
    assert result.answer == "A loop guardrail blocked this query before it could complete."
    assert result.trace.error_message == "retrieval_blocked"
    assert result.loop_report.run.final_decision == LoopDecision.BLOCK
    phases = [step.phase for step in result.loop_report.run.steps]
    assert phases == [
        LoopPhase.CONTEXT_SELECT,
        LoopPhase.ERROR,
        LoopPhase.FINAL,
    ]
    guardrail_step = next(
        step for step in result.loop_report.run.steps if step.phase == LoopPhase.ERROR
    )
    assert guardrail_step.metadata["guardrail_decision"] == "block"
    assert guardrail_step.metadata["policy"] == "test"
    assert LoopPhase.RETRIEVE not in phases


def test_loop_middleware_can_block_after_retrieval_before_draft(tmp_path):
    class BlockAfterRetrieveMiddleware:
        def after_step(self, run, step):
            if step.phase == LoopPhase.RETRIEVE:
                return GuardrailDecision(
                    decision=LoopDecision.BLOCK,
                    reason="post_retrieval_blocked",
                )
            return None

    qa, _document = create_processed_mock_qa(tmp_path)
    qa.loop_middlewares = (BlockAfterRetrieveMiddleware(),)
    active_state = qa._snapshot_active_document_state()

    class CountingSplitRetrievalChain:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.draft_calls = 0

        def invoke_with_trace(self, question, self_check_instruction=""):
            return self.wrapped.invoke_with_trace(question, self_check_instruction)

        def retrieve_with_trace(self, question):
            return self.wrapped.retrieve_with_trace(question)

        def draft_with_trace(
            self, question, retrieved_context, self_check_instruction=""
        ):
            self.draft_calls += 1
            return self.wrapped.draft_with_trace(
                question,
                retrieved_context,
                self_check_instruction=self_check_instruction,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return self.wrapped.retry_with_trace(
                question,
                previous_result,
                self_check_instruction,
            )

    retrieval_chain = CountingSplitRetrievalChain(active_state.retrieval_chain)
    replace_retrieval_chain(qa, retrieval_chain)

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert retrieval_chain.draft_calls == 0
    assert result.answer == "A loop guardrail blocked this query before it could complete."
    assert result.trace.error_message == "post_retrieval_blocked"
    assert result.loop_report.run.final_decision == LoopDecision.BLOCK
    phases = [step.phase for step in result.loop_report.run.steps]
    assert phases == [
        LoopPhase.CONTEXT_SELECT,
        LoopPhase.RETRIEVE,
        LoopPhase.ERROR,
        LoopPhase.FINAL,
    ]
    assert LoopPhase.DRAFT not in phases


def test_loop_middleware_refusal_uses_guardrail_specific_answer(tmp_path):
    class RefuseBeforeRunMiddleware:
        def before_run(self, run):
            return GuardrailDecision(
                decision=LoopDecision.REFUSE,
                reason="policy_refused",
            )

    qa, _document = create_processed_mock_qa(tmp_path)
    qa.loop_middlewares = (RefuseBeforeRunMiddleware(),)

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        "A loop guardrail refused this query before it could safely complete."
    )
    assert result.answer != (
        "I could not find enough relevant information in the document to answer that."
    )
    assert result.trace.error_message == "policy_refused"
    assert result.loop_report.run.final_decision == LoopDecision.REFUSE


def test_loop_middleware_retry_request_is_reported_as_unavailable_block(tmp_path):
    class RetryBeforeRunMiddleware:
        def before_run(self, run):
            return GuardrailDecision(
                decision=LoopDecision.RETRY,
                reason="policy_requested_retry",
            )

    qa, _document = create_processed_mock_qa(tmp_path)
    qa.loop_middlewares = (RetryBeforeRunMiddleware(),)

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        "A loop guardrail requested a retry, but no safe retry path is available for this step."
    )
    assert result.trace.error_message == "guardrail_retry_unavailable"
    assert result.loop_report.run.final_decision == LoopDecision.BLOCK
    guardrail_step = next(
        step for step in result.loop_report.run.steps if step.name == "Guardrail decision"
    )
    assert guardrail_step.decision == LoopDecision.RETRY
    assert guardrail_step.metadata["guardrail_reason"] == "policy_requested_retry"


def test_after_run_guardrail_overrides_answer_without_stale_self_check(tmp_path):
    class RequireReviewAfterRunMiddleware:
        def after_run(self, run):
            if run.final_decision == LoopDecision.NOT_VERIFIED:
                return GuardrailDecision(
                    decision=LoopDecision.REQUIRES_REVIEW,
                    reason="review_mock_answer",
                )
            return None

    document = tmp_path / "phoenix.txt"
    document.write_text(
        "Project Phoenix launches in June 2026.",
        encoding="utf-8",
    )
    qa = DocumentQA(
        fast_mode=True,
        hf_token="dummy",
        llm_backend="mock",
        loop_middlewares=(RequireReviewAfterRunMiddleware(),),
    )
    qa.embeddings = FakeEmbeddings()
    qa.process_document(str(document))

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == "This query requires human review before the loop can continue."
    assert result.trace.citations == []
    assert result.trace.self_check is None
    assert result.trace.error_message == "review_mock_answer"
    assert result.loop_report.run.final_decision == LoopDecision.REQUIRES_REVIEW
    assert result.loop_report.run.metadata["after_run_guardrail"] is True
    assert qa.chat_history[-1]["answer"] == result.answer
    assert qa.chat_history[-1]["self_check"] is None


def replace_retrieval_chain(qa, retrieval_chain):
    active_state = qa._snapshot_active_document_state()
    qa._commit_active_document_state(
        document_name=active_state.document_name,
        vector_store=active_state.vector_store,
        retrieval_chain=retrieval_chain,
        processing_report=qa.status().processing_report,
    )


def citation_for(document_name="phoenix.txt"):
    return AnswerCitation(
        citation_id=1,
        source_name=document_name,
        page=None,
        chunk_index=0,
        excerpt="Project Phoenix launches in June 2026.",
    )


class FakeVerifierLLM:
    def __init__(self, outcome="supported", raw_response=None, error=None):
        self.outcome = outcome
        self.raw_response = raw_response
        self.error = error
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        if self.raw_response is not None:
            return self.raw_response
        return json.dumps({"outcome": self.outcome, "reason": "test verifier"})


def enable_fake_verifier(qa, outcome="supported", raw_response=None, error=None):
    verifier = FakeVerifierLLM(
        outcome=outcome,
        raw_response=raw_response,
        error=error,
    )
    qa.active_llm_backend = "endpoint"
    qa.llm = verifier
    qa.loaded_model_label = "Fake verifier endpoint"
    return verifier


def test_self_check_refuses_when_answer_has_no_prompt_evidence(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)

    class NoEvidenceChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow.",
                retrieved_chunk_count=0,
                citations=[],
            )

    replace_retrieval_chain(qa, NoEvidenceChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        "I could not find enough relevant information in the document to answer that."
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert result.trace.self_check.reasons == ["no_prompt_evidence"]
    assert result.trace.self_check.retry_attempted is False
    assert result.trace.citations == []


def test_self_check_retries_missing_inline_citation(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)

    class RetryCitationChain:
        def __init__(self):
            self.calls = []

        def invoke_with_trace(self, question, self_check_instruction=""):
            self.calls.append(self_check_instruction)
            answer = "Project Phoenix launches in June 2026."
            return SimpleNamespace(
                answer=answer,
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="original context",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            self.calls.append(self_check_instruction)
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026 [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    retrieval_chain = RetryCitationChain()
    replace_retrieval_chain(qa, retrieval_chain)

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert len(retrieval_chain.calls) == 2
    assert retrieval_chain.calls[0] == ""
    assert "missing_inline_citation" in retrieval_chain.calls[1]
    assert result.answer == "Project Phoenix launches in June 2026 [1]."
    assert result.trace.self_check.outcome == "not_verified"
    assert result.trace.self_check.reasons == [
        "mechanical_checks_passed",
        "verifier_unavailable_mock_backend",
    ]
    assert result.trace.self_check.retry_attempted is True
    assert qa.chat_history[-1]["self_check"]["retry_attempted"] is True
    assert result.loop_report.run.final_decision == LoopDecision.NOT_VERIFIED
    retry_steps = [
        step for step in result.loop_report.run.steps if step.phase == LoopPhase.RETRY
    ]
    assert len(retry_steps) == 1
    assert retry_steps[0].metadata["reasons"] == ["missing_inline_citation"]
    retry_draft_steps = [
        step
        for step in result.loop_report.run.steps
        if step.phase == LoopPhase.DRAFT and step.retry_count == 1
    ]
    assert len(retry_draft_steps) == 1


def test_self_check_refuses_when_retry_still_fails(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)

    class FailedRetryChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026.",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="original context",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow.",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, FailedRetryChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        "I could not find enough relevant information in the document to answer that."
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "self_check_failed_closed" in result.trace.self_check.reasons
    assert "missing_inline_citation" in result.trace.self_check.reasons
    assert result.trace.self_check.retry_attempted is True


def test_self_check_rejects_cited_but_unsupported_answer(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)
    verifier = enable_fake_verifier(qa, outcome="unsupported")

    class ContradictedCitationChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, ContradictedCitationChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        "I could not find enough relevant information in the document to answer that."
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert result.trace.self_check.reasons == ["llm_verifier_unsupported"]
    assert result.trace.self_check.retry_attempted is False
    assert len(verifier.calls) == 1
    assert "Project Phoenix launches tomorrow [1]." in verifier.calls[0]
    assert "Project Phoenix launches in June 2026." in verifier.calls[0]
    assert result.loop_report.run.final_decision == LoopDecision.REFUSE
    phases = [step.phase for step in result.loop_report.run.steps]
    assert LoopPhase.VERIFY in phases
    assert LoopPhase.REFUSE in phases
    verify_step = next(
        step for step in result.loop_report.run.steps if step.phase == LoopPhase.VERIFY
    )
    assert verify_step.decision == LoopDecision.REFUSE
    assert verify_step.verification.outcome.value == "unsupported"


def test_llm_verifier_marks_real_backend_answer_supported(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)
    verifier = enable_fake_verifier(qa, outcome="supported")

    class SupportedCitationChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026 [1].",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

    replace_retrieval_chain(qa, SupportedCitationChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == "Project Phoenix launches in June 2026 [1]."
    assert result.trace.backend == "endpoint"
    assert result.trace.self_check.outcome == "supported"
    assert result.trace.self_check.reasons == [
        "mechanical_checks_passed",
        "llm_verifier_supported",
    ]
    assert len(verifier.calls) == 1
    assert "When does Project Phoenix launch?" in verifier.calls[0]
    assert "Project Phoenix launches in June 2026." in verifier.calls[0]


def test_loop_middleware_can_block_before_verifier_call(tmp_path):
    class BlockVerifyMiddleware:
        def before_step(self, run, step):
            if step.phase == LoopPhase.VERIFY:
                return GuardrailDecision(
                    decision=LoopDecision.BLOCK,
                    reason="verifier_blocked",
                )
            return None

    qa, document = create_processed_mock_qa(tmp_path)
    qa.loop_middlewares = (BlockVerifyMiddleware(),)
    verifier = enable_fake_verifier(qa, outcome="supported")

    class SupportedCitationChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026 [1].",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

    replace_retrieval_chain(qa, SupportedCitationChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert verifier.calls == []
    assert result.answer == "A loop guardrail blocked this query before it could complete."
    assert result.trace.error_message == "verifier_blocked"
    assert result.loop_report.run.final_decision == LoopDecision.BLOCK
    phases = [step.phase for step in result.loop_report.run.steps]
    assert LoopPhase.MECHANICAL_CHECK in phases
    assert LoopPhase.VERIFY not in phases


def test_self_check_retries_hallucinated_inline_citation_id(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)
    verifier = enable_fake_verifier(qa, outcome="supported")

    class HallucinatedCitationChain:
        def __init__(self):
            self.calls = []

        def invoke_with_trace(self, question, self_check_instruction=""):
            self.calls.append(self_check_instruction)
            return SimpleNamespace(
                answer=(
                    "Project Phoenix launches in June 2026 [1]. "
                    "Budget is $10 [999]."
                ),
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            self.calls.append(self_check_instruction)
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026 [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    retrieval_chain = HallucinatedCitationChain()
    replace_retrieval_chain(qa, retrieval_chain)

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == "Project Phoenix launches in June 2026 [1]."
    assert result.trace.self_check.outcome == "supported"
    assert result.trace.self_check.retry_attempted is True
    assert len(retrieval_chain.calls) == 2
    assert "invalid_inline_citation" in retrieval_chain.calls[1]
    assert len(verifier.calls) == 1
    assert "[999]" not in verifier.calls[0]


def test_self_check_refuses_when_retry_keeps_hallucinated_inline_citation_id(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)
    verifier = enable_fake_verifier(qa, outcome="supported")

    class StillHallucinatedCitationChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer=(
                    "Project Phoenix launches in June 2026 [1]. "
                    "Budget is $10 [999]."
                ),
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer=(
                    "Project Phoenix launches in June 2026 [1]. "
                    "Budget is $10 [999]."
                ),
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, StillHallucinatedCitationChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        "I could not find enough relevant information in the document to answer that."
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "self_check_failed_closed" in result.trace.self_check.reasons
    assert "invalid_inline_citation" in result.trace.self_check.reasons
    assert result.trace.self_check.retry_attempted is True
    assert verifier.calls == []


@pytest.mark.parametrize(
    "verifier_kwargs, expected_reasons",
    [
        ({"outcome": "insufficient"}, ["llm_verifier_insufficient"]),
        ({"raw_response": "not-json"}, ["llm_verifier_parse_failed", "missing_json"]),
        ({"error": RuntimeError("verifier down")}, ["llm_verifier_error"]),
    ],
)
def test_llm_verifier_failures_refuse_answer(
    tmp_path, verifier_kwargs, expected_reasons
):
    qa, document = create_processed_mock_qa(tmp_path)
    enable_fake_verifier(qa, **verifier_kwargs)

    class SupportedCitationChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026 [1].",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

    replace_retrieval_chain(qa, SupportedCitationChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        "I could not find enough relevant information in the document to answer that."
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert result.trace.self_check.reasons == expected_reasons


def test_self_check_rejects_inverted_relationship_claim(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    enable_fake_verifier(qa, outcome="unsupported")
    citation = AnswerCitation(
        citation_id=1,
        source_name="acquisition.txt",
        page=None,
        chunk_index=0,
        excerpt="Alice acquired Bob in 2026.",
    )

    class InvertedRelationshipChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Bob acquired Alice in 2026 [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Bob acquired Alice in 2026 [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, InvertedRelationshipChain())

    result = qa.query_with_trace("Who acquired whom?")

    assert result.answer == (
        "I could not find enough relevant information in the document to answer that."
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert result.trace.self_check.reasons == ["llm_verifier_unsupported"]


def test_self_check_rejects_denied_extractive_claim(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="denial.txt",
        page=None,
        chunk_index=0,
        excerpt="Project Phoenix launches tomorrow is false.",
    )

    class DeniedClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, DeniedClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        "I could not find enough relevant information in the document to answer that."
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "citation_text_does_not_support_answer" in result.trace.self_check.reasons


def test_self_check_rejects_later_refuted_repeated_claim(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="later-denial.txt",
        page=None,
        chunk_index=0,
        excerpt=(
            "Project Phoenix launches tomorrow. "
            "Project Phoenix launches tomorrow is false."
        ),
    )

    class LaterDeniedClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, LaterDeniedClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        "I could not find enough relevant information in the document to answer that."
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "citation_text_does_not_support_answer" in result.trace.self_check.reasons


@pytest.mark.parametrize(
    "excerpt",
    [
        "Project Phoenix launches tomorrow, but that is false.",
        "Project Phoenix launches tomorrow; however, that is false.",
        "Project Phoenix launches tomorrow, although that is not true.",
        "Project Phoenix launches tomorrow, which is incorrect.",
    ],
)
def test_self_check_rejects_connector_refuted_extractive_claim(tmp_path, excerpt):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="connector-denial.txt",
        page=None,
        chunk_index=0,
        excerpt=excerpt,
    )

    class ConnectorDeniedClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, ConnectorDeniedClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        "I could not find enough relevant information in the document to answer that."
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "citation_text_does_not_support_answer" in result.trace.self_check.reasons


@pytest.mark.parametrize(
    "excerpt",
    [
        "Project Phoenix launches tomorrow, but that claim is false.",
        "Project Phoenix launches tomorrow, but that claim has been denied.",
        "Project Phoenix launches tomorrow, but that claim has been rejected.",
        "Project Phoenix launches tomorrow, but that claim has been debunked.",
        "Project Phoenix launches tomorrow, but that claim is contradicted by the schedule.",
        "Project Phoenix launches tomorrow, but that claim is untrue.",
        "Project Phoenix launches tomorrow, but that claim is unsupported.",
        "Project Phoenix launches tomorrow, but that claim is not supported by the schedule.",
        "Project Phoenix launches tomorrow, but that claim is disputed.",
        "Project Phoenix launches tomorrow, but that claim is inaccurate.",
        "Project Phoenix launches tomorrow, but that claim is baseless.",
        "Project Phoenix launches tomorrow, but that claim is unfounded.",
        "Project Phoenix launches tomorrow, but that claim has been disproven.",
        "Project Phoenix launches tomorrow, but that claim has been disproved.",
        "Project Phoenix launches tomorrow, but the claim is false.",
        "Project Phoenix launches tomorrow, but the prior claim is false.",
        "Project Phoenix launches tomorrow, but the statement is false.",
        "Project Phoenix launches tomorrow, but this assertion has been refuted.",
        "Project Phoenix launches tomorrow, but this claim is not true.",
    ],
)
def test_self_check_rejects_connector_noun_phrase_refutation(tmp_path, excerpt):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="connector-noun-denial.txt",
        page=None,
        chunk_index=0,
        excerpt=excerpt,
    )

    class ConnectorNounDeniedClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, ConnectorNounDeniedClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        "I could not find enough relevant information in the document to answer that."
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "citation_text_does_not_support_answer" in result.trace.self_check.reasons


@pytest.mark.parametrize(
    "excerpt",
    [
        "Project Phoenix launches tomorrow, which is correct.",
        "Project Phoenix launches tomorrow, but that claim is correct.",
        "Project Phoenix launches tomorrow, but that claim has not been rejected.",
        "Project Phoenix launches tomorrow, but that claim has not been disproven.",
    ],
)
def test_self_check_does_not_treat_positive_connector_as_denial(tmp_path, excerpt):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="positive-connector.txt",
        page=None,
        chunk_index=0,
        excerpt=excerpt,
    )

    class PositiveConnectorClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, PositiveConnectorClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == "Project Phoenix launches tomorrow [1]."
    assert result.trace.self_check.outcome == "not_verified"
    assert result.trace.self_check.reasons == [
        "mechanical_checks_passed",
        "verifier_unavailable_mock_backend",
    ]


def test_self_check_rejects_prefix_refuted_extractive_claim(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="prefix-denial.txt",
        page=None,
        chunk_index=0,
        excerpt="It is false that Project Phoenix launches tomorrow.",
    )

    class PrefixDeniedClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, PrefixDeniedClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        "I could not find enough relevant information in the document to answer that."
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "citation_text_does_not_support_answer" in result.trace.self_check.reasons


@pytest.mark.parametrize(
    "excerpt",
    [
        "Project Phoenix launches tomorrow? No.",
        "Project Phoenix launches tomorrow? No, it launches in June 2026.",
    ],
)
def test_self_check_rejects_qa_style_denied_extractive_claim(tmp_path, excerpt):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="qa-denial.txt",
        page=None,
        chunk_index=0,
        excerpt=excerpt,
    )

    class QaDeniedClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, QaDeniedClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == (
        "I could not find enough relevant information in the document to answer that."
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert "citation_text_does_not_support_answer" in result.trace.self_check.reasons


def test_self_check_does_not_treat_no_later_qualifier_as_denial(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    citation = AnswerCitation(
        citation_id=1,
        source_name="schedule.txt",
        page=None,
        chunk_index=0,
        excerpt="Project Phoenix launches tomorrow no later than noon.",
    )

    class QualifiedClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation],
                context=citation.excerpt,
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, QualifiedClaimChain())

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert result.answer == "Project Phoenix launches tomorrow [1]."
    assert result.trace.self_check.outcome == "not_verified"
    assert result.trace.self_check.reasons == [
        "mechanical_checks_passed",
        "verifier_unavailable_mock_backend",
    ]


def test_self_check_rejects_false_premise_question_token_laundering(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)
    enable_fake_verifier(qa, outcome="unsupported")

    class FalsePremiseChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    replace_retrieval_chain(qa, FalsePremiseChain())

    result = qa.query_with_trace("Does Project Phoenix launch tomorrow?")

    assert result.answer == (
        "I could not find enough relevant information in the document to answer that."
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert result.trace.self_check.reasons == ["llm_verifier_unsupported"]


def test_self_check_rejects_unchecked_unicode_claim(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)
    enable_fake_verifier(qa, outcome="unsupported")

    class UnicodeClaimChain:
        def invoke_with_trace(self, question, self_check_instruction=""):
            return SimpleNamespace(
                answer="Привет [1].",
                retrieved_chunk_count=1,
                citations=[citation_for(document.name)],
                context="Project Phoenix launches in June 2026.",
            )

    replace_retrieval_chain(qa, UnicodeClaimChain())

    result = qa.query_with_trace("What greeting is in the document?")

    assert result.answer == (
        "I could not find enough relevant information in the document to answer that."
    )
    assert result.trace.self_check.outcome == "needs_refusal"
    assert result.trace.self_check.reasons == ["llm_verifier_unsupported"]


def test_self_check_retry_reuses_original_prompt_evidence(tmp_path):
    qa, document = create_processed_mock_qa(tmp_path)

    class DriftingEvidenceChain:
        def __init__(self):
            self.invoke_calls = 0
            self.retry_context = None

        def invoke_with_trace(self, question, self_check_instruction=""):
            self.invoke_calls += 1
            if self.invoke_calls == 1:
                return SimpleNamespace(
                    answer="Project Phoenix launches in June 2026.",
                    retrieved_chunk_count=1,
                    citations=[citation_for(document.name)],
                    context="original prompt evidence",
                )
            return SimpleNamespace(
                answer="Project Phoenix launches tomorrow [1].",
                retrieved_chunk_count=1,
                citations=[citation_for("drift.txt")],
                context="drifted prompt evidence",
            )

        def retry_with_trace(self, question, previous_result, self_check_instruction):
            self.retry_context = previous_result.context
            return SimpleNamespace(
                answer="Project Phoenix launches in June 2026 [1].",
                retrieved_chunk_count=previous_result.retrieved_chunk_count,
                citations=previous_result.citations,
                context=previous_result.context,
            )

    retrieval_chain = DriftingEvidenceChain()
    replace_retrieval_chain(qa, retrieval_chain)

    result = qa.query_with_trace("When does Project Phoenix launch?")

    assert retrieval_chain.invoke_calls == 1
    assert retrieval_chain.retry_context == "original prompt evidence"
    assert result.trace.citations[0].source_name == document.name
    assert result.trace.self_check.outcome == "not_verified"
    assert result.trace.self_check.reasons == [
        "mechanical_checks_passed",
        "verifier_unavailable_mock_backend",
    ]
    assert result.trace.self_check.retry_attempted is True


def test_document_chunks_record_citation_metadata(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)

    assert qa.vector_store.documents[0].metadata["chunk_index"] == 0


def test_status_reports_configured_backend_before_initialization():
    qa = DocumentQA(
        device="cpu",
        fast_mode=False,
        hf_token="real-token",
        llm_backend="endpoint",
        model_id="example/model",
    )

    status = qa.status()

    assert status.profile_label == "QUALITY"
    assert status.configured_backend == "endpoint"
    assert status.active_backend == "endpoint"
    assert status.active_model_label == "example/model"
    assert status.embeddings_device == "cpu"
    assert status.ready_for_queries is False
    assert status.mock_mode is False
    assert status.processing_report is None


def test_auto_status_before_initialization_reports_local_first_plan():
    qa = DocumentQA(fast_mode=True, llm_backend="auto", hf_token=None)

    status = qa.status()

    assert status.configured_backend == "auto"
    assert status.active_backend == "auto"
    assert status.active_model_label == "Auto (Ollama nemotron-3-nano:4b)"
    assert "Qwen" not in status.active_model_label


def test_embeddings_default_to_cpu_on_mps_device():
    qa = DocumentQA(
        device="mps",
        fast_mode=True,
        hf_token="dummy",
        llm_backend="mock",
    )

    assert qa.device == "mps"
    assert qa.embeddings_device == "cpu"
    assert qa.status().embeddings_device == "cpu"


def test_embeddings_device_can_be_explicitly_cpu_on_accelerated_device():
    qa = DocumentQA(
        device="mps",
        embeddings_device="cpu",
        fast_mode=True,
        hf_token="dummy",
        llm_backend="mock",
    )

    assert qa.embeddings_device == "cpu"


def test_initialize_embeddings_uses_safe_embeddings_device(monkeypatch):
    captured = {}

    class FakeHuggingFaceEmbeddings:
        def __init__(self, *, model_name, model_kwargs):
            captured["model_name"] = model_name
            captured["model_kwargs"] = model_kwargs

    monkeypatch.setattr(
        document_qa_module,
        "HuggingFaceEmbeddings",
        FakeHuggingFaceEmbeddings,
    )
    qa = DocumentQA(
        device="mps",
        fast_mode=True,
        hf_token="dummy",
        llm_backend="mock",
    )

    qa._initialize_embeddings()

    assert captured["model_name"] == qa.embeddings_model
    assert captured["model_kwargs"] == {"device": "cpu"}
    assert qa.embeddings is not None


def test_failed_unsupported_replacement_keeps_previous_document_queryable(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    replacement = tmp_path / "replacement.csv"
    replacement.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unsupported file type: .csv"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "validate"
    assert report.attempted_document_name == "replacement.csv"
    assert report.active_document_name == "phoenix.txt"
    assert report.file_extension == ".csv"
    assert report.chunk_count == 0
    assert "Unsupported file type: .csv" in report.error_message


def test_query_uses_active_state_snapshot_not_legacy_mixed_state(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)

    qa.current_document_name = "replacement.txt"
    qa.vector_store = None
    qa.retrieval_chain = None

    assert qa.status().document_name == "phoenix.txt"
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."


def test_concurrent_replacement_uploads_are_serialized(monkeypatch, tmp_path):
    first = tmp_path / "first.txt"
    first.write_text("First document for Project Phoenix.", encoding="utf-8")
    second = tmp_path / "second.txt"
    second.write_text("Second document for Project Phoenix.", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")
    qa.embeddings = FakeEmbeddings()
    original_process_document = DocumentQA.process_document
    original_load_documents = DocumentQA._load_documents
    second_upload_attempted = threading.Event()
    first_load_started = threading.Event()
    second_load_started = threading.Event()
    release_first_load = threading.Event()
    errors = []

    def tracked_process_document(self, document_path, text_encoding=None):
        if document_path == str(second):
            second_upload_attempted.set()
        return original_process_document(
            self, document_path, text_encoding=text_encoding
        )

    def delayed_first_load(self, document_path, file_extension, text_encoding=None):
        if document_path == str(first):
            first_load_started.set()
            assert release_first_load.wait(timeout=5)
        elif document_path == str(second):
            second_load_started.set()
        return original_load_documents(
            self, document_path, file_extension, text_encoding=text_encoding
        )

    def process(path):
        try:
            qa.process_document(str(path))
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(DocumentQA, "process_document", tracked_process_document)
    monkeypatch.setattr(DocumentQA, "_load_documents", delayed_first_load)

    first_thread = threading.Thread(target=process, args=(first,))
    first_thread.start()
    assert first_load_started.wait(timeout=5)

    second_thread = threading.Thread(target=process, args=(second,))
    second_thread.start()
    assert second_upload_attempted.wait(timeout=5)
    assert not second_load_started.wait(timeout=0.2)
    release_first_load.set()

    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert second_load_started.is_set()
    assert errors == []
    assert qa.status().document_name == "second.txt"
    assert qa.query("Which file did I upload?") == "The uploaded document is `second.txt`."


def test_failed_ambiguous_text_replacement_keeps_previous_document_queryable(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    replacement = tmp_path / "ambiguous.txt"
    replacement.write_bytes("Привет Phoenix".encode("cp1251"))

    with pytest.raises(RuntimeError, match="Could not decode text document"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "load"
    assert report.attempted_document_name == "ambiguous.txt"
    assert report.active_document_name == "phoenix.txt"
    assert report.file_extension == ".txt"
    assert report.text_encoding_mode == "auto"
    assert "Could not decode text document" in report.error_message


def test_failed_embedding_initialization_keeps_previous_document_queryable(
    monkeypatch, tmp_path
):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    qa.embeddings = None
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("Replacement document should not commit.", encoding="utf-8")

    def fail_initialize_embeddings(self):
        self.embeddings_error = "boom"
        self.embeddings = None

    monkeypatch.setattr(DocumentQA, "_initialize_embeddings", fail_initialize_embeddings)

    with pytest.raises(RuntimeError, match="Embedding model is unavailable"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "initialize_embeddings"
    assert report.attempted_document_name == "replacement.txt"
    assert report.active_document_name == "phoenix.txt"
    assert "Embedding model is unavailable" in report.error_message


def test_empty_replacement_keeps_previous_document_queryable(tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    replacement = tmp_path / "empty.txt"
    replacement.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="No text chunks were generated"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "split"
    assert report.attempted_document_name == "empty.txt"
    assert report.active_document_name == "phoenix.txt"
    assert report.chunk_count == 0
    assert "No text chunks were generated" in report.error_message


def test_index_failure_keeps_previous_document_queryable(monkeypatch, tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("Replacement document should not commit.", encoding="utf-8")

    def fail_from_documents(documents, embedding):
        raise ValueError("index boom")

    monkeypatch.setattr(FaissVectorStore, "from_documents", fail_from_documents)

    with pytest.raises(RuntimeError, match="index boom"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "index"
    assert report.attempted_document_name == "replacement.txt"
    assert report.active_document_name == "phoenix.txt"
    assert report.chunk_count > 0
    assert "index boom" in report.error_message


def test_chain_failure_keeps_previous_document_queryable(monkeypatch, tmp_path):
    qa, _document = create_processed_mock_qa(tmp_path)
    previous_vector_store = qa.vector_store
    previous_retrieval_chain = qa.retrieval_chain
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("Replacement document should not commit.", encoding="utf-8")

    def fail_build_retrieval_chain(self, vector_store, document_name):
        raise RuntimeError("chain boom")

    monkeypatch.setattr(
        DocumentQA, "_build_retrieval_chain", fail_build_retrieval_chain
    )

    with pytest.raises(RuntimeError, match="chain boom"):
        qa.process_document(str(replacement))

    assert qa.current_document_name == "phoenix.txt"
    assert qa.vector_store is previous_vector_store
    assert qa.retrieval_chain is previous_retrieval_chain
    assert qa.query("Which file did I upload?") == "The uploaded document is `phoenix.txt`."
    report = qa.status().processing_report
    assert report.success is False
    assert report.phase == "chain"
    assert report.attempted_document_name == "replacement.txt"
    assert report.active_document_name == "phoenix.txt"
    assert report.chunk_count > 0
    assert "chain boom" in report.error_message


def test_successful_upload_report_records_truncation(tmp_path):
    document = tmp_path / "long.txt"
    document.write_text(("Project Phoenix launch notes.\n" * 500), encoding="utf-8")
    qa = DocumentQA(
        fast_mode=True,
        hf_token="dummy",
        llm_backend="mock",
        max_document_chunks=1,
    )
    qa.embeddings = FakeEmbeddings()

    qa.process_document(str(document))

    report = qa.status().processing_report
    assert report.success is True
    assert report.phase == "complete"
    assert report.chunk_count == 1
    assert report.truncated is True
    assert report.max_chunk_limit == 1


def test_text_loader_sets_source_metadata(tmp_path):
    document = tmp_path / "notes.md"
    document.write_text("# Notes\nProject Phoenix launch notes.", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    loaded_documents = qa._load_documents(str(document), ".md")

    assert len(loaded_documents) == 1
    assert loaded_documents[0].page_content.startswith("# Notes")
    assert loaded_documents[0].metadata["source"] == str(document)


@pytest.mark.parametrize(
    ("encoding", "text"),
    [
        ("latin-1", "Café"),
        ("latin-1", "Crème brûlée"),
        ("latin-1", "Voilà"),
        ("latin-1", "Résumé"),
        ("latin-1", "naïve façade"),
        ("latin-1", "piñata"),
        ("latin-1", "paño Phoenix"),
        ("latin-1", "mañana Phoenix"),
        ("latin-1", "Málaga"),
        ("latin-1", "Córdoba"),
        ("latin-1", "García"),
        ("latin-1", "María"),
        ("latin-1", "Hélène"),
        ("latin-1", "Québec"),
        ("latin-1", "Montréal"),
        ("latin-1", "Montréal Québec"),
        ("latin-1", "München"),
        ("latin-1", "Zürich"),
        ("latin-1", "Düsseldorf"),
        ("latin-1", "Göteborg"),
        ("latin-1", "Köln"),
        ("latin-1", "Jürgen"),
        ("latin-1", "Bücher"),
        ("latin-1", "Søren"),
        ("latin-1", "København"),
        ("latin-1", "Å"),
        ("latin-1", "Ångström"),
        ("latin-1", "Tromsø"),
        ("latin-1", "smørrebrød"),
        ("latin-1", "A Coruña"),
        ("latin-1", "açaí"),
        ("latin-1", "Ação"),
        ("latin-1", "coração"),
        ("latin-1", "João"),
        ("latin-1", "mãe"),
        ("latin-1", "smörgåsbord"),
        ("latin-1", "El Niño"),
        ("latin-1", "São Paulo"),
        ("latin-1", "François"),
        ("latin-1", "über"),
        ("latin-1", "garçon"),
        ("latin-1", "Area 10 m²"),
        ("latin-1", "Volume 5 cm³"),
        ("latin-1", "Temp 20 °C"),
        ("latin-1", "Price £10"),
        ("latin-1", "Half ½ cup"),
        ("latin-1", "Café Phoenix résumé"),
        ("cp1252", "Café Phoenix — résumé"),
        ("cp1252", "Price €10 — Phoenix"),
        ("cp1252", "¿Cómo estás? Phoenix"),
        ("cp1252", "Trademark ™ Phoenix"),
    ],
)
def test_text_loader_detects_common_non_utf8_encodings(tmp_path, encoding, text):
    document = tmp_path / "legacy.txt"
    document.write_bytes(text.encode(encoding))
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    loaded_documents = qa._load_documents(
        str(document), ".txt", text_encoding=encoding
    )

    assert loaded_documents[0].page_content == text
    assert "\ufffd" not in loaded_documents[0].page_content


@pytest.mark.parametrize(
    ("encoding", "text"),
    [
        ("utf-8", "Café Phoenix"),
        ("utf-8", "Résumé Phoenix"),
        ("utf-8", "München Phoenix"),
        ("utf-8", "Price €10 — Phoenix"),
        ("utf-8-sig", "Café Phoenix"),
        ("latin-1", "Göteborg Phoenix"),
        ("cp1252", "Price €10 — Phoenix"),
    ],
)
def test_text_loader_utf8_or_western_mode_preserves_text(tmp_path, encoding, text):
    document = tmp_path / "default_text.txt"
    document.write_bytes(text.encode(encoding))
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    loaded_documents = qa._load_documents(
        str(document), ".txt", text_encoding="utf-8-or-western"
    )

    assert loaded_documents[0].page_content == text
    assert "Ã" not in loaded_documents[0].page_content
    assert "â" not in loaded_documents[0].page_content


@pytest.mark.parametrize(
    ("encoding", "text"),
    [
        ("utf-16-le", "Project Phoenix"),
        ("utf-16-be", "Project Phoenix"),
        ("utf-32-le", "Project Phoenix"),
        ("utf-32-be", "Project Phoenix"),
        ("utf-32", "Project Phoenix"),
    ],
)
def test_text_loader_detects_utf_family_encodings(tmp_path, encoding, text):
    document = tmp_path / "unicode.txt"
    document.write_bytes(text.encode(encoding))
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    loaded_documents = qa._load_documents(str(document), ".txt")

    assert loaded_documents[0].page_content == text
    assert "\x00" not in loaded_documents[0].page_content


def test_text_loader_uses_confident_detector_for_legacy_encoding(tmp_path):
    text = "Zażółć gęślą jaźń"
    document = tmp_path / "polish.txt"
    document.write_bytes(text.encode("cp1250"))
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    loaded_documents = qa._load_documents(str(document), ".txt")

    assert loaded_documents[0].page_content == text


@pytest.mark.parametrize(
    ("encoding", "text"),
    [
        ("cp1250", "Dvořák Phoenix"),
        ("cp1250", "město Phoenix"),
        ("cp1251", "Привет Phoenix"),
        ("cp1254", "İstanbul Phoenix"),
        ("cp1257", "māja Phoenix"),
        ("cp1257", "Rīga Phoenix"),
    ],
)
def test_text_loader_uses_explicit_legacy_encoding(tmp_path, encoding, text):
    document = tmp_path / "legacy.txt"
    document.write_bytes(text.encode(encoding))
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    loaded_documents = qa._load_documents(
        str(document), ".txt", text_encoding=encoding
    )

    assert loaded_documents[0].page_content == text


def test_text_loader_reports_invalid_explicit_encoding(tmp_path):
    document = tmp_path / "legacy.txt"
    document.write_bytes("Phoenix".encode("utf-8"))
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    with pytest.raises(ValueError, match="Could not decode text document"):
        qa._load_documents(str(document), ".txt", text_encoding="not-a-codec")


@pytest.mark.parametrize(
    ("encoding", "text", "mojibake"),
    [
        ("cp1251", "Привет Phoenix", "Ïðèâåò Phoenix"),
        ("cp1251", "аб Phoenix", "àá Phoenix"),
        ("cp1251", "я и ты Phoenix", "ÿ è òû Phoenix"),
        ("cp1251", "Привет — Phoenix", "Ïðèâåò — Phoenix"),
        ("cp1251", "Привет – Phoenix", "Ïðèâåò – Phoenix"),
        ("cp1251", "Привет… Phoenix", "Ïðèâåò… Phoenix"),
        ("cp1251", "Ј Phoenix", "£ Phoenix"),
        ("cp1251", "Јован Phoenix", "£îâàí Phoenix"),
        ("cp1251", "Ђ Phoenix", "€ Phoenix"),
        ("cp1251", "№ Phoenix", "¹ Phoenix"),
        ("gb18030", "项目 Phoenix", "ÏîÄ¿ Phoenix"),
        ("cp1250", "Ł10 Phoenix", "£10 Phoenix"),
        ("cp1250", "Łódź Phoenix", "£ódŸ Phoenix"),
        ("cp1254", "İstanbul Phoenix", "Ýstanbul Phoenix"),
        ("cp1257", "māja Phoenix", "mâja Phoenix"),
        ("cp1257", "Rīga Phoenix", "Rîga Phoenix"),
        ("cp1250", "Dvořák Phoenix", "Dvoøák Phoenix"),
        ("cp1250", "město Phoenix", "mìsto Phoenix"),
        ("cp1250", "książka Phoenix", "ksi¹¿ka Phoenix"),
        ("cp1250", "Dąb Phoenix", "D¹b Phoenix"),
        ("cp1250", "zażalenie Phoenix", "za¿alenie Phoenix"),
        ("iso-8859-2", "Łódź Phoenix", "£ód¼ Phoenix"),
        ("big5", "項目 Phoenix", "¶µ¥Ø Phoenix"),
    ],
)
def test_text_loader_rejects_ambiguous_legacy_text_instead_of_mojibake(
    tmp_path, encoding, text, mojibake
):
    document = tmp_path / "ambiguous.txt"
    document.write_bytes(text.encode(encoding))
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    with pytest.raises(ValueError, match="Could not decode text document"):
        qa._load_documents(str(document), ".txt")

    assert document.read_bytes().decode("cp1252") == mojibake


def test_faiss_vector_store_returns_relevant_documents():
    documents = [
        Document(page_content="Project Phoenix launch notes"),
        Document(page_content="Unrelated accounting memo"),
    ]
    vector_store = FaissVectorStore.from_documents(documents, FakeEmbeddings())

    results = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 1, "fetch_k": 2}
    ).invoke("Phoenix project")

    assert results == [documents[0]]


def test_rejects_unsupported_file_type_before_model_initialization(tmp_path):
    document = tmp_path / "data.csv"
    document.write_text("a,b\n1,2\n", encoding="utf-8")
    qa = DocumentQA(fast_mode=True, hf_token="dummy", llm_backend="mock")

    with pytest.raises(RuntimeError, match="Unsupported file type: .csv"):
        qa.process_document(str(document))

    assert qa.llm is None


def test_rejects_oversized_document_before_model_initialization(tmp_path):
    document = tmp_path / "too-large.txt"
    document.write_text("too large", encoding="utf-8")
    qa = DocumentQA(
        fast_mode=True, hf_token="dummy", llm_backend="mock", max_document_bytes=1
    )

    with pytest.raises(RuntimeError, match="File is too large"):
        qa.process_document(str(document))

    assert qa.llm is None


def test_auto_backend_uses_ollama_when_available_without_hf_token(monkeypatch):
    def fake_ollama_loader(self, model_id):
        return MockLLM()

    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "nemotron-3-nano:4b")
    monkeypatch.setattr(DocumentQA, "_load_ollama_model", fake_ollama_loader)
    qa = DocumentQA(device="cpu", hf_token=None, llm_backend="auto")

    qa._initialize_llm()

    assert qa.active_llm_backend == "ollama"
    assert qa.loaded_model_id == "nemotron-3-nano:4b"
    assert qa.loaded_model_label == "Ollama (nemotron-3-nano:4b)"


def test_invalid_llm_backend_env_fails_closed_before_loading_model(monkeypatch):
    load_attempted = False

    def forbidden_ollama_loader(self, model_id):
        nonlocal load_attempted
        load_attempted = True
        raise AssertionError("invalid backend must not select Ollama")

    monkeypatch.setenv("LLM_BACKEND", "mockk")
    monkeypatch.setattr(DocumentQA, "_load_ollama_model", forbidden_ollama_loader)

    with pytest.raises(RuntimeError, match="Unsupported LLM_BACKEND='mockk'"):
        DocumentQA(device="cpu", hf_token=None)

    assert load_attempted is False


def test_endpoint_url_does_not_also_set_repo_id(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT_URL", "https://example.invalid")
    qa = DocumentQA(device="cpu", hf_token="real-token", llm_backend="endpoint")

    llm = qa._load_endpoint_model("Qwen/Qwen2.5-1.5B-Instruct")

    assert llm.endpoint_url == "https://example.invalid"
    assert llm.repo_id is None


def test_custom_endpoint_records_endpoint_label_not_candidate_model(monkeypatch):
    def fake_endpoint_loader(self, model_id):
        return MockLLM()

    monkeypatch.setenv("HF_ENDPOINT_URL", "https://example.invalid")
    monkeypatch.setattr(DocumentQA, "_load_endpoint_model", fake_endpoint_loader)
    qa = DocumentQA(device="cpu", hf_token="real-token", llm_backend="endpoint")

    qa._initialize_llm()

    assert qa.loaded_model_id is None
    assert qa.loaded_model_label == "Custom endpoint (https://example.invalid)"


def test_auto_backend_uses_ollama_on_cuda_when_available(monkeypatch):
    def fake_ollama_loader(self, model_id):
        return MockLLM()

    monkeypatch.setattr(DocumentQA, "_load_ollama_model", fake_ollama_loader)
    qa = DocumentQA(device="cuda", hf_token="real-token", llm_backend="auto")

    qa._initialize_llm()

    assert qa.active_llm_backend == "ollama"
    assert qa.loaded_model_id == "nemotron-3-nano:4b"


def test_auto_backend_uses_ollama_on_mps_when_available(monkeypatch):
    def fake_ollama_loader(self, model_id):
        return MockLLM()

    monkeypatch.setattr(DocumentQA, "_load_ollama_model", fake_ollama_loader)
    qa = DocumentQA(device="mps", hf_token="real-token", llm_backend="auto")

    qa._initialize_llm()

    assert qa.active_llm_backend == "ollama"
    assert qa.loaded_model_id == "nemotron-3-nano:4b"


def test_auto_backend_fails_closed_when_ollama_unavailable(monkeypatch):
    def failing_ollama_loader(self, model_id):
        raise ValueError("ollama unavailable")

    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.setattr(DocumentQA, "_load_ollama_model", failing_ollama_loader)
    qa = DocumentQA(device="mps", llm_backend="auto", hf_token=None)

    with pytest.raises(RuntimeError, match="Unable to initialize auto-selected"):
        qa._initialize_llm()

    assert qa.active_llm_backend is None
    assert qa.llm is None


def test_auto_backend_does_not_use_hf_when_hf_token_exists(monkeypatch):
    def failing_ollama_loader(self, model_id):
        raise ValueError("ollama unavailable")

    def forbidden_endpoint_loader(self, model_id):
        raise AssertionError("auto should not use Hugging Face endpoint")

    def forbidden_local_loader(self, model_id):
        raise AssertionError("auto should not use in-process Hugging Face local")

    monkeypatch.setattr(DocumentQA, "_load_ollama_model", failing_ollama_loader)
    monkeypatch.setattr(DocumentQA, "_load_endpoint_model", forbidden_endpoint_loader)
    monkeypatch.setattr(DocumentQA, "_load_local_model", forbidden_local_loader)
    qa = DocumentQA(device="cpu", llm_backend="auto", hf_token="real-token")

    with pytest.raises(RuntimeError, match="Unable to initialize auto-selected"):
        qa._initialize_llm()

    assert qa.active_llm_backend is None
    assert qa.llm is None


def test_explicit_endpoint_without_token_fails_closed(monkeypatch):
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    qa = DocumentQA(device="cpu", llm_backend="endpoint", hf_token=None)

    with pytest.raises(RuntimeError, match="token is required for endpoint inference"):
        qa._initialize_llm()


def test_explicit_local_backend_failure_fails_closed(monkeypatch):
    def failing_local_loader(self, model_id):
        raise ValueError("boom")

    monkeypatch.setattr(DocumentQA, "_load_local_model", failing_local_loader)
    qa = DocumentQA(device="cpu", llm_backend="local", hf_token="real-token")

    with pytest.raises(RuntimeError, match="Unable to initialize local LLM"):
        qa._initialize_llm()

    assert qa.active_llm_backend is None
    assert qa.llm is None


def test_explicit_ollama_backend_uses_ollama_without_hf_token(monkeypatch):
    loaded_models = []

    def fake_ollama_loader(self, model_id):
        loaded_models.append(model_id)
        return MockLLM()

    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "nemotron-3-nano:4b")
    monkeypatch.setattr(DocumentQA, "_load_ollama_model", fake_ollama_loader)
    qa = DocumentQA(device="cpu", llm_backend="ollama", hf_token=None)

    qa._initialize_llm()

    assert loaded_models == ["nemotron-3-nano:4b"]
    assert qa.active_llm_backend == "ollama"
    assert qa.loaded_model_id == "nemotron-3-nano:4b"
    assert qa.loaded_model_label == "Ollama (nemotron-3-nano:4b)"
    assert qa.status().active_model_label == "Ollama (nemotron-3-nano:4b)"


def test_explicit_ollama_backend_with_dummy_hf_token_does_not_mock(monkeypatch):
    def fake_ollama_loader(self, model_id):
        return MockLLM()

    monkeypatch.setattr(DocumentQA, "_load_ollama_model", fake_ollama_loader)
    qa = DocumentQA(device="cpu", llm_backend="ollama", hf_token="dummy")

    qa._initialize_llm()

    assert qa.active_llm_backend == "ollama"


def test_explicit_ollama_backend_failure_fails_closed(monkeypatch):
    def failing_ollama_loader(self, model_id):
        raise ValueError("ollama down")

    monkeypatch.setattr(DocumentQA, "_load_ollama_model", failing_ollama_loader)
    qa = DocumentQA(device="cpu", llm_backend="ollama", hf_token=None)

    with pytest.raises(RuntimeError, match="Unable to initialize ollama LLM"):
        qa._initialize_llm()

    assert qa.active_llm_backend is None
    assert qa.llm is None


def test_explicit_openai_compatible_backend_uses_chat_completions(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "ok"}}]}
            ).encode("utf-8")

    def fake_open_openai_compatible_request(request, *, timeout):
        requests.append(
            {
                "url": request.full_url,
                "headers": {
                    key.lower(): value for key, value in request.header_items()
                },
                "payload": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return FakeResponse()

    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "secret-token")
    monkeypatch.setenv("OPENAI_COMPAT_TIMEOUT", "7")
    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible", hf_token=None)

    qa._initialize_llm()
    answer = qa.llm._call("Question?", stop=["END"])

    assert answer == "ok"
    assert qa.active_llm_backend == "openai-compatible"
    assert qa.loaded_model_id == "local-chat"
    assert qa.loaded_model_label == "OpenAI-compatible (local-chat)"
    assert len(requests) == 2
    assert requests[0]["url"] == "http://localhost:8000/v1/chat/completions"
    assert requests[0]["headers"]["authorization"] == "Bearer secret-token"
    assert requests[0]["timeout"] == 7
    assert requests[0]["payload"]["model"] == "local-chat"
    assert requests[0]["payload"]["messages"] == [
        {"role": "user", "content": "Respond with ok."}
    ]
    assert requests[0]["payload"]["max_tokens"] == 1
    assert requests[1]["payload"]["messages"] == [
        {"role": "user", "content": "Question?"}
    ]
    assert requests[1]["payload"]["stop"] == ["END"]


def test_openai_compatible_does_not_use_generic_openai_api_key(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "ok"}}]}
            ).encode("utf-8")

    def fake_open_openai_compatible_request(request, *, timeout):
        requests.append(
            {key.lower(): value for key, value in request.header_items()}
        )
        return FakeResponse()

    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-be-used")
    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible", hf_token=None)

    qa._initialize_llm()

    assert "authorization" not in requests[0]


def test_openai_compatible_http_error_body_is_not_reflected(monkeypatch):
    class SecretHttpError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__(
                url="http://localhost:8000/v1/chat/completions",
                code=401,
                msg="Unauthorized SECRET_STATUS_TEXT",
                hdrs={},
                fp=None,
            )

        def read(self):
            return b"debug echo SECRET_API_KEY from proxy"

    def fake_open_openai_compatible_request(request, *, timeout):
        raise SecretHttpError()

    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible", hf_token=None)

    with pytest.raises(RuntimeError) as exc_info:
        qa._initialize_llm()

    message = str(exc_info.value)
    inner_error = exc_info.value.__cause__
    assert inner_error is not None
    cause_message = str(inner_error)
    assert "HTTP 401" in message
    assert "HTTP 401" in cause_message
    assert "SECRET_API_KEY" not in message
    assert "SECRET_API_KEY" not in cause_message
    assert "SECRET_STATUS_TEXT" not in message
    assert "SECRET_STATUS_TEXT" not in cause_message
    assert "debug echo" not in message
    assert "debug echo" not in cause_message
    assert inner_error.__cause__ is None


def test_openai_compatible_url_error_reason_is_not_reflected(monkeypatch):
    def fake_open_openai_compatible_request(request, *, timeout):
        raise urllib.error.URLError("proxy failed with SECRET_REASON_TOKEN")

    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible", hf_token=None)

    with pytest.raises(RuntimeError) as exc_info:
        qa._initialize_llm()

    message = str(exc_info.value)
    inner_error = exc_info.value.__cause__
    assert inner_error is not None
    cause_message = str(inner_error)
    assert "http://localhost:8000/v1/chat/completions" in message
    assert "http://localhost:8000/v1/chat/completions" in cause_message
    assert "SECRET_REASON_TOKEN" not in message
    assert "SECRET_REASON_TOKEN" not in cause_message
    assert "proxy failed" not in message
    assert "proxy failed" not in cause_message
    assert inner_error.__cause__ is None


def test_openai_compatible_backend_alias_is_supported(monkeypatch):
    def fake_openai_compatible_loader(self, model_id):
        return MockLLM()

    monkeypatch.setattr(
        DocumentQA, "_load_openai_compatible_model", fake_openai_compatible_loader
    )
    qa = DocumentQA(device="cpu", llm_backend="openai_compatible", hf_token=None)

    qa._initialize_llm()

    assert qa.llm_backend == "openai-compatible"
    assert qa.active_llm_backend == "openai-compatible"


def test_explicit_openai_compatible_without_base_url_fails_closed(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible", hf_token=None)

    with pytest.raises(RuntimeError, match="OPENAI_COMPAT_BASE_URL"):
        qa._initialize_llm()

    assert qa.active_llm_backend is None
    assert qa.llm is None


def test_explicit_openai_compatible_without_model_fails_closed(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("OPENAI_COMPAT_MODEL", raising=False)
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible", hf_token=None)

    with pytest.raises(RuntimeError, match="OPENAI_COMPAT_MODEL"):
        qa._initialize_llm()

    assert qa.active_llm_backend is None
    assert qa.llm is None


@pytest.mark.parametrize(
    "base_url, expected",
    [
        ("http://localhost:8000/v1/", "http://localhost:8000/v1"),
        ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1"),
        ("http://[::1]:8000/v1", "http://[::1]:8000/v1"),
        ("https://gateway.example/v1", "https://gateway.example/v1"),
    ],
)
def test_normalize_openai_compatible_base_url_accepts_http_forms(
    base_url, expected
):
    assert normalize_openai_compatible_base_url(base_url) == expected


@pytest.mark.parametrize(
    "base_url",
    [
        "not-a-url",
        "file:///tmp/model",
        "http://user:pass@localhost:8000/v1",
        "http://localhost:bad/v1",
        "http://gateway.example/v1",
        "http://localhost:8000/v1?token=x",
        "http://localhost:8000/v1#fragment",
    ],
)
def test_normalize_openai_compatible_base_url_rejects_unsafe_forms(base_url):
    with pytest.raises(RuntimeError, match="OPENAI_COMPAT_BASE_URL"):
        normalize_openai_compatible_base_url(base_url)


@pytest.mark.parametrize(
    "base_url, expected",
    [
        (
            "https://user:SECRET_PASSWORD@gateway.example/v1",
            "https://gateway.example/v1",
        ),
        (
            "https://gateway.example/v1?api_key=SECRET_QUERY_TOKEN",
            "https://gateway.example/v1",
        ),
        (
            "https://gateway.example:8443/v1#SECRET_FRAGMENT",
            "https://gateway.example:8443/v1",
        ),
        ("not-a-url", "<invalid>"),
        ("", "<unset>"),
    ],
)
def test_safe_openai_compatible_base_url_for_error_redacts_secrets(
    base_url, expected
):
    assert safe_openai_compatible_base_url_for_error(base_url) == expected


def exception_chain_messages(error):
    messages = []
    seen = set()
    stack = [error]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        messages.append(str(current))
        stack.extend([current.__cause__, current.__context__])
    return messages


@pytest.mark.parametrize(
    "unsafe_base_url, secret",
    [
        ("https://user:SECRET_PASSWORD@gateway.example/v1", "SECRET_PASSWORD"),
        (
            "https://gateway.example/v1?api_key=SECRET_QUERY_TOKEN",
            "SECRET_QUERY_TOKEN",
        ),
        ("https://gateway.example/v1#SECRET_FRAGMENT", "SECRET_FRAGMENT"),
    ],
)
def test_openai_compatible_initialization_error_redacts_unsafe_url(
    monkeypatch, unsafe_base_url, secret
):
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", unsafe_base_url)
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible", hf_token=None)

    with pytest.raises(RuntimeError) as exc_info:
        qa._initialize_llm()

    message = str(exc_info.value)
    cause_message = str(exc_info.value.__cause__)
    assert secret not in message
    assert secret not in cause_message
    assert unsafe_base_url not in message
    assert unsafe_base_url not in cause_message
    assert "gateway.example" in message


def test_openai_compatible_invalid_port_error_drops_parser_cause(monkeypatch):
    monkeypatch.setenv(
        "OPENAI_COMPAT_BASE_URL", "http://localhost:SECRET_PORT/v1"
    )
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible", hf_token=None)

    with pytest.raises(RuntimeError) as exc_info:
        qa._initialize_llm()

    chain_messages = exception_chain_messages(exc_info.value)
    assert chain_messages
    assert all("SECRET_PORT" not in message for message in chain_messages)
    assert all(
        "http://localhost:SECRET_PORT/v1" not in message
        for message in chain_messages
    )
    assert exc_info.value.__cause__ is not None
    assert exc_info.value.__cause__.__cause__ is None
    assert exc_info.value.__cause__.__context__ is None


def test_openai_compatible_rejects_remote_http_before_sending_api_key(
    monkeypatch,
):
    request_attempted = False

    def fake_open_openai_compatible_request(request, *, timeout):
        nonlocal request_attempted
        request_attempted = True
        raise AssertionError("remote HTTP request should not be sent")

    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://gateway.example/v1")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "local-chat")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "SECRET_API_KEY")
    monkeypatch.setattr(
        document_qa_module,
        "open_openai_compatible_request",
        fake_open_openai_compatible_request,
    )
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible", hf_token=None)

    with pytest.raises(RuntimeError) as exc_info:
        qa._initialize_llm()

    chain_messages = exception_chain_messages(exc_info.value)
    assert not request_attempted
    assert any("HTTPS" in message for message in chain_messages)
    assert all("SECRET_API_KEY" not in message for message in chain_messages)


def test_openai_compatible_loopback_http_bypasses_proxy_urlopen(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "ok"}}]}
            ).encode("utf-8")

    def fake_no_proxy_open(request, timeout):
        requests.append(
            {
                "url": request.full_url,
                "headers": {
                    key.lower(): value for key, value in request.header_items()
                },
                "timeout": timeout,
            }
        )
        return FakeResponse()

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8443")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(
            AssertionError(
                "OpenAI-compatible loopback HTTP must bypass proxy-aware urlopen"
            )
        ),
    )
    monkeypatch.setattr(
        document_qa_module.OPENAI_COMPAT_NO_PROXY_OPENER,
        "open",
        fake_no_proxy_open,
    )
    llm = OpenAICompatibleLLM(
        model="local-chat",
        base_url="http://localhost:8000/v1",
        api_key="SECRET_PROXY_TOKEN",
        timeout=7,
    )

    llm.validate_model_available()

    assert requests == [
        {
            "url": "http://localhost:8000/v1/chat/completions",
            "headers": {
                "content-type": "application/json",
                "authorization": "Bearer SECRET_PROXY_TOKEN",
            },
            "timeout": 7,
        }
    ]


def test_openai_compatible_ignores_dummy_hf_token(monkeypatch):
    def fake_openai_compatible_loader(self, model_id):
        return MockLLM()

    monkeypatch.setattr(
        DocumentQA, "_load_openai_compatible_model", fake_openai_compatible_loader
    )
    qa = DocumentQA(device="cpu", llm_backend="openai-compatible", hf_token="dummy")

    qa._initialize_llm()

    assert qa.active_llm_backend == "openai-compatible"
    assert isinstance(qa.llm, MockLLM)


@pytest.mark.parametrize("backend", ["endpoint", "local"])
def test_explicit_real_backend_with_dummy_token_fails_closed(backend):
    qa = DocumentQA(device="cpu", llm_backend=backend, hf_token="dummy")

    with pytest.raises(RuntimeError, match="Dummy HuggingFace token"):
        qa._initialize_llm()

    assert qa.active_llm_backend is None
    assert qa.llm is None


def test_local_backend_without_token_still_attempts_local_public_model(monkeypatch):
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)

    def fake_local_loader(self, model_id):
        return MockLLM()

    monkeypatch.setattr(DocumentQA, "_load_local_model", fake_local_loader)
    qa = DocumentQA(device="cpu", llm_backend="local", hf_token=None)

    qa._initialize_llm()

    assert qa.active_llm_backend == "local"
    assert qa.loaded_model_id == "Qwen/Qwen2.5-1.5B-Instruct"


def test_ollama_llm_validates_model_and_generates(monkeypatch):
    requests = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_open_ollama_request_no_proxy(request, *, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        requests.append((request.full_url, payload, timeout))
        if request.full_url.endswith("/api/show"):
            return FakeResponse({"model_info": {}})
        if request.full_url.endswith("/api/generate"):
            return FakeResponse(
                {
                    "response": (
                        "The model is reasoning about the prompt.\n"
                        "</think>\nProject Phoenix answer [1]."
                    )
                }
            )
        raise AssertionError(f"Unexpected Ollama URL: {request.full_url}")

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(
            AssertionError("OllamaLLM must bypass proxy-aware urlopen")
        ),
    )
    monkeypatch.setattr(
        "src.DocumentQA.open_ollama_request_no_proxy",
        fake_open_ollama_request_no_proxy,
    )
    llm = OllamaLLM(
        model=DEFAULT_OLLAMA_MODEL,
        base_url="http://ollama.test/",
        timeout=7,
        options={"temperature": 0, "num_predict": 160},
    )

    llm.validate_model_available()
    answer = llm.invoke("Answer with citation.", stop=["END"])

    assert answer == "Project Phoenix answer [1]."
    assert llm._strip_thinking_text("<think>unfinished reasoning") == ""
    assert requests[0] == (
        "http://ollama.test/api/show",
        {"model": DEFAULT_OLLAMA_MODEL},
        7,
    )
    assert requests[1][0] == "http://ollama.test/api/generate"
    assert requests[1][1]["model"] == DEFAULT_OLLAMA_MODEL
    assert requests[1][1]["prompt"] == "Answer with citation."
    assert requests[1][1]["stream"] is False
    assert requests[1][1]["think"] is False
    assert requests[1][1]["options"] == {
        "temperature": 0,
        "num_predict": 160,
        "stop": ["END"],
    }
