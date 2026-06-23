import io
import json

from src.DocumentQA import (
    SELF_CHECK_REFUSAL_ANSWER,
    AnswerCitation,
    AnswerSelfCheck,
    AnswerTrace,
    QueryResult,
)
from src.ollama_model_eval import evaluate_model, main


def traced_result(answer, *, outcome="supported", reasons=None, citation_count=1):
    citations = [
        AnswerCitation(
            citation_id=1,
            source_name="project_phoenix_brief.md",
            page=None,
            chunk_index=0,
            excerpt=(
                "Project Phoenix launches in June 2026. "
                "The approved Project Phoenix budget is $42 million. "
                "Alex Rivera owns the Project Phoenix rollout."
            ),
        )
        for _ in range(citation_count)
    ]
    return QueryResult(
        answer=answer,
        trace=AnswerTrace(
            question="",
            document_name="project_phoenix_brief.md",
            backend="ollama",
            model_label="fake-model",
            retrieved_chunk_count=len(citations),
            citations=citations,
            self_check=AnswerSelfCheck(
                outcome=outcome,
                reasons=reasons or ["mechanical_checks_passed", "llm_verifier_supported"],
            ),
        ),
    )


class PassingEvalQA:
    def process_document(self, document_path):
        self.document_path = document_path

    def query_with_trace(self, question):
        lowered = question.lower()
        if "venue" in lowered:
            return traced_result(
                SELF_CHECK_REFUSAL_ANSWER,
                outcome="needs_refusal",
                reasons=["llm_verifier_insufficient"],
            )
        if "budget" in lowered:
            return traced_result("The approved Project Phoenix budget is $42 million [1].")
        if "owns" in lowered or "owner" in lowered:
            return traced_result("Alex Rivera owns the Project Phoenix rollout [1].")
        return traced_result("Project Phoenix launches in June 2026 [1].")


class HallucinatingEvalQA(PassingEvalQA):
    def query_with_trace(self, question):
        if "venue" in question.lower():
            return traced_result(
                "Project Phoenix launches from Lunar Base Alpha [1].",
                outcome="supported",
            )
        return super().query_with_trace(question)


class HallucinatedCitationEvalQA(PassingEvalQA):
    def query_with_trace(self, question):
        if question.lower().startswith("when does"):
            return traced_result(
                "Project Phoenix launches in June 2026 [1]. Budget is $42 million [999].",
                outcome="supported",
            )
        return super().query_with_trace(question)


def passing_factory(model, base_url, timeout):
    return PassingEvalQA()


def hallucinating_factory(model, base_url, timeout):
    return HallucinatingEvalQA()


def hallucinated_citation_factory(model, base_url, timeout):
    return HallucinatedCitationEvalQA()


def failing_factory(model, base_url, timeout):
    raise RuntimeError("ollama model not found")


def test_evaluate_model_scores_all_golden_cases_provider_free():
    result = evaluate_model(
        "fake-model",
        qa_factory=passing_factory,
        unload_after=False,
    )

    assert result.passed is True
    assert result.passed_cases == result.total_cases == 4
    assert [case.case_id for case in result.case_results] == [
        "launch_date",
        "budget",
        "owner",
        "unsupported_venue",
    ]


def test_evaluate_model_fails_hallucinated_unsupported_answer():
    result = evaluate_model(
        "fake-model",
        qa_factory=hallucinating_factory,
        unload_after=False,
    )

    assert result.passed is False
    failed_cases = [case for case in result.case_results if not case.passed]
    assert [case.case_id for case in failed_cases] == ["unsupported_venue"]
    assert "Lunar Base Alpha" in failed_cases[0].answer


def test_evaluate_model_fails_hallucinated_inline_citation_id():
    result = evaluate_model(
        "fake-model",
        qa_factory=hallucinated_citation_factory,
        unload_after=False,
    )

    assert result.passed is False
    failed_cases = [case for case in result.case_results if not case.passed]
    assert [case.case_id for case in failed_cases] == ["launch_date"]
    assert "[999]" in failed_cases[0].answer
    assert "invalid_inline_citation" in failed_cases[0].self_check_reasons


def test_main_outputs_json_and_can_compare_without_failing(monkeypatch):
    unloaded_models = []
    monkeypatch.setattr(
        "src.ollama_model_eval.unload_ollama_model",
        lambda model, **kwargs: unloaded_models.append((model, kwargs)),
    )
    output = io.StringIO()

    exit_code = main(
        [
            "--models",
            "model-a",
            "model-b",
            "--allow-multi-model",
            "--json",
            "--no-fail",
        ],
        qa_factory=passing_factory,
        output_stream=output,
    )

    assert exit_code == 0
    payload = json.loads(output.getvalue())
    assert [result["model"] for result in payload] == ["model-a", "model-b"]
    assert all(result["passed"] for result in payload)
    assert [model for model, _kwargs in unloaded_models] == ["model-a", "model-b"]


def test_main_rejects_multiple_models_without_explicit_override():
    called = False

    def fail_if_called(model, base_url, timeout):
        nonlocal called
        called = True
        return PassingEvalQA()

    output = io.StringIO()

    exit_code = main(
        ["--models", "model-a", "model-b", "--no-fail"],
        qa_factory=fail_if_called,
        output_stream=output,
    )

    assert exit_code == 2
    assert called is False
    assert "Run one model per command" in output.getvalue()


def test_main_returns_failure_for_initialization_error(monkeypatch):
    monkeypatch.setattr(
        "src.ollama_model_eval.unload_ollama_model",
        lambda model, **kwargs: None,
    )
    output = io.StringIO()

    exit_code = main(
        ["--models", "missing-model"],
        qa_factory=failing_factory,
        output_stream=output,
    )

    assert exit_code == 1
    assert "missing-model" in output.getvalue()
    assert "ollama model not found" in output.getvalue()


def test_evaluate_model_cleanup_ignores_malformed_base_url():
    result = evaluate_model(
        "missing-model",
        base_url="not-a-url",
        qa_factory=failing_factory,
    )

    assert result.passed is False
    assert result.initialization_error == "ollama model not found"


def test_main_can_run_single_case_for_low_memory_smoke(monkeypatch):
    monkeypatch.setattr(
        "src.ollama_model_eval.unload_ollama_model",
        lambda model, **kwargs: None,
    )
    output = io.StringIO()

    exit_code = main(
        ["--models", "model-a", "--case", "launch_date"],
        qa_factory=passing_factory,
        output_stream=output,
    )

    assert exit_code == 0
    rendered = output.getvalue()
    assert "PASS model-a: 1/1 cases" in rendered
    assert "launch_date" in rendered
    assert "budget" not in rendered


def test_main_rejects_unknown_case_before_running_model(monkeypatch):
    called = False

    def fail_if_called(model, base_url, timeout):
        nonlocal called
        called = True
        return PassingEvalQA()

    output = io.StringIO()

    exit_code = main(
        ["--models", "model-a", "--case", "missing_case"],
        qa_factory=fail_if_called,
        output_stream=output,
    )

    assert exit_code == 2
    assert called is False
    assert "Unknown golden case id" in output.getvalue()


def test_main_uses_ollama_base_url_from_environment(monkeypatch):
    observed_base_urls = []
    unloaded_models = []

    def recording_factory(model, base_url, timeout):
        observed_base_urls.append(base_url)
        return PassingEvalQA()

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.example")
    monkeypatch.setattr(
        "src.ollama_model_eval.unload_ollama_model",
        lambda model, **kwargs: unloaded_models.append(kwargs["base_url"]),
    )
    output = io.StringIO()

    exit_code = main(
        ["--models", "model-a", "--case", "launch_date"],
        qa_factory=recording_factory,
        output_stream=output,
    )

    assert exit_code == 0
    assert observed_base_urls == ["http://ollama.example"]
    assert unloaded_models == ["http://ollama.example"]
