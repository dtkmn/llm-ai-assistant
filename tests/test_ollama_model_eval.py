import io
import json

import src.ollama_model_eval as ollama_model_eval
from src.DocumentQA import (
    DEFAULT_OLLAMA_EMBEDDINGS_MODEL,
    SELF_CHECK_REFUSAL_ANSWER,
    AnswerCitation,
    AnswerSelfCheck,
    AnswerTrace,
    QueryResult,
)
from src.ollama_model_eval import (
    build_ollama_qa,
    evaluate_model,
    main,
    normalize_local_ollama_base_url,
    unload_ollama_model,
)


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


def test_build_ollama_qa_uses_provider_embedding_configuration(monkeypatch):
    monkeypatch.delenv("EMBEDDINGS_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_EMBED_MODEL", raising=False)

    qa = build_ollama_qa("nemotron-3-nano:4b", "http://localhost:11434", 30)

    assert qa.llm_backend == "ollama"
    assert qa.ollama_model == "nemotron-3-nano:4b"
    assert qa.embeddings_model == DEFAULT_OLLAMA_EMBEDDINGS_MODEL
    assert qa.embeddings is None


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


def test_normalize_local_ollama_base_url_accepts_loopback_forms():
    assert (
        normalize_local_ollama_base_url("http://localhost:11434/")
        == "http://localhost:11434"
    )
    assert (
        normalize_local_ollama_base_url("http://127.0.0.1:11435")
        == "http://127.0.0.1:11435"
    )
    assert normalize_local_ollama_base_url("http://[::1]:11434") == "http://[::1]:11434"


def test_main_rejects_remote_ollama_base_url_before_running_model():
    called = False

    def fail_if_called(model, base_url, timeout):
        nonlocal called
        called = True
        return PassingEvalQA()

    output = io.StringIO()

    exit_code = main(
        ["--models", "model-a", "--base-url", "http://ollama.example"],
        qa_factory=fail_if_called,
        output_stream=output,
    )

    assert exit_code == 2
    assert called is False
    assert "loopback base URLs" in output.getvalue()


def test_unload_ollama_model_uses_no_proxy_opener(monkeypatch):
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b"{}"

    def fake_open(request, timeout):
        calls.append((request.full_url, json.loads(request.data.decode("utf-8")), timeout))
        return FakeResponse()

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8443")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(
            AssertionError("Ollama cleanup must bypass proxy-aware urlopen")
        ),
    )
    monkeypatch.setattr(ollama_model_eval.NO_PROXY_OPENER, "open", fake_open)

    unload_ollama_model("model-a", base_url="http://localhost:11434", timeout=3)

    assert calls == [
        (
            "http://localhost:11434/api/generate",
            {"model": "model-a", "prompt": "", "stream": False, "keep_alive": 0},
            3,
        )
    ]


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

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435/")
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
    assert observed_base_urls == ["http://127.0.0.1:11435"]
    assert unloaded_models == ["http://127.0.0.1:11435"]


def test_main_uses_generic_llm_model_env(monkeypatch):
    observed_models = []

    def recording_factory(model, base_url, timeout):
        observed_models.append(model)
        return PassingEvalQA()

    monkeypatch.setenv("LLM_MODEL", "custom-chat:4b")
    monkeypatch.setenv("OLLAMA_MODEL", "ignored-chat:4b")
    monkeypatch.delenv("OLLAMA_EVAL_MODELS", raising=False)
    monkeypatch.setattr(
        "src.ollama_model_eval.unload_ollama_model",
        lambda model, **kwargs: None,
    )
    output = io.StringIO()

    exit_code = main(
        ["--case", "launch_date", "--no-unload"],
        qa_factory=recording_factory,
        output_stream=output,
    )

    assert exit_code == 0
    assert observed_models == ["custom-chat:4b"]


def test_main_empty_eval_models_env_falls_through_to_default(monkeypatch):
    observed_models = []

    def recording_factory(model, base_url, timeout):
        observed_models.append(model)
        return PassingEvalQA()

    monkeypatch.setenv("OLLAMA_EVAL_MODELS", ",")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setattr(
        "src.ollama_model_eval.unload_ollama_model",
        lambda model, **kwargs: None,
    )
    output = io.StringIO()

    exit_code = main(
        ["--case", "launch_date", "--no-unload", "--json"],
        qa_factory=recording_factory,
        output_stream=output,
    )

    payload = json.loads(output.getvalue())
    assert exit_code == 0
    assert observed_models == ["nemotron-3-nano:4b"]
    assert len(payload) == 1


def test_main_rejects_explicit_blank_model_tags():
    called = False

    def fail_if_called(model, base_url, timeout):
        nonlocal called
        called = True
        return PassingEvalQA()

    output = io.StringIO()

    exit_code = main(
        ["--models", " ", "--case", "launch_date"],
        qa_factory=fail_if_called,
        output_stream=output,
    )

    assert exit_code == 2
    assert called is False
    assert "No model tags selected" in output.getvalue()
