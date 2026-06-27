import io
import json

import src.loop_eval as loop_eval
from src.loop_eval import (
    build_provider_free_qa,
    evaluate_model,
    main,
    results_to_dict,
)


def test_provider_free_loop_eval_scores_all_cases_with_loop_reports():
    result = evaluate_model("provider-free-golden", mode="fake")

    assert result.passed is True
    assert result.passed_cases == result.total_cases == 4
    assert [case.case_id for case in result.case_results] == [
        "launch_date",
        "budget",
        "owner",
        "unsupported_venue",
    ]
    launch_case = result.case_results[0]
    assert launch_case.final_decision == "supported"
    assert launch_case.loop_report["schema_version"] == "loop-report/v1"
    assert launch_case.loop_report["run"]["backend"] == "provider-free"
    assert launch_case.loop_report["run"]["model_label"] == (
        "Provider-free golden model (provider-free-golden)"
    )
    assert launch_case.loop_report["run"]["context_provider"] == "document"
    assert launch_case.phases == [
        "context_select",
        "retrieve",
        "draft",
        "format_check",
        "mechanical_check",
        "verify",
        "final",
    ]
    refusal_case = result.case_results[-1]
    assert refusal_case.final_decision == "refuse"
    assert "refuse" in refusal_case.phases
    assert refusal_case.self_check_outcome == "needs_refusal"


def test_loop_eval_results_dict_has_artifact_schema_and_loop_evidence():
    result = evaluate_model(
        "provider-free-golden",
        mode="fake",
    )

    payload = results_to_dict([result])

    assert payload["schema_version"] == "loop-eval/v1"
    case_payload = payload["results"][0]["case_results"][0]
    assert case_payload["passed"] is True
    assert case_payload["loop_report"]["run"]["final_decision"] == "supported"
    assert "verify" in case_payload["phases"]


def test_loop_eval_cli_fake_mode_writes_json_artifact(tmp_path):
    artifact_path = tmp_path / "loop-eval.json"
    output = io.StringIO()

    exit_code = main(
        [
            "--mode",
            "fake",
            "--case",
            "launch_date",
            "--artifact",
            str(artifact_path),
            "--json",
        ],
        output_stream=output,
    )

    assert exit_code == 0
    rendered_payload = json.loads(output.getvalue())
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert rendered_payload == artifact_payload
    assert artifact_payload["schema_version"] == "loop-eval/v1"
    assert artifact_payload["results"][0]["mode"] == "fake"
    case_result = artifact_payload["results"][0]["case_results"][0]
    assert case_result["case_id"] == "launch_date"
    assert case_result["loop_report"]["schema_version"] == "loop-report/v1"
    assert case_result["loop_report"]["run"]["final_decision"] == "supported"


def test_loop_eval_fake_mode_ignores_ollama_base_url_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.example")
    output = io.StringIO()

    exit_code = main(
        ["--mode", "fake", "--case", "launch_date"],
        output_stream=output,
    )

    assert exit_code == 0
    assert "PASS fake/provider-free-golden: 1/1 cases" in output.getvalue()


def test_loop_eval_cli_rejects_unknown_case_before_running_model():
    called = False

    def fail_if_called(model, base_url, timeout):
        nonlocal called
        called = True
        return build_provider_free_qa(model, base_url, timeout)

    output = io.StringIO()

    exit_code = main(
        ["--mode", "fake", "--case", "missing_case"],
        qa_factory=fail_if_called,
        output_stream=output,
    )

    assert exit_code == 2
    assert called is False
    assert "Unknown golden case id" in output.getvalue()


def test_loop_eval_ollama_mode_rejects_remote_base_url_before_model_work():
    called = False

    def fail_if_called(model, base_url, timeout):
        nonlocal called
        called = True
        return build_provider_free_qa(model, base_url, timeout)

    output = io.StringIO()

    exit_code = main(
        [
            "--mode",
            "ollama",
            "--models",
            "model-a",
            "--base-url",
            "http://ollama.example",
        ],
        qa_factory=fail_if_called,
        output_stream=output,
    )

    assert exit_code == 2
    assert called is False
    assert "loopback base URLs" in output.getvalue()


def test_loop_eval_ollama_mode_rejects_multi_model_without_override():
    called = False

    def fail_if_called(model, base_url, timeout):
        nonlocal called
        called = True
        return build_provider_free_qa(model, base_url, timeout)

    output = io.StringIO()

    exit_code = main(
        ["--mode", "ollama", "--models", "model-a", "model-b"],
        qa_factory=fail_if_called,
        output_stream=output,
    )

    assert exit_code == 2
    assert called is False
    assert "Run one model per command" in output.getvalue()


def test_loop_eval_ollama_empty_eval_models_env_falls_through_to_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_EVAL_MODELS", ",")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setattr(loop_eval, "unload_ollama_model", lambda model, **kwargs: None)
    output = io.StringIO()

    exit_code = main(
        ["--mode", "ollama", "--no-unload", "--json"],
        qa_factory=build_provider_free_qa,
        output_stream=output,
    )

    payload = json.loads(output.getvalue())
    assert exit_code == 0
    assert len(payload["results"]) == 1
    assert payload["results"][0]["model"] == loop_eval.DEFAULT_OLLAMA_MODEL


def test_loop_eval_ollama_uses_generic_llm_model_env(monkeypatch):
    monkeypatch.delenv("OLLAMA_EVAL_MODELS", raising=False)
    monkeypatch.setenv("LLM_MODEL", "custom-chat:4b")
    monkeypatch.setenv("OLLAMA_MODEL", "ignored-chat:4b")
    monkeypatch.setattr(loop_eval, "unload_ollama_model", lambda model, **kwargs: None)
    output = io.StringIO()

    exit_code = main(
        ["--mode", "ollama", "--no-unload", "--json"],
        qa_factory=build_provider_free_qa,
        output_stream=output,
    )

    payload = json.loads(output.getvalue())
    assert exit_code == 0
    assert payload["results"][0]["model"] == "custom-chat:4b"


def test_loop_eval_rejects_empty_model_list_before_scoring():
    called = False

    def fail_if_called(model, base_url, timeout):
        nonlocal called
        called = True
        return build_provider_free_qa(model, base_url, timeout)

    output = io.StringIO()

    exit_code = main(
        ["--mode", "ollama", "--models", " "],
        qa_factory=fail_if_called,
        output_stream=output,
    )

    assert exit_code == 2
    assert called is False
    assert "No model tags selected" in output.getvalue()


def test_loop_eval_ollama_mode_defaults_to_one_case(monkeypatch):
    monkeypatch.setattr(loop_eval, "unload_ollama_model", lambda model, **kwargs: None)
    output = io.StringIO()

    exit_code = main(
        ["--mode", "ollama", "--models", "model-a"],
        qa_factory=build_provider_free_qa,
        output_stream=output,
    )

    assert exit_code == 0
    assert "PASS ollama/model-a: 1/1 cases" in output.getvalue()


def test_loop_eval_ollama_mode_all_cases_requires_explicit_flag():
    output = io.StringIO()

    exit_code = main(
        ["--mode", "ollama", "--models", "model-a", "--all-cases", "--no-unload"],
        qa_factory=build_provider_free_qa,
        output_stream=output,
    )

    assert exit_code == 0
    assert "PASS ollama/model-a: 4/4 cases" in output.getvalue()


def test_loop_eval_ollama_mode_unloads_after_run(monkeypatch):
    unloaded = []
    output = io.StringIO()
    monkeypatch.setattr(
        loop_eval,
        "unload_ollama_model",
        lambda model, **kwargs: unloaded.append((model, kwargs["base_url"])),
    )

    exit_code = main(
        [
            "--mode",
            "ollama",
            "--models",
            "model-a",
            "--case",
            "launch_date",
            "--no-fail",
        ],
        qa_factory=build_provider_free_qa,
        output_stream=output,
    )

    assert exit_code == 0
    assert unloaded == [("model-a", "http://localhost:11434")]
    assert "PASS ollama/model-a: 1/1 cases" in output.getvalue()
