import json
from datetime import datetime, timezone

import pytest

from src.loop_engine import (
    DEFAULT_LOOP_RECIPE_ID,
    GuardrailDecision,
    HumanReviewRequest,
    LoopDecision,
    LoopPhase,
    LoopPolicy,
    LoopReport,
    LoopRecipe,
    LoopRun,
    LoopSession,
    LoopStep,
    PUBLIC_REDACTION_REASON,
    PUBLIC_REDACTION_TEXT,
    VerificationOutcome,
    VerificationResult,
)


def utc(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def test_loop_report_round_trips_through_json():
    started_at = utc("2026-06-23T10:00:00")
    ended_at = utc("2026-06-23T10:00:01")
    verification = VerificationResult(
        outcome=VerificationOutcome.SUPPORTED,
        reasons=("mechanical_checks_passed", "llm_verifier_supported"),
        verifier="ollama",
        raw_response='{"outcome":"supported"}',
        metadata={"citation_count": 1},
    )
    step = LoopStep(
        step_id="step_verify",
        phase=LoopPhase.VERIFY,
        decision=LoopDecision.SUPPORTED,
        name="LLM verifier",
        started_at=started_at,
        ended_at=ended_at,
        input_summary="answer plus cited excerpts",
        output_summary="answer supported",
        backend="ollama",
        model_label="Ollama (nemotron-3-nano:4b)",
        retry_count=0,
        verification=verification,
        metadata={"prompt_chunks": 1},
    )
    run = LoopRun(
        run_id="run_phoenix",
        session_id="session_local",
        user_input="When does Project Phoenix launch?",
        context_provider="document",
        backend="ollama",
        model_label="Ollama (nemotron-3-nano:4b)",
        policy=LoopPolicy(max_retries=1),
        started_at=started_at,
        completed_at=ended_at,
        steps=(step,),
        final_decision=LoopDecision.SUPPORTED,
        final_answer="Project Phoenix launches in June 2026 [1].",
        metadata={"document": "project_phoenix_brief.md"},
    )
    report = LoopReport(run=run)

    payload = json.loads(json.dumps(report.to_dict()))
    restored = LoopReport.from_dict(payload)

    assert restored == report
    assert payload["schema_version"] == "loop-report/v1"
    assert payload["run"]["steps"][0]["phase"] == "verify"
    assert payload["run"]["steps"][0]["duration_ms"] == 1000
    assert payload["run"]["final_decision"] == "supported"


def test_loop_recipe_round_trips_and_summarizes():
    recipe = LoopRecipe(
        recipe_id=DEFAULT_LOOP_RECIPE_ID,
        name="General assistant loop",
        description="Default recipe",
        goal="Answer clearly.",
        instructions="Be direct.",
        success_criteria=("Addresses the request.", "Names uncertainty."),
        stop_condition="Stop after a safe final answer.",
        context_provider="auto",
        model_profile="quality",
        verifier="default",
        created_at=utc("2026-06-23T10:00:00"),
        updated_at=utc("2026-06-23T10:01:00"),
    )

    restored = LoopRecipe.from_dict(json.loads(json.dumps(recipe.to_dict())))

    assert restored == recipe
    assert restored.summary_dict()["is_default"] is True
    assert restored.runtime_dict()["success_criteria"] == [
        "Addresses the request.",
        "Names uncertainty.",
    ]


def test_loop_recipe_rejects_missing_goal():
    with pytest.raises(ValueError, match="goal"):
        LoopRecipe(recipe_id="recipe_bad", name="Bad", goal="")


def test_loop_session_keeps_reports_and_exports_jsonl(tmp_path):
    started_at = utc("2026-06-23T10:00:00")
    report = LoopReport(
        run=LoopRun(
            run_id="run_export",
            session_id="session_local",
            user_input="When does Project Phoenix launch?",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            started_at=started_at,
            completed_at=started_at,
            final_decision=LoopDecision.NOT_VERIFIED,
            final_answer="Project Phoenix launches in June 2026 [1].",
        )
    )

    session = LoopSession(session_id="session_local").add_report(report)
    artifact_path = session.write_jsonl(tmp_path / "session.jsonl")
    restored = LoopSession.from_jsonl(artifact_path.read_text(encoding="utf-8"))

    assert session.report_count == 1
    assert session.to_dict()["schema_version"] == "loop-session/v1"
    assert artifact_path.exists()
    assert restored == session
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["run"]["run_id"] == (
        "run_export"
    )


def test_loop_session_rejects_cross_session_reports():
    report = LoopReport(
        run=LoopRun(
            run_id="run_other",
            session_id="other_session",
            user_input="What happened?",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
        )
    )

    with pytest.raises(ValueError, match="another session"):
        LoopSession(session_id="session_local").add_report(report)


def test_verification_result_rejects_unknown_outcome():
    with pytest.raises(ValueError):
        VerificationResult.from_dict(
            {
                "outcome": "suported",
                "reasons": ["typo_should_not_survive"],
            }
        )


def test_loop_report_rejects_unknown_schema_version():
    report = LoopReport(
        run=LoopRun(
            run_id="run_unsupported_schema",
            user_input="What happened?",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
        )
    ).to_dict()
    report["schema_version"] = "loop-report/v99"

    with pytest.raises(ValueError, match="Unsupported loop report schema"):
        LoopReport.from_dict(report)


def test_public_report_redacts_terminal_verification_reasons():
    secret_reason = "SECRET_VERIFIER_REASON"
    secret_verifier = "SECRET_GATEWAY_VERIFIER"
    report = LoopReport(
        run=LoopRun(
            run_id="run_terminal_verifier",
            user_input="Blocked prompt",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            steps=(
                LoopStep(
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.BLOCK,
                    name="Guardrail decision",
                    output_summary="blocked",
                    metadata={"guardrail_decision": "block"},
                    verification=VerificationResult(
                        outcome=VerificationOutcome.UNSUPPORTED,
                        reasons=(secret_reason,),
                        verifier=secret_verifier,
                        raw_response=secret_reason,
                        metadata={"debug": secret_reason},
                    ),
                ),
            ),
            final_decision=LoopDecision.BLOCK,
            final_answer="Blocked.",
            error_message=secret_reason,
        )
    )

    raw_json = json.dumps(report.to_dict())
    public_payload = report.to_public_dict()
    public_json = json.dumps(public_payload)
    verification = public_payload["run"]["steps"][0]["verification"]

    assert secret_verifier in raw_json
    assert secret_reason not in public_json
    assert secret_verifier not in public_json
    assert public_payload["public_redaction"]["reason"] == PUBLIC_REDACTION_REASON
    assert public_payload["run"]["error_message"] == PUBLIC_REDACTION_REASON
    assert verification["verifier"] is None
    assert verification["reasons"] == [PUBLIC_REDACTION_TEXT]
    assert verification["raw_response"] is None
    assert verification["metadata"]["redacted"] is True


def test_public_report_redacts_unmarked_terminal_verifier_identity():
    secret_verifier = "SECRET_OPERATOR_VERIFIER"
    report = LoopReport(
        run=LoopRun(
            run_id="run_terminal_without_guardrail_step",
            user_input="Sensitive prompt",
            context_provider="none",
            backend="openai-compatible",
            model_label="gateway-model",
            steps=(
                LoopStep(
                    phase=LoopPhase.VERIFY,
                    decision=LoopDecision.REFUSE,
                    name="LLM verifier",
                    output_summary="refused by verifier",
                    verification=VerificationResult(
                        outcome=VerificationOutcome.UNSUPPORTED,
                        reasons=("unsupported_claim",),
                        verifier=secret_verifier,
                    ),
                ),
            ),
            final_decision=LoopDecision.REFUSE,
            final_answer="I cannot verify that.",
        )
    )

    raw_json = json.dumps(report.to_dict())
    public_payload = report.to_public_dict()
    public_json = json.dumps(public_payload)
    verification = public_payload["run"]["steps"][0]["verification"]

    assert secret_verifier in raw_json
    assert secret_verifier not in public_json
    assert public_payload["public_redaction"]["applied"] is True
    assert public_payload["public_redaction"]["reason"] == PUBLIC_REDACTION_REASON
    assert verification["verifier"] is None


@pytest.mark.parametrize("field_name", ["allow_tool_calls", "allow_mock_supported"])
def test_loop_policy_rejects_string_false_booleans(field_name):
    payload = LoopPolicy().to_dict()
    payload[field_name] = "false"

    with pytest.raises(ValueError, match=f"{field_name} must be a JSON boolean"):
        LoopPolicy.from_dict(payload)


def test_loop_policy_constructor_rejects_non_boolean_guardrail_values():
    with pytest.raises(ValueError, match="allow_tool_calls must be a boolean"):
        LoopPolicy(allow_tool_calls="false")


def test_guardrail_decision_round_trips_human_review_request():
    review = HumanReviewRequest(
        request_id="review_guardrail",
        reason="tool output needs approval",
        instructions="Inspect untrusted tool output before continuing.",
        created_at=utc("2026-06-23T12:00:00"),
    )
    decision = GuardrailDecision(
        decision=LoopDecision.REQUIRES_REVIEW,
        reason="untrusted_tool_output",
        human_review=review,
        metadata={"source": "middleware"},
    )

    restored = GuardrailDecision.from_dict(decision.to_dict())

    assert restored == decision
    assert restored.can_continue is False


def test_guardrail_decision_rejects_non_guardrail_outcome():
    with pytest.raises(ValueError, match="supported is not a guardrail decision"):
        GuardrailDecision(decision=LoopDecision.SUPPORTED)


def test_loop_run_add_step_and_complete_are_immutable():
    run = LoopRun(
        run_id="run_1",
        user_input="What changed?",
        context_provider="document",
        backend="mock",
        model_label="MockLLM (explicit demo)",
    )
    step = LoopStep(
        step_id="step_1",
        phase=LoopPhase.DRAFT,
        decision=LoopDecision.NOT_VERIFIED,
        output_summary="drafted answer",
    )

    with_step = run.with_step(step)
    completed = with_step.complete(
        final_decision=LoopDecision.NOT_VERIFIED,
        final_answer="Draft answer [1].",
        metadata={"retry_count": 0},
    )

    assert run.steps == ()
    assert with_step.steps == (step,)
    assert completed.final_decision == LoopDecision.NOT_VERIFIED
    assert completed.final_answer == "Draft answer [1]."
    assert completed.metadata == {"retry_count": 0}
    assert completed.completed_at is not None


def test_human_review_request_and_policy_keep_safety_boundaries_explicit():
    review = HumanReviewRequest(
        request_id="review_1",
        reason="tool call requires approval",
        instructions="Approve only after checking destination and payload.",
        requested_by_step_id="step_tool",
        created_at=utc("2026-06-23T11:00:00"),
        metadata={"tool": "filesystem_write"},
    )
    step = LoopStep(
        step_id="step_tool",
        phase=LoopPhase.MECHANICAL_CHECK,
        decision=LoopDecision.REQUIRES_REVIEW,
        human_review=review,
    )
    policy = LoopPolicy()

    restored_step = LoopStep.from_dict(step.to_dict())

    assert policy.allow_tool_calls is False
    assert policy.require_human_review_for_tools is True
    assert policy.require_verifier_for_supported is True
    assert restored_step.human_review == review
    assert restored_step.decision == LoopDecision.REQUIRES_REVIEW
