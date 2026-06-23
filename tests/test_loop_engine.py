import json
from datetime import datetime, timezone

import pytest

from src.loop_engine import (
    GuardrailDecision,
    HumanReviewRequest,
    LoopDecision,
    LoopPhase,
    LoopPolicy,
    LoopReport,
    LoopRun,
    LoopStep,
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
            model_label="MockLLM (fallback)",
        )
    ).to_dict()
    report["schema_version"] = "loop-report/v99"

    with pytest.raises(ValueError, match="Unsupported loop report schema"):
        LoopReport.from_dict(report)


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
        model_label="MockLLM (fallback)",
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
