from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.loop_engine import (
    PUBLIC_REDACTION_REASON,
    PUBLIC_REDACTION_TEXT,
    TERMINAL_PUBLIC_REDACTION_DECISION_VALUES,
    LoopReport,
)


def require_public_bool(public: bool) -> None:
    if type(public) is not bool:
        raise ValueError("public must be a boolean")


def report_payload(report: LoopReport, *, public: bool = True) -> Dict[str, Any]:
    require_public_bool(public)
    if not public:
        return report.to_dict()

    payload = report.to_public_dict()
    if _is_terminal_public_redaction_payload(payload):
        return _redact_terminal_payload(payload)
    return payload


def run_with_session_fallback(
    run: Mapping[str, Any],
    session_id: Optional[str],
) -> Mapping[str, Any]:
    if run.get("session_id") or not session_id:
        return run
    run_with_session = dict(run)
    run_with_session["session_id"] = session_id
    return run_with_session


def _is_terminal_public_redaction_payload(payload: Mapping[str, Any]) -> bool:
    final_decision = payload.get("run", {}).get("final_decision")
    return final_decision in TERMINAL_PUBLIC_REDACTION_DECISION_VALUES


def _redact_terminal_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    redacted = {
        "schema_version": payload["schema_version"],
        "run": dict(payload["run"]),
        "public_redaction": {
            "applied": True,
            "reason": PUBLIC_REDACTION_REASON,
        },
    }
    run = redacted["run"]
    run["user_input"] = PUBLIC_REDACTION_TEXT
    run["final_answer"] = PUBLIC_REDACTION_TEXT
    run["error_message"] = PUBLIC_REDACTION_REASON
    run["metadata"] = _redacted_metadata()
    if run.get("policy"):
        policy = dict(run["policy"])
        policy["metadata"] = _redacted_metadata()
        run["policy"] = policy
    run["steps"] = [_redact_terminal_step(step) for step in run.get("steps", ())]
    return redacted


def _redact_terminal_step(step: Mapping[str, Any]) -> Dict[str, Any]:
    redacted = dict(step)
    redacted["input_summary"] = (
        PUBLIC_REDACTION_TEXT
        if redacted.get("input_summary") is not None
        else None
    )
    redacted["output_summary"] = (
        PUBLIC_REDACTION_TEXT
        if redacted.get("output_summary") is not None
        else None
    )
    redacted["error_message"] = (
        PUBLIC_REDACTION_REASON
        if redacted.get("error_message") is not None
        else None
    )
    redacted["metadata"] = _redacted_metadata()
    if redacted.get("verification") is not None:
        verification = dict(redacted["verification"])
        verification["reasons"] = [PUBLIC_REDACTION_TEXT]
        verification["verifier"] = None
        verification["raw_response"] = None
        verification["metadata"] = _redacted_metadata()
        redacted["verification"] = verification
    if redacted.get("human_review") is not None:
        redacted["human_review"] = None
    return redacted


def _redacted_metadata() -> Dict[str, Any]:
    return {
        "redacted": True,
        "redaction_reason": PUBLIC_REDACTION_REASON,
    }
