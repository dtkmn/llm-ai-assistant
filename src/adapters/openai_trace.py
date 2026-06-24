from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.loop_engine import (
    PUBLIC_REDACTION_TEXT,
    TERMINAL_GUARDRAIL_DECISION_VALUES,
    LoopReport,
    LoopSession,
)


ADAPTER_NAME = "openai_trace"
ADAPTER_SCHEMA_VERSION = "openai-trace-export/v1"
WORKFLOW_NAME = "AI Loop Engine"
GENERATION_PHASES = frozenset({"draft"})
GUARDRAIL_PHASES = frozenset({"verify"})


class OpenAITraceAdapter:
    adapter_name = ADAPTER_NAME
    adapter_schema_version = ADAPTER_SCHEMA_VERSION

    def export_report(self, report: LoopReport, *, public: bool = True) -> Dict[str, Any]:
        _require_public_bool(public)
        report_payload = (
            _public_report_payload(report) if public else report.to_dict()
        )
        run = report_payload["run"]
        trace_id = _trace_id(run["run_id"])
        return {
            "adapter_name": self.adapter_name,
            "adapter_schema_version": self.adapter_schema_version,
            "source_schema_version": report_payload["schema_version"],
            "public": public,
            "trace": _trace_from_run(run, trace_id=trace_id, public=public),
            "source_report": report_payload,
        }

    def export_session(
        self, session: LoopSession, *, public: bool = True
    ) -> Dict[str, Any]:
        _require_public_bool(public)
        report_exports = [
            self.export_report(report, public=public) for report in session.reports
        ]
        return {
            "adapter_name": self.adapter_name,
            "adapter_schema_version": self.adapter_schema_version,
            "source_schema_version": session.schema_version,
            "public": public,
            "session_id": session.session_id,
            "trace_count": len(report_exports),
            "traces": [report_export["trace"] for report_export in report_exports],
        }


def export_report(report: LoopReport, *, public: bool = True) -> Dict[str, Any]:
    return OpenAITraceAdapter().export_report(report, public=public)


def export_session(session: LoopSession, *, public: bool = True) -> Dict[str, Any]:
    return OpenAITraceAdapter().export_session(session, public=public)


def _require_public_bool(public: bool) -> None:
    if type(public) is not bool:
        raise ValueError("public must be a boolean")


def _public_report_payload(report: LoopReport) -> Dict[str, Any]:
    payload = report.to_public_dict()
    if _is_terminal_guardrail_payload(payload):
        return _redact_terminal_payload(payload)
    return payload


def _is_terminal_guardrail_payload(payload: Mapping[str, Any]) -> bool:
    final_decision = payload.get("run", {}).get("final_decision")
    return final_decision in TERMINAL_GUARDRAIL_DECISION_VALUES


def _redact_terminal_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    redacted = {
        "schema_version": payload["schema_version"],
        "run": dict(payload["run"]),
        "public_redaction": {
            "applied": True,
            "reason": "terminal_guardrail_decision",
        },
    }
    run = redacted["run"]
    run["user_input"] = PUBLIC_REDACTION_TEXT
    run["final_answer"] = PUBLIC_REDACTION_TEXT
    run["error_message"] = "terminal_guardrail_decision"
    run["metadata"] = {
        "redacted": True,
        "redaction_reason": "terminal_guardrail_decision",
    }
    if run.get("policy"):
        policy = dict(run["policy"])
        policy["metadata"] = {
            "redacted": True,
            "redaction_reason": "terminal_guardrail_decision",
        }
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
        "terminal_guardrail_decision"
        if redacted.get("error_message") is not None
        else None
    )
    redacted["metadata"] = {
        "redacted": True,
        "redaction_reason": "terminal_guardrail_decision",
    }
    if redacted.get("verification") is not None:
        verification = dict(redacted["verification"])
        verification["reasons"] = [PUBLIC_REDACTION_TEXT]
        verification["verifier"] = None
        verification["raw_response"] = None
        verification["metadata"] = {
            "redacted": True,
            "redaction_reason": "terminal_guardrail_decision",
        }
        redacted["verification"] = verification
    if redacted.get("human_review") is not None:
        redacted["human_review"] = None
    return redacted


def _trace_from_run(
    run: Mapping[str, Any], *, trace_id: str, public: bool
) -> Dict[str, Any]:
    metadata = {
        "ai_loop_engine_run_id": run["run_id"],
        "ai_loop_engine_schema": "loop-report/v1",
        "context_provider": run["context_provider"],
        "backend": run["backend"],
        "model_label": run["model_label"],
        "final_decision": run["final_decision"],
        "public": public,
    }
    if run.get("metadata"):
        metadata["loop_metadata"] = run["metadata"]

    return {
        "id": trace_id,
        "trace_id": trace_id,
        "workflow_name": WORKFLOW_NAME,
        "group_id": run.get("session_id"),
        "started_at": run.get("started_at"),
        "ended_at": run.get("completed_at"),
        "metadata": metadata,
        "spans": [
            _span_from_step(step, trace_id=trace_id)
            for step in run.get("steps", ())
        ],
    }


def _span_from_step(step: Mapping[str, Any], *, trace_id: str) -> Dict[str, Any]:
    phase = step["phase"]
    span_id = _span_id(step["step_id"])
    span_data = {
        "type": _span_type_for_phase(phase),
        "name": step.get("name") or phase,
        "phase": phase,
        "decision": step.get("decision"),
        "input": step.get("input_summary"),
        "output": step.get("output_summary"),
        "backend": step.get("backend"),
        "model": step.get("model_label"),
        "retry_count": step.get("retry_count", 0),
        "error": step.get("error_message"),
        "metadata": step.get("metadata") or {},
    }
    if step.get("verification"):
        span_data["verification"] = step["verification"]
    if step.get("human_review"):
        span_data["human_review"] = step["human_review"]

    return {
        "id": span_id,
        "span_id": span_id,
        "trace_id": trace_id,
        "parent_id": None,
        "started_at": step.get("started_at"),
        "ended_at": step.get("ended_at"),
        "duration_ms": step.get("duration_ms"),
        "status": _status_for_step(step),
        "span_data": span_data,
    }


def _span_type_for_phase(phase: str) -> str:
    if phase in GENERATION_PHASES:
        return "generation"
    if phase in GUARDRAIL_PHASES:
        return "guardrail"
    return "custom"


def _status_for_step(step: Mapping[str, Any]) -> str:
    decision = step.get("decision")
    if step.get("error_message") or decision in {"block", "error"}:
        return "error"
    if decision in {"refuse", "requires_review", "retry"}:
        return decision
    return "ok"


def _trace_id(run_id: str) -> str:
    return _prefixed_id("trace", run_id)


def _span_id(step_id: str) -> str:
    return _prefixed_id("span", step_id)


def _prefixed_id(prefix: str, value: Optional[str]) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in str(value or "unknown")
    )
    return cleaned if cleaned.startswith(f"{prefix}_") else f"{prefix}_{cleaned}"
