import io
import json
from datetime import datetime, timezone

import pytest

from src.loop_engine import (
    LoopDecision,
    LoopPhase,
    LoopReport,
    LoopRun,
    LoopSession,
    LoopStep,
)
from src.loop_export import main


def utc(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def sample_report(run_id: str, *, secret: str | None = None) -> LoopReport:
    started_at = utc("2026-06-24T05:00:00")
    if secret:
        return LoopReport(
            run=LoopRun(
                run_id=run_id,
                session_id="session_cli",
                user_input=secret,
                context_provider="document",
                backend="mock",
                model_label="MockLLM (explicit demo)",
                started_at=started_at,
                completed_at=started_at,
                steps=(
                    LoopStep(
                        step_id="step_block",
                        phase=LoopPhase.ERROR,
                        decision=LoopDecision.BLOCK,
                        started_at=started_at,
                        ended_at=started_at,
                        output_summary=secret,
                        metadata={"secret": secret},
                    ),
                ),
                final_decision=LoopDecision.BLOCK,
                final_answer=secret,
                error_message=secret,
                metadata={"secret": secret},
            )
        )
    return LoopReport(
        run=LoopRun(
            run_id=run_id,
            session_id="session_cli",
            user_input="When does Project Phoenix launch?",
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            started_at=started_at,
            completed_at=started_at,
            steps=(
                LoopStep(
                    step_id="step_final",
                    phase=LoopPhase.FINAL,
                    decision=LoopDecision.NOT_VERIFIED,
                    started_at=started_at,
                    ended_at=started_at,
                    output_summary="Project Phoenix launches in June 2026 [1].",
                ),
            ),
            final_decision=LoopDecision.NOT_VERIFIED,
            final_answer="Project Phoenix launches in June 2026 [1].",
        )
    )


def sample_report_without_session(run_id: str) -> LoopReport:
    started_at = utc("2026-06-24T05:15:00")
    return LoopReport(
        run=LoopRun(
            run_id=run_id,
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


def write_jsonl(tmp_path, *reports: LoopReport):
    session = LoopSession(session_id="session_cli")
    for report in reports:
        session = session.add_report(report)
    path = tmp_path / "loop-session.jsonl"
    path.write_text(session.to_jsonl(), encoding="utf-8")
    return path


def test_loop_export_writes_langgraph_session_to_stdout(tmp_path):
    input_path = write_jsonl(tmp_path, sample_report("run_one"), sample_report("run_two"))
    output = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "langgraph-manifest",
            "--input",
            str(input_path),
        ],
        output_stream=output,
    )

    payload = json.loads(output.getvalue())
    assert exit_code == 0
    assert payload["adapter_name"] == "langgraph_manifest"
    assert payload["thread_id"] == "session_cli"
    assert payload["manifest_count"] == 2
    assert [manifest["source_jsonl_line"] for manifest in payload["manifests"]] == [
        1,
        2,
    ]


def test_loop_export_writes_one_openai_report_to_file(tmp_path):
    input_path = write_jsonl(tmp_path, sample_report("run_one"), sample_report("run_two"))
    output_path = tmp_path / "trace.json"

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
            "--report-index",
            "2",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["adapter_name"] == "openai_trace"
    assert payload["trace"]["trace_id"] == "trace_run_two"
    assert "traces" not in payload


def test_loop_export_preserves_selected_line_for_single_langgraph_manifest(tmp_path):
    input_path = write_jsonl(tmp_path, sample_report("run_one"), sample_report("run_two"))
    output = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "langgraph-manifest",
            "--input",
            str(input_path),
            "--report-index",
            "2",
        ],
        output_stream=output,
    )

    payload = json.loads(output.getvalue())
    manifest = payload["manifest"]
    assert exit_code == 0
    assert manifest["run_id"] == "run_two"
    assert manifest["source_jsonl_line"] == 2
    assert manifest["checkpoints"][0]["source_jsonl_line"] == 2


def test_loop_export_uses_supplied_session_id_for_langgraph_manifest(tmp_path):
    input_path = write_jsonl(tmp_path, sample_report_without_session("run_no_session"))
    output = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "langgraph-manifest",
            "--input",
            str(input_path),
            "--session-id",
            "session_supplied",
        ],
        output_stream=output,
    )

    payload = json.loads(output.getvalue())
    assert exit_code == 0
    assert payload["thread_id"] == "session_supplied"
    assert payload["manifests"][0]["thread_id"] == "session_supplied"


def test_loop_export_uses_supplied_session_id_for_single_openai_trace(tmp_path):
    input_path = write_jsonl(tmp_path, sample_report_without_session("run_no_session"))
    output = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
            "--session-id",
            "session_supplied",
            "--report-index",
            "1",
        ],
        output_stream=output,
    )

    payload = json.loads(output.getvalue())
    assert exit_code == 0
    assert payload["trace"]["group_id"] == "session_supplied"


def test_loop_export_public_default_redacts_terminal_content(tmp_path):
    secret = "cli secret should not leak"
    input_path = write_jsonl(tmp_path, sample_report("run_secret", secret=secret))
    output = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "langgraph-manifest",
            "--input",
            str(input_path),
        ],
        output_stream=output,
    )

    assert exit_code == 0
    assert secret not in output.getvalue()


def test_loop_export_raw_requires_explicit_flag(tmp_path):
    secret = "cli raw secret"
    input_path = write_jsonl(tmp_path, sample_report("run_secret", secret=secret))
    output = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "langgraph-manifest",
            "--input",
            str(input_path),
            "--raw",
        ],
        output_stream=output,
    )

    assert exit_code == 0
    assert secret in output.getvalue()


def test_loop_export_rejects_empty_jsonl(tmp_path):
    input_path = tmp_path / "empty.jsonl"
    input_path.write_text("", encoding="utf-8")
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "No loop reports found" in errors.getvalue()


def test_loop_export_rejects_blank_lines_to_preserve_line_provenance(tmp_path):
    input_path = tmp_path / "blank-lines.jsonl"
    input_path.write_text(
        "\n" + json.dumps(sample_report("run_one").to_dict()) + "\n",
        encoding="utf-8",
    )
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "langgraph-manifest",
            "--input",
            str(input_path),
            "--report-index",
            "1",
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "Invalid loop report JSONL at line 1: blank line" in errors.getvalue()


def test_loop_export_rejects_malformed_report_jsonl_with_line_number(tmp_path):
    input_path = tmp_path / "malformed.jsonl"
    input_path.write_text('{"not":"a loop report"}\n', encoding="utf-8")
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "Invalid loop report JSONL at line 1" in errors.getvalue()
    assert "run" in errors.getvalue()


@pytest.mark.parametrize("payload", ['"not a report"', "[]", "123", "null"])
def test_loop_export_rejects_non_object_jsonl_with_line_number(tmp_path, payload):
    input_path = tmp_path / "non-object.jsonl"
    input_path.write_text(payload + "\n", encoding="utf-8")
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "Invalid loop report JSONL at line 1" in errors.getvalue()
    assert "expected object" in errors.getvalue()


def test_loop_export_rejects_invalid_json_with_line_number(tmp_path):
    input_path = tmp_path / "invalid.jsonl"
    input_path.write_text(
        json.dumps(sample_report("run_one").to_dict()) + "\n{bad json}\n",
        encoding="utf-8",
    )
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "Invalid loop report JSONL at line 2" in errors.getvalue()


def test_loop_export_rejects_out_of_range_report_index(tmp_path):
    input_path = write_jsonl(tmp_path, sample_report("run_one"))
    errors = io.StringIO()

    exit_code = main(
        [
            "--adapter",
            "openai-trace",
            "--input",
            str(input_path),
            "--report-index",
            "2",
        ],
        error_stream=errors,
    )

    assert exit_code == 2
    assert "--report-index must be between 1 and 1" in errors.getvalue()
