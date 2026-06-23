import argparse
import ipaddress
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from src.DocumentQA import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_BASE_URL_ENV_VAR,
    SELF_CHECK_REFUSAL_ANSWER,
    DocumentQA,
    QueryResult,
)
from src.golden_eval import (
    GOLDEN_DOCUMENT_TEXT,
    GOLDEN_EVAL_CASES,
    GoldenEvalCase,
    GoldenEvalEmbeddings,
)


@dataclass(frozen=True)
class ModelEvalCaseResult:
    case_id: str
    question: str
    passed: bool
    answer: str
    expected_terms: List[str]
    expect_refusal: bool
    self_check_outcome: Optional[str]
    self_check_reasons: List[str]
    citation_count: int
    error_message: Optional[str] = None


@dataclass(frozen=True)
class ModelEvalResult:
    model: str
    passed: bool
    passed_cases: int
    total_cases: int
    case_results: List[ModelEvalCaseResult]
    initialization_error: Optional[str] = None


QaFactory = Callable[[str, str, int], DocumentQA]
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _is_loopback_host(host: str) -> bool:
    normalized_host = host.strip().lower()
    if normalized_host == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized_host).is_loopback
    except ValueError:
        return False


def normalize_local_ollama_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Ollama base URL must use http or https.")
    if parsed.username or parsed.password:
        raise ValueError("Ollama base URL must not include credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("Ollama base URL must not include query or fragment data.")
    if parsed.path not in {"", "/"}:
        raise ValueError("Ollama base URL must not include a path.")
    if not parsed.hostname or not _is_loopback_host(parsed.hostname):
        raise ValueError(
            "Live Ollama eval only accepts loopback base URLs such as "
            "http://localhost:11434 or http://127.0.0.1:11434."
        )

    # Accessing port forces urllib to reject invalid port values before use.
    port = parsed.port
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    return urllib.parse.urlunsplit((parsed.scheme, netloc, "", "", ""))


def build_ollama_qa(model: str, base_url: str, timeout: int) -> DocumentQA:
    qa = DocumentQA(fast_mode=True, llm_backend="ollama", hf_token=None)
    qa.ollama_model = model
    qa.ollama_base_url = normalize_local_ollama_base_url(base_url)
    qa.ollama_timeout = timeout
    qa.embeddings = GoldenEvalEmbeddings()
    return qa


def score_case(case: GoldenEvalCase, result: QueryResult) -> ModelEvalCaseResult:
    self_check = result.trace.self_check
    self_check_outcome = self_check.outcome if self_check else None
    self_check_reasons = list(self_check.reasons) if self_check else []
    citation_count = len(result.trace.citations)
    valid_citation_ids = {
        str(citation.citation_id) for citation in result.trace.citations
    }
    inline_citation_ids = re.findall(r"\[(\d+)\]", result.answer)
    invalid_inline_citations = [
        citation_id
        for citation_id in inline_citation_ids
        if citation_id not in valid_citation_ids
    ]
    answer_lower = result.answer.lower()

    if case.expect_refusal:
        passed = (
            result.answer == SELF_CHECK_REFUSAL_ANSWER
            and self_check_outcome != "supported"
        )
    else:
        expected_terms_present = all(
            expected_term.lower() in answer_lower
            for expected_term in case.expected_terms
        )
        inline_citation_present = any(
            f"[{citation.citation_id}]" in result.answer
            for citation in result.trace.citations
        )
        passed = (
            expected_terms_present
            and inline_citation_present
            and not invalid_inline_citations
            and citation_count > 0
            and self_check_outcome == "supported"
        )

    if invalid_inline_citations and "invalid_inline_citation" not in self_check_reasons:
        self_check_reasons.append("invalid_inline_citation")

    return ModelEvalCaseResult(
        case_id=case.case_id,
        question=case.question,
        passed=passed,
        answer=result.answer,
        expected_terms=list(case.expected_terms),
        expect_refusal=case.expect_refusal,
        self_check_outcome=self_check_outcome,
        self_check_reasons=self_check_reasons,
        citation_count=citation_count,
    )


def failed_case(case: GoldenEvalCase, error: Exception) -> ModelEvalCaseResult:
    return ModelEvalCaseResult(
        case_id=case.case_id,
        question=case.question,
        passed=False,
        answer="",
        expected_terms=list(case.expected_terms),
        expect_refusal=case.expect_refusal,
        self_check_outcome=None,
        self_check_reasons=[],
        citation_count=0,
        error_message=str(error),
    )


def evaluate_model(
    model: str,
    *,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout: int = 60,
    qa_factory: QaFactory = build_ollama_qa,
    cases: Sequence[GoldenEvalCase] = GOLDEN_EVAL_CASES,
    unload_after: bool = True,
) -> ModelEvalResult:
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            document_path = Path(tmp_dir) / "project_phoenix_brief.md"
            document_path.write_text(GOLDEN_DOCUMENT_TEXT, encoding="utf-8")
            try:
                qa = qa_factory(model, base_url, timeout)
                qa.process_document(str(document_path))
            except Exception as exc:
                return ModelEvalResult(
                    model=model,
                    passed=False,
                    passed_cases=0,
                    total_cases=len(cases),
                    case_results=[],
                    initialization_error=str(exc),
                )

            case_results = []
            for case in cases:
                try:
                    case_results.append(
                        score_case(case, qa.query_with_trace(case.question))
                    )
                except Exception as exc:
                    case_results.append(failed_case(case, exc))
    finally:
        if unload_after:
            unload_ollama_model(model, base_url=base_url, timeout=min(timeout, 15))

    passed_cases = sum(1 for result in case_results if result.passed)
    total_cases = len(case_results)
    return ModelEvalResult(
        model=model,
        passed=passed_cases == total_cases,
        passed_cases=passed_cases,
        total_cases=total_cases,
        case_results=case_results,
    )


def unload_ollama_model(
    model: str,
    *,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout: int = 15,
) -> None:
    try:
        safe_base_url = normalize_local_ollama_base_url(base_url)
        payload = {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }
        request = urllib.request.Request(
            f"{safe_base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with NO_PROXY_OPENER.open(request, timeout=timeout) as response:
            response.read()
    except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError):
        # Unloading is a best-effort memory hygiene step. The eval result should
        # report model quality, not fail because cleanup could not reach Ollama.
        return


def models_from_args(args) -> List[str]:
    if args.models:
        return args.models
    env_models = os.getenv("OLLAMA_EVAL_MODELS", "").strip()
    if env_models:
        return [model.strip() for model in env_models.split(",") if model.strip()]
    env_model = os.getenv("OLLAMA_MODEL", "").strip()
    if env_model:
        return [env_model]
    return [DEFAULT_OLLAMA_MODEL]


def format_text_results(results: Sequence[ModelEvalResult]) -> str:
    lines = []
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"{status} {result.model}: {result.passed_cases}/{result.total_cases} cases"
        )
        if result.initialization_error:
            lines.append(f"  initialization_error: {result.initialization_error}")
            continue
        for case_result in result.case_results:
            case_status = "PASS" if case_result.passed else "FAIL"
            lines.append(
                "  "
                f"{case_status} {case_result.case_id}: "
                f"self_check={case_result.self_check_outcome}, "
                f"citations={case_result.citation_count}"
            )
            if case_result.error_message:
                lines.append(f"    error: {case_result.error_message}")
            if not case_result.passed:
                answer = case_result.answer.replace("\n", " ").strip()
                lines.append(f"    answer: {answer}")
                if case_result.self_check_reasons:
                    lines.append(
                        "    reasons: " + ", ".join(case_result.self_check_reasons)
                    )
    return "\n".join(lines)


def results_to_json(results: Sequence[ModelEvalResult]) -> str:
    return json.dumps([asdict(result) for result in results], indent=2)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the golden document QA eval against one or more live Ollama models."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Ollama model tags to compare, for example "
            "`nemotron-3-nano:4b qwen3:8b`. Defaults to OLLAMA_EVAL_MODELS, "
            "OLLAMA_MODEL, or the app default."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv(OLLAMA_BASE_URL_ENV_VAR, DEFAULT_OLLAMA_BASE_URL),
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        help=(
            "Run only a selected golden case id. Can be repeated. "
            "Useful for low-memory smoke tests."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON results.")
    parser.add_argument(
        "--allow-multi-model",
        action="store_true",
        help=(
            "Allow evaluating multiple Ollama models in one command. Disabled "
            "by default because loading multiple local models can spike unified "
            "memory on Macs; prefer one model per command."
        ),
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help=(
            "Exit 0 for failing eval scores, useful when comparing models manually. "
            "Argument validation and memory-safety guard failures still return nonzero."
        ),
    )
    parser.add_argument(
        "--no-unload",
        action="store_true",
        help="Do not ask Ollama to unload each model after its eval run.",
    )
    return parser.parse_args(argv)


def selected_cases(case_ids: Optional[Sequence[str]]) -> Sequence[GoldenEvalCase]:
    if not case_ids:
        return GOLDEN_EVAL_CASES
    requested_case_ids = set(case_ids)
    cases = [case for case in GOLDEN_EVAL_CASES if case.case_id in requested_case_ids]
    missing_case_ids = sorted(requested_case_ids - {case.case_id for case in cases})
    if missing_case_ids:
        valid_case_ids = ", ".join(case.case_id for case in GOLDEN_EVAL_CASES)
        raise ValueError(
            f"Unknown golden case id(s): {', '.join(missing_case_ids)}. "
            f"Valid case ids: {valid_case_ids}"
        )
    return cases


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    qa_factory: QaFactory = build_ollama_qa,
    output_stream=None,
) -> int:
    args = parse_args(argv)
    output_stream = output_stream or sys.stdout
    try:
        cases = selected_cases(args.case_ids)
    except ValueError as exc:
        print(str(exc), file=output_stream)
        return 2
    try:
        base_url = normalize_local_ollama_base_url(args.base_url)
    except ValueError as exc:
        print(str(exc), file=output_stream)
        return 2
    models = models_from_args(args)
    if len(models) > 1 and not args.allow_multi_model:
        print(
            "Multiple Ollama models in one eval are disabled by default to avoid "
            "local memory pressure. Run one model per command, or pass "
            "--allow-multi-model only on a machine with enough free unified memory.",
            file=output_stream,
        )
        return 2
    results = [
        evaluate_model(
            model,
            base_url=base_url,
            timeout=args.timeout,
            qa_factory=qa_factory,
            cases=cases,
            unload_after=not args.no_unload,
        )
        for model in models
    ]
    if args.json:
        print(results_to_json(results), file=output_stream)
    else:
        print(format_text_results(results), file=output_stream)
    if args.no_fail:
        return 0
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
