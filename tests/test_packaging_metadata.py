import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def requirement_lines(path: str) -> tuple[str, ...]:
    lines = []
    for raw_line in (PROJECT_ROOT / path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-r "):
            continue
        lines.append(line)
    return tuple(lines)


def pyproject() -> dict:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())


def uv_lock() -> dict:
    return tomllib.loads((PROJECT_ROOT / "uv.lock").read_text())


def locked_dependency_strings(entries: list[dict]) -> tuple[str, ...]:
    return tuple(f"{entry['name']}{entry.get('specifier', '')}" for entry in entries)


def locked_project_package() -> dict:
    for package in uv_lock()["package"]:
        if package["name"] == "ai-loop-engine":
            return package
    raise AssertionError("ai-loop-engine package missing from uv.lock")


def test_runtime_requirements_match_pyproject_dependencies():
    assert tuple(pyproject()["project"]["dependencies"]) == requirement_lines(
        "requirements.txt"
    )


def test_dev_requirements_match_pyproject_dev_group():
    assert tuple(pyproject()["dependency-groups"]["dev"]) == requirement_lines(
        "requirements-dev.txt"
    )


def test_console_scripts_are_declared():
    scripts = pyproject()["project"]["scripts"]
    assert scripts["ai-loop-engine"] == "src.app:main"
    assert scripts["ai-loop-eval"] == "src.loop_eval:main"
    assert scripts["ai-loop-ollama-eval"] == "src.ollama_model_eval:main"


def test_removed_model_stack_is_not_direct_dependency():
    removed_direct_dependencies = {
        "accelerate",
        "huggingface-hub",
        "langchain-huggingface",
        "sentence-transformers",
        "torch",
        "transformers",
    }
    direct_dependency_names = {
        dependency.split("==", 1)[0]
        for dependency in pyproject()["project"]["dependencies"]
    }

    assert direct_dependency_names.isdisjoint(removed_direct_dependencies)


def test_uv_lock_matches_project_metadata():
    project = pyproject()
    locked_project = locked_project_package()

    assert uv_lock()["requires-python"].replace(" ", "") == project["project"][
        "requires-python"
    ].replace(" ", "")
    assert sorted(
        locked_dependency_strings(locked_project["metadata"]["requires-dist"])
    ) == sorted(project["project"]["dependencies"])
    assert sorted(
        locked_dependency_strings(
            locked_project["metadata"]["requires-dev"]["dev"]
        )
    ) == sorted(project["dependency-groups"]["dev"])
