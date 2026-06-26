import os

import pytest

from src.env_file import DISABLE_ENV_FILE_VAR, load_local_env_files


def test_load_local_env_files_reads_env_and_local_override(tmp_path, monkeypatch):
    monkeypatch.delenv(DISABLE_ENV_FILE_VAR, raising=False)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDINGS_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "# local defaults",
                "LLM_BACKEND=ollama",
                "LLM_MODEL=nemotron-3-nano:4b",
                "EMBEDDINGS_MODEL=embeddinggemma",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text(
        "LLM_MODEL=local-override:4b\n",
        encoding="utf-8",
    )

    loaded = load_local_env_files(tmp_path)

    assert loaded == (tmp_path / ".env", tmp_path / ".env.local")
    assert os.environ["LLM_BACKEND"] == "ollama"
    assert os.environ["LLM_MODEL"] == "local-override:4b"
    assert os.environ["EMBEDDINGS_MODEL"] == "embeddinggemma"


def test_load_local_env_files_respects_shell_environment(tmp_path, monkeypatch):
    monkeypatch.delenv(DISABLE_ENV_FILE_VAR, raising=False)
    monkeypatch.setenv("LLM_MODEL", "shell-model:8b")
    (tmp_path / ".env").write_text(
        "LLM_MODEL=file-model:4b\n",
        encoding="utf-8",
    )

    load_local_env_files(tmp_path)

    assert os.environ["LLM_MODEL"] == "shell-model:8b"


def test_load_local_env_files_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv(DISABLE_ENV_FILE_VAR, "true")
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    (tmp_path / ".env").write_text(
        "LLM_BACKEND=ollama\n",
        encoding="utf-8",
    )

    loaded = load_local_env_files(tmp_path)

    assert loaded == ()
    assert "LLM_BACKEND" not in os.environ


def test_load_local_env_files_rejects_malformed_entries(tmp_path, monkeypatch):
    monkeypatch.delenv(DISABLE_ENV_FILE_VAR, raising=False)
    (tmp_path / ".env").write_text(
        "not a valid assignment\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Invalid env file line"):
        load_local_env_files(tmp_path)
