from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


DEFAULT_ENV_FILES = (".env", ".env.local")
DISABLE_ENV_FILE_VAR = "AI_LOOP_DISABLE_ENV_FILE"


def _parse_env_line(line: str, *, path: Path, line_number: int) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        raise RuntimeError(f"Invalid env file line in {path} at {line_number}.")
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key.replace("_", "A").isalnum() or not (
        key[0].isalpha() or key[0] == "_"
    ):
        raise RuntimeError(f"Invalid env var name in {path} at {line_number}.")
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_local_env_files(
    root: str | Path | None = None,
    filenames: Iterable[str] = DEFAULT_ENV_FILES,
) -> tuple[Path, ...]:
    """Load local env files without overriding variables set by the shell."""

    disabled = os.getenv(DISABLE_ENV_FILE_VAR, "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return ()

    base_dir = Path(root) if root is not None else Path.cwd()
    protected_keys = set(os.environ)
    loaded_files = []

    for filename in filenames:
        path = base_dir / filename
        if not path.exists():
            continue
        if not path.is_file():
            raise RuntimeError(f"Env path is not a file: {path}")
        loaded_files.append(path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            parsed = _parse_env_line(line, path=path, line_number=line_number)
            if parsed is None:
                continue
            key, value = parsed
            if key in protected_keys:
                continue
            os.environ[key] = value

    return tuple(loaded_files)
