"""Native runtime bootstrap helpers.

Keep this module free of NumPy, torch, FAISS, Gradio, or other native imports.
It exists so entrypoints can set conservative process defaults before native
libraries initialize their thread pools.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping


NATIVE_THREAD_DEFAULTS: Mapping[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def apply_native_runtime_defaults() -> None:
    """Apply conservative native-library defaults while respecting overrides."""

    for name, value in NATIVE_THREAD_DEFAULTS.items():
        os.environ.setdefault(name, value)


def apply_torch_thread_limit(
    torch_module: Any, *, logger: logging.Logger | None = None
) -> None:
    """Apply the torch thread limit after torch is imported."""

    try:
        torch_module.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))
    except Exception:
        if logger is not None:
            logger.warning("Unable to apply torch thread limit.", exc_info=True)
