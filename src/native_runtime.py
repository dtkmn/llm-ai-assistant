"""Native runtime bootstrap helpers.

Keep this module free of NumPy, FAISS, Gradio, or other native imports. It
exists so entrypoints can set conservative process defaults before native
libraries initialize their thread pools.
"""

from __future__ import annotations

import os
from typing import Mapping


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
