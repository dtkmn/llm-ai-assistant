"""Compatibility module for the historical ``src.DocumentQA`` import path.

The runtime implementation now lives in ``src.ai_loop_runtime``. This module
aliases itself to that runtime module so legacy imports and monkeypatch targets
continue to operate on the real implementation surface.
"""

import sys

try:
    from . import ai_loop_runtime as _runtime
except ImportError:
    import ai_loop_runtime as _runtime


sys.modules[__name__] = _runtime
