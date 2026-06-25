import json
import os
import subprocess
import sys
from pathlib import Path

from src.native_runtime import apply_native_runtime_defaults


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TOKENIZERS_PARALLELISM",
}
EXPECTED_NATIVE_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def test_native_runtime_defaults_respect_existing_overrides(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("VECLIB_MAXIMUM_THREADS", raising=False)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

    apply_native_runtime_defaults()

    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["VECLIB_MAXIMUM_THREADS"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_retrieval_bootstraps_native_defaults_before_native_imports():
    env = os.environ.copy()
    for name in NATIVE_THREAD_ENV_VARS:
        env.pop(name, None)

    code = """
import builtins
import json
import os

real_import = builtins.__import__
native_modules = {"faiss", "numpy"}

def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
    root_name = name.split(".", 1)[0]
    if root_name in native_modules:
        print(json.dumps({
            "module": root_name,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        }, sort_keys=True))
        raise SystemExit(0)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = tracking_import
import src.retrieval
raise SystemExit("src.retrieval did not import a tracked native module")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(result.stdout)

    assert observed.pop("module") in {"faiss", "numpy"}
    assert observed == EXPECTED_NATIVE_DEFAULTS


def test_model_adapters_bootstrap_native_defaults_before_native_imports():
    env = os.environ.copy()
    for name in NATIVE_THREAD_ENV_VARS:
        env.pop(name, None)

    code = """
import builtins
import json
import os

real_import = builtins.__import__
native_modules = {"numpy", "torch"}

def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
    root_name = name.split(".", 1)[0]
    if root_name in native_modules:
        print(json.dumps({
            "module": root_name,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        }, sort_keys=True))
        raise SystemExit(0)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = tracking_import
import src.model_adapters
print(json.dumps({
    "module": None,
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
    "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(result.stdout)

    assert observed.pop("module") in {None, "numpy", "torch"}
    assert observed == EXPECTED_NATIVE_DEFAULTS


def test_document_ingestion_bootstraps_native_defaults_before_native_imports():
    env = os.environ.copy()
    for name in NATIVE_THREAD_ENV_VARS:
        env.pop(name, None)

    code = """
import builtins
import json
import os

real_import = builtins.__import__
native_modules = {"numpy", "torch"}

def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
    root_name = name.split(".", 1)[0]
    if root_name in native_modules:
        print(json.dumps({
            "module": root_name,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        }, sort_keys=True))
        raise SystemExit(0)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = tracking_import
import src.document_ingestion
print(json.dumps({
    "module": None,
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
    "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(result.stdout)

    assert observed.pop("module") in {None, "numpy", "torch"}
    assert observed == EXPECTED_NATIVE_DEFAULTS


def test_answer_loop_import_is_native_light():
    code = """
import json
import sys

import src.answer_loop

watched_modules = [
    "faiss",
    "numpy",
    "src.retrieval",
    "docx2txt",
    "langchain_text_splitters",
    "pypdf",
]
print(json.dumps({
    name: name in sys.modules
    for name in watched_modules
}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(result.stdout)

    assert observed == {
        "faiss": False,
        "numpy": False,
        "src.retrieval": False,
        "docx2txt": False,
        "langchain_text_splitters": False,
        "pypdf": False,
    }


def test_ai_loop_runtime_does_not_eagerly_import_document_ingestion_stack():
    code = """
import json
import sys

import src.ai_loop_runtime

qa = src.ai_loop_runtime.AILoopEngine(fast_mode=True, llm_backend="mock")
try:
    qa._validate_document("/definitely/missing.txt")
except ValueError:
    pass

watched_modules = [
    "faiss",
    "docx2txt",
    "langchain_text_splitters",
    "pypdf",
    "src.document_ingestion",
    "src.document_text",
    "src.retrieval",
]
print(json.dumps({
    name: name in sys.modules
    for name in watched_modules
}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(result.stdout)

    assert observed == {
        "faiss": False,
        "docx2txt": False,
        "langchain_text_splitters": False,
        "pypdf": False,
        "src.document_ingestion": False,
        "src.document_text": False,
        "src.retrieval": False,
    }
