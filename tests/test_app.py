import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src import app as web_app
from src.DocumentQA import (
    AnswerCitation,
    AnswerSelfCheck,
    AnswerTrace,
    DocumentQA,
    DocumentProcessingError,
    DocumentProcessingReport,
    DocumentQAStatus,
    QueryResult,
)
from src.loop_engine import (
    LoopDecision,
    LoopPhase,
    LoopReport,
    LoopRun,
    LoopStep,
    VerificationOutcome,
    VerificationResult,
)
from src.thread_store import ThreadStore
from src.web_contract import (
    answer_trace_dict,
    loop_summary_dict,
    loop_timeline_dict,
    query_response_dict,
    runtime_status_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TOKENIZERS_PARALLELISM",
}


class FakeQA:
    fast_mode = True
    loaded_model_id = None
    loaded_model_label = None
    active_llm_backend = "mock"
    llm_backend = "mock"
    current_document_name = None
    latest_processing_report = None

    def process_document(self, document_path, text_encoding=None):
        self.document_path = document_path
        self.text_encoding = text_encoding
        self.uploaded_text = Path(document_path).read_text(encoding="utf-8")
        self.current_document_name = Path(document_path).name
        active_backend = self.active_llm_backend or self.llm_backend
        active_model_label = (
            self.loaded_model_label
            or self.loaded_model_id
            or ("MockLLM (explicit demo)" if active_backend == "mock" else "unknown")
        )
        self.latest_processing_report = DocumentProcessingReport(
            attempted_document_name=self.current_document_name,
            active_document_name=self.current_document_name,
            success=True,
            phase="complete",
            file_extension=".txt",
            chunk_count=1,
            truncated=False,
            max_chunk_limit=2000,
            text_encoding_mode=text_encoding or "auto",
            backend=active_backend,
            model_label=active_model_label,
            error_message=None,
        )
        return self.status()

    def status(self):
        self.status_calls = getattr(self, "status_calls", 0) + 1
        active_backend = self.active_llm_backend or self.llm_backend
        active_model_label = (
            self.loaded_model_label
            or self.loaded_model_id
            or ("MockLLM (explicit demo)" if active_backend == "mock" else "unknown")
        )
        return DocumentQAStatus(
            profile_label="FAST" if self.fast_mode else "QUALITY",
            max_output_tokens=384 if self.fast_mode else 1024,
            configured_backend=self.llm_backend,
            active_backend=active_backend,
            active_model_label=active_model_label,
            loaded_model_id=self.loaded_model_id,
            loaded_model_label=self.loaded_model_label,
            embeddings_model="fake-embeddings",
            embeddings_device="cpu",
            device="cpu",
            document_name=self.current_document_name,
            ready_for_queries=bool(self.current_document_name),
            processing_report=self.latest_processing_report,
        )

    def query_with_trace(
        self,
        message,
        session_id="default",
        conversation_history=None,
        semantic_memory=None,
        semantic_memory_status="not_requested",
    ):
        self.last_query_session_id = session_id
        self.last_conversation_history = list(conversation_history or [])
        self.last_semantic_memory = list(semantic_memory or [])
        self.last_semantic_memory_status = semantic_memory_status
        active_backend = self.active_llm_backend or self.llm_backend
        active_model_label = (
            self.loaded_model_label
            or self.loaded_model_id
            or ("MockLLM (explicit demo)" if active_backend == "mock" else "unknown")
        )
        answer = "Project Phoenix is described in the uploaded document."
        steps = []
        if semantic_memory_status != "not_requested":
            steps.append(
                LoopStep(
                    phase=LoopPhase.CONTEXT_SELECT,
                    decision=LoopDecision.CONTINUE,
                    name="Retrieve thread memory",
                    output_summary=(
                        f"{len(self.last_semantic_memory)} semantic memories"
                        if self.last_semantic_memory
                        else semantic_memory_status
                    ),
                    metadata={
                        "semantic_memory_count": len(self.last_semantic_memory),
                        "semantic_memory_status": semantic_memory_status,
                    },
                )
            )
        steps.extend(
            [
                LoopStep(
                    phase=LoopPhase.RETRIEVE,
                    decision=LoopDecision.CONTINUE,
                    name="Retrieve prompt evidence",
                    output_summary="1 prompt chunks",
                    metadata={"retrieved_chunk_count": 1, "citation_ids": [1]},
                ),
                LoopStep(
                    phase=LoopPhase.MECHANICAL_CHECK,
                    decision=LoopDecision.NOT_VERIFIED,
                    name="Mechanical checks",
                    output_summary="mechanical_checks_passed",
                ),
            ]
        )
        loop_report = LoopReport(
            run=LoopRun(
                run_id="run_fake",
                session_id=session_id,
                user_input=message,
                context_provider="document",
                backend=active_backend,
                model_label=active_model_label,
                steps=tuple(steps),
                final_decision=LoopDecision.NOT_VERIFIED,
                final_answer=answer,
                metadata={
                    "semantic_memory_turns": len(self.last_semantic_memory),
                    "semantic_memory_status": semantic_memory_status,
                },
            )
        )
        return QueryResult(
            answer=answer,
            trace=AnswerTrace(
                question=message,
                document_name=self.current_document_name,
                backend=active_backend,
                model_label=active_model_label,
                retrieved_chunk_count=1,
                citations=[
                    AnswerCitation(
                        citation_id=1,
                        source_name=self.current_document_name or "demo.txt",
                        page=None,
                        chunk_index=0,
                        excerpt="Project Phoenix is a loop workbench.",
                    )
                ],
                self_check=AnswerSelfCheck(
                    outcome="not_verified",
                    reasons=[
                        "mechanical_checks_passed",
                        "verifier_unavailable_mock_backend",
                    ],
                    retry_attempted=False,
                ),
                model_thinking="I matched Project Phoenix against citation [1].",
            ),
            loop_report=loop_report,
        )

    def memory_embedding_model_label(self):
        return "fake-memory"

    def embed_memory_texts(self, texts):
        vectors = []
        for text in texts:
            lowered = str(text).lower()
            if "dynamic programming" in lowered or "algorithm" in lowered:
                vectors.append([1.0, 0.0])
            elif "phoenix" in lowered:
                vectors.append([0.8, 0.2])
            else:
                vectors.append([0.0, 1.0])
        self.embedded_memory_texts = list(getattr(self, "embedded_memory_texts", []))
        self.embedded_memory_texts.extend(str(text) for text in texts)
        return "fake-memory", vectors

    def clear_loop_session(self, session_id="default"):
        self.cleared_loop_session_id = session_id


def processed_report(
    *,
    document_name="good.txt",
    success=True,
    phase="complete",
    error_message=None,
):
    return DocumentProcessingReport(
        attempted_document_name=document_name,
        active_document_name=document_name,
        success=success,
        phase=phase,
        file_extension=".txt",
        chunk_count=1 if success else 0,
        truncated=False,
        max_chunk_limit=2000,
        text_encoding_mode="auto",
        backend="mock",
        model_label="MockLLM (explicit demo)",
        error_message=error_message,
    )


def test_app_bootstraps_native_defaults_before_fastapi_import():
    env = os.environ.copy()
    for name in NATIVE_THREAD_ENV_VARS:
        env.pop(name, None)

    code = """
import builtins
import json
import os

real_import = builtins.__import__

def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "fastapi" or name.startswith("fastapi."):
        print(json.dumps({
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        }, sort_keys=True))
        raise SystemExit(0)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = tracking_import
import src.app
raise SystemExit("src.app did not import fastapi")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(result.stdout) == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def test_app_loads_env_file_before_native_defaults_and_fastapi_import(tmp_path):
    env = os.environ.copy()
    env.pop("AI_LOOP_DISABLE_ENV_FILE", None)
    env.pop("FAST_MODE", None)
    env.pop("LLM_BACKEND", None)
    for name in NATIVE_THREAD_ENV_VARS:
        env.pop(name, None)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(REPO_ROOT)
    )
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OMP_NUM_THREADS=7",
                "LLM_BACKEND=mock",
                "FAST_MODE=true",
            ]
        ),
        encoding="utf-8",
    )

    code = """
import builtins
import json
import os

real_import = builtins.__import__

def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "fastapi" or name.startswith("fastapi."):
        print(json.dumps({
            "FAST_MODE": os.environ.get("FAST_MODE"),
            "LLM_BACKEND": os.environ.get("LLM_BACKEND"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        }, sort_keys=True))
        raise SystemExit(0)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = tracking_import
import src.app
raise SystemExit("src.app did not import fastapi")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(result.stdout) == {
        "FAST_MODE": "true",
        "LLM_BACKEND": "mock",
        "OMP_NUM_THREADS": "7",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def test_app_import_keeps_engine_lazy():
    assert web_app.qa_system is None


def test_app_main_defaults_to_loopback(monkeypatch):
    observed = {}

    def fake_run(app, *, host, port, log_level):
        observed.update(
            {"app": app, "host": host, "port": port, "log_level": log_level}
        )

    monkeypatch.delenv("WEB_HOST", raising=False)
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("WEB_PORT", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setattr(web_app.uvicorn, "run", fake_run)

    web_app.main()

    assert observed["host"] == "127.0.0.1"
    assert observed["port"] == 7860


def test_app_main_allows_explicit_non_loopback_host(monkeypatch):
    observed = {}

    def fake_run(app, *, host, port, log_level):
        observed.update({"host": host, "port": port, "log_level": log_level})

    monkeypatch.setenv("WEB_HOST", "0.0.0.0")
    monkeypatch.setenv("WEB_PORT", "8799")
    monkeypatch.setattr(web_app.uvicorn, "run", fake_run)

    web_app.main()

    assert observed["host"] == "0.0.0.0"
    assert observed["port"] == 8799


def test_static_frontend_is_served():
    client = TestClient(web_app.create_app(FakeQA()))

    response = client.get("/")
    script = client.get("/assets/app.js")
    styles = client.get("/assets/styles.css")

    assert response.status_code == 200
    assert "AI Loop Engine" in response.text
    assert "Threads" in response.text
    assert "Optional Context" in response.text
    assert "You can still run the loop without documents" in response.text
    assert "Model Thinking" in response.text
    assert "/assets/app.js" in response.text
    assert script.status_code == 200
    assert "Ask a question, or add context" in script.text
    assert "direct mode" in script.text
    assert "renderMessageThinking" in script.text
    assert "session_id" in script.text
    assert "switchThread" in script.text
    assert "result.trace?.model_thinking" in script.text
    assert "message-thinking" in script.text
    assert "innerHTML" not in script.text
    assert styles.status_code == 200
    assert ".thread-button" in styles.text
    assert ".message-thinking" in styles.text
    assert ".message-code-block" in styles.text


def test_static_frontend_renders_model_thinking_inside_assistant_message():
    if shutil.which("node") is None:
        pytest.skip("node is required for static frontend execution test")

    script = r"""
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

class ClassList {
  constructor(element) {
    this.element = element;
  }

  toggle(name, force) {
    const tokens = new Set(this.element.className.split(/\s+/).filter(Boolean));
    const enabled = force === undefined ? !tokens.has(name) : Boolean(force);
    if (enabled) {
      tokens.add(name);
    } else {
      tokens.delete(name);
    }
    this.element.className = Array.from(tokens).join(" ");
  }
}

class Element {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.className = "";
    this.dataset = {};
    this.disabled = false;
    this.files = [];
    this.listeners = {};
    this.open = false;
    this.scrollHeight = 0;
    this.scrollTop = 0;
    this.textContent = "";
    this.value = "";
    this.classList = new ClassList(this);
  }

  append(...nodes) {
    for (const node of nodes) {
      node.parentElement = this;
      this.children.push(node);
    }
    this.scrollHeight = this.children.length;
  }

  replaceChildren(...nodes) {
    this.children = [];
    this.append(...nodes);
  }

  addEventListener(type, listener) {
    this.listeners[type] = listener;
  }

  async dispatch(type) {
    const listener = this.listeners[type];
    if (!listener) {
      return;
    }
    await listener({ preventDefault() {} });
  }

  focus() {}

  querySelector(selector) {
    if (selector === "summary span") {
      const summary = findNode(this, (node) => node.tagName === "SUMMARY");
      return summary ? findNode(summary, (node) => node.tagName === "SPAN") : null;
    }
    return findNode(this, (node) => matches(node, selector));
  }
}

function matches(node, selector) {
  if (selector.startsWith(".")) {
    return node.className.split(/\s+/).includes(selector.slice(1));
  }
  return node.tagName === selector.toUpperCase();
}

function findNode(root, predicate) {
  for (const child of root.children) {
    if (predicate(child)) {
      return child;
    }
    const nested = findNode(child, predicate);
    if (nested) {
      return nested;
    }
  }
  return null;
}

function createMemoryStorage() {
  const values = new Map();
  return {
    getItem(key) {
      return values.has(key) ? values.get(key) : null;
    },
    setItem(key, value) {
      values.set(key, String(value));
    },
    removeItem(key) {
      values.delete(key);
    },
  };
}

function createDom() {
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: createMemoryStorage(),
  });
  const ids = [
    "backend-pill",
    "model-pill",
    "ready-pill",
    "upload-form",
    "upload-button",
    "upload-status",
    "document-file",
    "text-encoding",
    "new-thread",
    "thread-list",
    "active-thread-title",
    "refresh-status",
    "runtime-grid",
    "query-form",
    "query-input",
    "query-button",
    "clear-chat",
    "messages",
    "timeline",
    "final-decision",
    "thinking-panel",
    "thinking-state",
    "thinking-note",
    "thinking-content",
    "summary-json",
    "trace-json",
  ];
  const byId = Object.fromEntries(ids.map((id) => [id, new Element("div")]));
  byId["query-input"].value = "";
  const summary = new Element("summary");
  summary.append(new Element("span"));
  byId["thinking-panel"].append(summary);
  const tabButtons = [new Element("button"), new Element("button")];
  tabButtons[0].dataset.target = "summary-json";
  tabButtons[1].dataset.target = "trace-json";

  globalThis.document = {
    createElement(tagName) {
      return new Element(tagName);
    },
    querySelector(selector) {
      if (!selector.startsWith("#")) {
        return null;
      }
      return byId[selector.slice(1)] || null;
    },
    querySelectorAll(selector) {
      return selector === ".tab-button" ? tabButtons : [];
    },
  };
  return byId;
}

function jsonResponse(payload) {
  return {
    ok: true,
    status: 200,
    headers: { get: () => "application/json" },
    async json() {
      return payload;
    },
    async text() {
      return JSON.stringify(payload);
    },
  };
}

function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function createThreadPayload(id, messages = [], latest = null) {
  return {
    id,
    title: "New thread",
    messages,
    latest,
    message_count: messages.length,
    created_at: "2026-06-26T00:00:00.000Z",
    updated_at: "2026-06-26T00:00:00.000Z",
  };
}

async function importFreshApp() {
  const source = await readFile(process.env.APP_JS_PATH, "utf8");
  const encoded = Buffer.from(
    `${source}\n// frontend harness case ${Date.now()} ${Math.random()}`,
    "utf8",
  ).toString("base64");
  await import(`data:text/javascript;base64,${encoded}`);
}

async function runCase(modelThinking, answer = "Loop answer") {
  const dom = createDom();
  const queryBodies = [];
  const serverThreads = [createThreadPayload("thread_initial")];
  globalThis.fetch = async (url, options = {}) => {
    const method = String(options.method || "GET").toUpperCase();
    if (url === "/api/config") {
      return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
    }
    if (url === "/api/status") {
      return jsonResponse({
        backend: "ollama",
        model: "thinking-model",
        ready_for_queries: false,
        query_mode: "direct",
        chunk_count: 0,
      });
    }
    if (url === "/api/threads" && method === "GET") {
      return jsonResponse({ threads: serverThreads });
    }
    if (url === "/api/threads" && method === "POST") {
      const thread = createThreadPayload(`thread_created_${serverThreads.length}`);
      serverThreads.unshift(thread);
      return jsonResponse(thread);
    }
    if (url.startsWith("/api/threads/") && method === "GET") {
      const id = decodeURIComponent(url.slice("/api/threads/".length));
      const thread = serverThreads.find((item) => item.id === id);
      if (thread) {
        return jsonResponse(thread);
      }
    }
    if (url === "/api/query") {
      queryBodies.push(JSON.parse(options.body));
      return jsonResponse({
        answer,
        timeline: { rows: [], final_decision: "not_verified" },
        summary: {},
        trace: { model_thinking: modelThinking },
      });
    }
    if (url === "/api/chat/clear") {
      return jsonResponse({
        timeline: { rows: [], final_decision: null },
        summary: {},
        trace: {},
      });
    }
    throw new Error(`unexpected fetch ${url}`);
  };

  await importFreshApp();
  await new Promise((resolve) => setTimeout(resolve, 0));
  dom["query-input"].value = "What happened?";
  await dom["query-form"].dispatch("submit");
  return { dom, queryBodies };
}

const capturedCase = await runCase({
  available: true,
  redacted: false,
  label: "Model Thinking (unverified)",
  content: "Captured thinking from model.",
  note: "Model-emitted thinking is useful for debugging the loop.",
}, [
  "Here is `dp[0]` safely:",
  "```java",
  "public class FibonacciDP {",
  "  public static int fib(int n) { return n; }",
  "}",
  "```",
  "- uses memoization",
  "<img src=x onerror=alert(1)>",
].join("\n"));
const capturedDom = capturedCase.dom;
assert.equal(capturedCase.queryBodies.length, 1);
assert.ok(capturedCase.queryBodies[0].session_id.startsWith("thread_"));
const capturedAssistantMessage = findNode(
  capturedDom.messages,
  (node) => node.className === "message assistant",
);
assert.ok(capturedAssistantMessage, "assistant message should render");
const capturedMessageContent = findNode(
  capturedAssistantMessage,
  (node) => node.className === "message-content",
);
assert.ok(capturedMessageContent, "assistant message should include rich content");
const capturedCodeBlock = findNode(
  capturedMessageContent,
  (node) => node.className === "message-code-block",
);
assert.ok(capturedCodeBlock, "assistant markdown fence should become a code block");
const capturedLanguage = findNode(
  capturedCodeBlock,
  (node) => node.className === "message-code-language",
);
assert.equal(capturedLanguage.textContent, "java");
const capturedAnswerCode = findNode(capturedCodeBlock, (node) => node.tagName === "CODE");
assert.ok(capturedAnswerCode.textContent.includes("public class FibonacciDP"));
const capturedInlineCode = findNode(
  capturedMessageContent,
  (node) => node.className === "message-inline-code",
);
assert.equal(capturedInlineCode.textContent, "dp[0]");
const capturedList = findNode(capturedMessageContent, (node) => node.tagName === "UL");
assert.ok(capturedList, "assistant markdown list should render as a list");
const capturedImage = findNode(capturedMessageContent, (node) => node.tagName === "IMG");
assert.equal(capturedImage, null, "model HTML must stay inert text");
const capturedThinking = findNode(
  capturedDom.messages,
  (node) => node.className === "message-thinking",
);
assert.ok(capturedThinking, "assistant message should include thinking details");
const capturedPre = findNode(capturedThinking, (node) => node.tagName === "PRE");
assert.equal(capturedPre.textContent, "Captured thinking from model.");

const inlineFenceCase = await runCase({
  available: false,
  redacted: false,
  label: "Model Thinking (unverified)",
  content: null,
  note: "Model-emitted thinking is useful for debugging the loop.",
}, "Inline fence: ```java public class InlineFence {}``` done.");
const inlineAssistant = findNode(
  inlineFenceCase.dom.messages,
  (node) => node.className === "message assistant",
);
const inlineCodeBlock = findNode(
  inlineAssistant,
  (node) => node.className === "message-code-block",
);
assert.ok(inlineCodeBlock, "inline fenced code should become a code block");
const inlineFenceLanguage = findNode(
  inlineCodeBlock,
  (node) => node.className === "message-code-language",
);
assert.equal(inlineFenceLanguage.textContent, "java");
const inlineFenceCode = findNode(inlineCodeBlock, (node) => node.tagName === "CODE");
assert.equal(inlineFenceCode.textContent, "public class InlineFence {}");

const emptyCase = await runCase({
  available: false,
  redacted: false,
  label: "Model Thinking (unverified)",
  content: null,
  note: "Model-emitted thinking is useful for debugging the loop.",
});
const emptyDom = emptyCase.dom;
const emptyThinking = findNode(
  emptyDom.messages,
  (node) => node.className === "message-thinking",
);
assert.equal(emptyThinking, null);

const threadCase = await runCase({
  available: false,
  redacted: false,
  label: "Model Thinking (unverified)",
  content: null,
  note: "Model-emitted thinking is useful for debugging the loop.",
});
const firstThreadId = threadCase.queryBodies[0].session_id;
await threadCase.dom["new-thread"].dispatch("click");
threadCase.dom["query-input"].value = "Second thread question";
await threadCase.dom["query-form"].dispatch("submit");
assert.equal(threadCase.queryBodies.length, 2);
assert.notEqual(threadCase.queryBodies[1].session_id, firstThreadId);
assert.equal(threadCase.dom["thread-list"].children.length, 2);

const staleDom = createDom();
const staleQuery = deferred();
const staleServerThreads = [createThreadPayload("thread_stale")];
globalThis.fetch = async (url, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  if (url === "/api/config") {
    return jsonResponse({ text_encodings: [{ label: "Auto", value: "auto" }] });
  }
  if (url === "/api/status") {
    return jsonResponse({
      backend: "ollama",
      model: "thinking-model",
      ready_for_queries: false,
      query_mode: "direct",
      chunk_count: 0,
    });
  }
  if (url === "/api/threads" && method === "GET") {
    return jsonResponse({ threads: staleServerThreads });
  }
  if (url.startsWith("/api/threads/") && method === "GET") {
    const id = decodeURIComponent(url.slice("/api/threads/".length));
    const thread = staleServerThreads.find((item) => item.id === id);
    if (thread) {
      return jsonResponse(thread);
    }
  }
  if (url === "/api/query") {
    await staleQuery.promise;
    return jsonResponse({
      answer: "Stale answer should not reappear",
      timeline: { rows: [], final_decision: "not_verified" },
      summary: {},
      trace: { model_thinking: null },
    });
  }
  if (url === "/api/chat/clear") {
    return jsonResponse({
      timeline: { rows: [], final_decision: null },
      summary: {},
      trace: {},
    });
  }
  throw new Error(`unexpected fetch ${url}`);
};
await importFreshApp();
await new Promise((resolve) => setTimeout(resolve, 0));
staleDom["query-input"].value = "Question that will be cleared";
const pendingSubmit = staleDom["query-form"].dispatch("submit");
await new Promise((resolve) => setTimeout(resolve, 0));
await staleDom["clear-chat"].dispatch("click");
staleQuery.resolve();
await pendingSubmit;
const staleAssistant = findNode(
  staleDom.messages,
  (node) => node.className === "message assistant",
);
assert.equal(staleAssistant, null);
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "APP_JS_PATH": str(REPO_ROOT / "src" / "web_static" / "app.js"),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stderr == ""


def test_config_and_status_endpoints_return_runtime_contract():
    fake_qa = FakeQA()
    client = TestClient(web_app.create_app(fake_qa))

    config = client.get("/api/config").json()
    status = client.get("/api/status").json()

    assert config["title"] == "AI Loop Engine"
    assert {"label": "Auto", "value": "auto"} in config["text_encodings"]
    assert status["active_document"] is None
    assert status["backend"] == "mock"
    assert status["ready_for_queries"] is False
    assert status["readiness_scope"] == "retrieval_pipeline"
    assert status["direct_query_available"] is True
    assert status["query_mode"] == "direct"
    assert status["context_optional"] is True


def test_upload_document_indexes_context_and_reports_status():
    fake_qa = FakeQA()
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post(
        "/api/documents",
        data={"text_encoding": "cp1251"},
        files={"file": ("demo.txt", b"Project Phoenix", "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "processed in mock mode" in payload["message"]
    assert payload["status"]["active_document"] == "demo.txt"
    assert payload["status"]["text_encoding_mode"] == "cp1251"
    assert fake_qa.text_encoding == "cp1251"
    assert fake_qa.uploaded_text == "Project Phoenix"


def test_upload_document_rejects_unknown_encoding():
    client = TestClient(web_app.create_app(FakeQA()))

    response = client.post(
        "/api/documents",
        data={"text_encoding": "not-real"},
        files={"file": ("demo.txt", b"Project Phoenix", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported text encoding."


def test_upload_document_rejects_oversized_body_before_processing(monkeypatch):
    fake_qa = FakeQA()
    client = TestClient(web_app.create_app(fake_qa))
    monkeypatch.setattr(web_app, "MAX_DOCUMENT_BYTES", 8)

    response = client.post(
        "/api/documents",
        data={"text_encoding": "auto"},
        files={"file": ("too-large.txt", b"more than eight", "text/plain")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Uploaded document exceeds the 25 MB limit."
    assert getattr(fake_qa, "status_calls", 0) == 0
    assert not hasattr(fake_qa, "document_path")


def test_upload_document_rejects_oversized_request_before_status(monkeypatch):
    fake_qa = FakeQA()
    client = TestClient(web_app.create_app(fake_qa))
    monkeypatch.setattr(web_app, "MAX_DOCUMENT_BYTES", 8)
    monkeypatch.setattr(web_app, "MAX_MULTIPART_OVERHEAD_BYTES", 0)

    response = client.post(
        "/api/documents",
        data={"text_encoding": "auto"},
        files={"file": ("too-large.txt", b"too-large", "text/plain")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Uploaded document exceeds the 25 MB limit."
    assert getattr(fake_qa, "status_calls", 0) == 0
    assert not hasattr(fake_qa, "document_path")


def test_upload_document_rejects_missing_content_length_before_parsing():
    fake_qa = FakeQA()
    api = web_app.create_app(fake_qa)
    sent_messages = []

    async def receive():
        return {
            "type": "http.request",
            "body": b"unparsed multipart body",
            "more_body": False,
        }

    async def send(message):
        sent_messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/documents",
        "raw_path": b"/api/documents",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "root_path": "",
    }

    asyncio.run(api(scope, receive, send))

    response_start = next(
        message for message in sent_messages if message["type"] == "http.response.start"
    )
    response_body = b"".join(
        message.get("body", b"")
        for message in sent_messages
        if message["type"] == "http.response.body"
    )
    assert response_start["status"] == 411
    assert json.loads(response_body)["detail"] == (
        "Content-Length is required for document uploads."
    )
    assert getattr(fake_qa, "status_calls", 0) == 0
    assert not hasattr(fake_qa, "document_path")


def test_upload_document_rejects_directory_like_filenames():
    fake_qa = FakeQA()
    client = TestClient(web_app.create_app(fake_qa))

    for filename in (".", ".."):
        response = client.post(
            "/api/documents",
            data={"text_encoding": "auto"},
            files={"file": (filename, b"content", "text/plain")},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid upload filename."
    assert getattr(fake_qa, "status_calls", 0) == 0
    assert not hasattr(fake_qa, "document_path")


def test_safe_upload_name_rejects_control_and_overlong_names():
    for filename in (
        "bad\0.txt",
        "bad\n.txt",
        "bad\x7f.txt",
        "\nbad.txt",
        "bad.txt\n",
        "\tbad.txt",
        "a" * 181,
    ):
        with pytest.raises(web_app.HTTPException) as exc_info:
            web_app.safe_upload_name(filename)

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Invalid upload filename."


def test_upload_document_rejects_overlong_filename_before_filesystem():
    fake_qa = FakeQA()
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post(
        "/api/documents",
        data={"text_encoding": "auto"},
        files={"file": ("a" * 181, b"content", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid upload filename."
    assert getattr(fake_qa, "status_calls", 0) == 0
    assert not hasattr(fake_qa, "document_path")


def test_visually_hidden_label_remains_accessible_to_assistive_tech():
    css = (REPO_ROOT / "src" / "web_static" / "styles.css").read_text(
        encoding="utf-8"
    )

    assert ".hidden {\n  display: none;\n}" in css
    assert ".visually-hidden {" in css
    assert "clip-path: inset(50%);" in css
    assert "position: absolute;" in css
    assert ".hidden,\n.visually-hidden" not in css


def test_upload_document_reports_processing_failure_without_losing_active_status():
    fake_qa = FakeQA()
    fake_qa.current_document_name = "good.txt"
    fake_qa.latest_processing_report = processed_report()

    def fail_process_document(document_path, text_encoding=None):
        fake_qa.text_encoding = text_encoding
        fake_qa.latest_processing_report = DocumentProcessingReport(
            attempted_document_name="bad.txt",
            active_document_name="good.txt",
            success=False,
            phase="load",
            file_extension=".txt",
            chunk_count=0,
            truncated=False,
            max_chunk_limit=2000,
            text_encoding_mode=text_encoding or "auto",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            error_message="Could not decode text document",
        )
        raise DocumentProcessingError(
            "Error loading document: Could not decode text document",
            fake_qa.status(),
        )

    fake_qa.process_document = fail_process_document
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post(
        "/api/documents",
        data={"text_encoding": "auto"},
        files={"file": ("bad.txt", b"bad", "text/plain")},
    )

    assert response.status_code == 400
    payload = response.json()
    assert "failed during load" in payload["message"]
    assert "Active document remains good.txt" in payload["message"]
    assert payload["status"]["active_document"] == "good.txt"
    assert payload["status"]["last_attempted_document"] == "bad.txt"
    assert payload["status"]["last_error"] == "Could not decode text document"


def test_upload_document_unexpected_error_uses_pre_upload_status():
    class UnexpectedFailureQA(FakeQA):
        def __init__(self):
            self.current_document_name = "good.txt"
            self.latest_processing_report = processed_report()
            self.status_calls = 0

        def process_document(self, document_path, text_encoding=None):
            self.current_document_name = "mutated.txt"
            self.latest_processing_report = processed_report(
                document_name="mutated.txt"
            )
            raise RuntimeError("unexpected boom")

    fake_qa = UnexpectedFailureQA()
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post(
        "/api/documents",
        data={"text_encoding": "auto"},
        files={"file": ("bad.txt", b"bad", "text/plain")},
    )

    assert response.status_code == 500
    payload = response.json()
    assert fake_qa.status_calls == 1
    assert "failed during unexpected" in payload["message"]
    assert "Active document remains good.txt" in payload["message"]
    assert payload["status"]["active_document"] == "good.txt"
    assert payload["status"]["phase"] == "unexpected"
    assert payload["status"]["last_error"] == "unexpected boom"


def test_query_endpoint_returns_visible_loop_payload():
    fake_qa = FakeQA()
    fake_qa.current_document_name = "demo.txt"
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post(
        "/api/query",
        json={"message": "What is Project Phoenix?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Project Phoenix" in payload["answer"]
    assert payload["timeline"]["rows"][0]["phase"] == "Retrieve"
    assert payload["timeline"]["rows"][0]["signals"] == "1 prompt chunks; chunks: 1; citations: 1"
    assert payload["summary"]["document"] == "demo.txt"
    assert payload["summary"]["final_decision"] == "not_verified"
    assert payload["trace"]["question"] == "What is Project Phoenix?"
    assert payload["trace"]["citations"][0]["source"] == "demo.txt"
    assert payload["trace"]["model_thinking"] == {
        "available": True,
        "redacted": False,
        "label": "Model Thinking (unverified)",
        "content": "I matched Project Phoenix against citation [1].",
        "note": (
            "Model-emitted thinking is useful for debugging the loop, but it is "
            "not verified evidence."
        ),
    }


def test_query_endpoint_passes_session_id_to_loop_runtime():
    fake_qa = FakeQA()
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post(
        "/api/query",
        json={"message": "What is Project Phoenix?", "session_id": "thread_alpha"},
    )

    assert response.status_code == 200
    assert fake_qa.last_query_session_id == "thread_alpha"
    assert response.json()["trace"]["loop_report"]["run"]["session_id"] == (
        "thread_alpha"
    )


def test_thread_endpoints_create_list_get_rename_and_delete():
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))

    initial = client.get("/api/threads")
    assert initial.status_code == 200
    assert len(initial.json()["threads"]) == 1

    created = client.post("/api/threads", json={"title": "Research thread"})
    assert created.status_code == 200
    thread_id = created.json()["id"]
    assert thread_id.startswith("thread_")
    assert created.json()["title"] == "Research thread"

    listed = client.get("/api/threads")
    assert listed.status_code == 200
    assert any(thread["id"] == thread_id for thread in listed.json()["threads"])

    fetched = client.get(f"/api/threads/{thread_id}")
    assert fetched.status_code == 200
    assert fetched.json()["messages"] == []

    renamed = client.patch(
        f"/api/threads/{thread_id}",
        json={"title": "Renamed thread"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed thread"

    deleted = client.delete(f"/api/threads/{thread_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "thread_id": thread_id}
    assert fake_qa.cleared_loop_session_id == thread_id
    assert client.get(f"/api/threads/{thread_id}").status_code == 404


def test_query_endpoint_persists_thread_messages_and_latest_payload():
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))

    response = client.post(
        "/api/query",
        json={"message": "What is Project Phoenix?", "session_id": "thread_alpha"},
    )

    assert response.status_code == 200
    thread = client.get("/api/threads/thread_alpha").json()
    listed_thread = next(
        thread
        for thread in client.get("/api/threads").json()["threads"]
        if thread["id"] == "thread_alpha"
    )
    assert "latest" not in listed_thread
    assert thread["title"] == "What is Project Phoenix?"
    assert [message["role"] for message in thread["messages"]] == [
        "user",
        "assistant",
    ]
    assert thread["messages"][0]["content"] == "What is Project Phoenix?"
    assert "Project Phoenix" in thread["messages"][1]["content"]
    assert thread["messages"][1]["thinking"]["available"] is True
    assert thread["latest"]["summary"]["final_decision"] == "not_verified"
    assert thread["latest"]["trace"]["loop_report"]["run"]["session_id"] == (
        "thread_alpha"
    )


def test_query_endpoint_passes_recent_same_thread_history_to_runtime():
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_alpha")
    store.append_message(
        "thread_alpha",
        role="user",
        content="Do you know what dynamic programming is?",
    )
    store.append_message("thread_alpha", role="assistant", content="Yes.")
    store.create_thread(thread_id="thread_other")
    store.append_message(
        "thread_other",
        role="user",
        content="This should stay out of alpha.",
    )
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))

    response = client.post(
        "/api/query",
        json={
            "message": "Please explain it in layman terms.",
            "session_id": "thread_alpha",
        },
    )

    assert response.status_code == 200
    assert fake_qa.last_query_session_id == "thread_alpha"
    assert fake_qa.last_conversation_history == [
        {
            "role": "user",
            "content": "Do you know what dynamic programming is?",
        },
        {"role": "assistant", "content": "Yes."},
    ]
    assert all(
        "stay out" not in entry["content"]
        for entry in fake_qa.last_conversation_history
    )


def test_query_endpoint_retrieves_older_semantic_thread_memory():
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_alpha")
    relevant = store.append_message(
        "thread_alpha",
        role="user",
        content="Dynamic programming means reusing answers to subproblems.",
    )
    unrelated = store.append_message(
        "thread_alpha",
        role="assistant",
        content="Banana bread uses ripe bananas.",
    )
    store.upsert_message_embedding(
        relevant,
        embedding_model="fake-memory",
        vector=[1.0, 0.0],
    )
    store.upsert_message_embedding(
        unrelated,
        embedding_model="fake-memory",
        vector=[0.0, 1.0],
    )
    for index in range(web_app.MAX_QUERY_HISTORY_MESSAGES):
        store.append_message(
            "thread_alpha",
            role="user" if index % 2 == 0 else "assistant",
            content=f"recent filler {index}",
        )
    other_thread_memory = store.append_message(
        "thread_other",
        role="user",
        content="Dynamic programming in another thread must not leak.",
    )
    store.upsert_message_embedding(
        other_thread_memory,
        embedding_model="fake-memory",
        vector=[1.0, 0.0],
    )
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))

    response = client.post(
        "/api/query",
        json={
            "message": "Can you explain that algorithm simply?",
            "session_id": "thread_alpha",
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert fake_qa.last_semantic_memory_status == "retrieved"
    assert fake_qa.last_semantic_memory == [
        {
            "message_id": relevant.id,
            "role": "user",
            "content": "Dynamic programming means reusing answers to subproblems.",
            "score": 1.0,
        }
    ]
    assert all(
        "another thread" not in entry["content"]
        for entry in fake_qa.last_semantic_memory
    )
    assert all(
        "Dynamic programming means" not in entry["content"]
        for entry in fake_qa.last_conversation_history
    )
    memory_rows = [
        row
        for row in payload["timeline"]["rows"]
        if row["step"] == "Retrieve thread memory"
    ]
    assert memory_rows
    assert "memory: 1" in memory_rows[0]["signals"]
    assert payload["summary"]["semantic_memory_count"] == 1
    assert payload["summary"]["semantic_memory_status"] == "retrieved"
    indexed_memories = store.semantic_memories(
        "thread_alpha",
        embedding_model="fake-memory",
        query_vector=[1.0, 0.0],
        min_score=0.0,
    )
    assert any(
        memory.content == "Can you explain that algorithm simply?"
        for memory in indexed_memories
    )


def test_query_endpoint_does_not_persist_partial_turn_on_runtime_failure():
    class FailingQA(FakeQA):
        def query_with_trace(
            self,
            message,
            session_id="default",
            conversation_history=None,
            semantic_memory=None,
            semantic_memory_status="not_requested",
        ):
            self.last_query_session_id = session_id
            self.last_conversation_history = list(conversation_history or [])
            self.last_semantic_memory = list(semantic_memory or [])
            self.last_semantic_memory_status = semantic_memory_status
            raise RuntimeError("backend unavailable")

    fake_qa = FailingQA()
    store = ThreadStore.in_memory()
    client = TestClient(
        web_app.create_app(fake_qa, thread_store=store),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/api/query",
        json={"message": "hello", "session_id": "thread_fail"},
    )

    assert response.status_code == 500
    thread = client.get("/api/threads/thread_fail").json()
    assert thread["messages"] == []
    assert thread["message_count"] == 0
    assert thread["latest"] is None


def test_clear_chat_blocks_stale_in_flight_query_persistence():
    class BlockingQA(FakeQA):
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def query_with_trace(
            self,
            message,
            session_id="default",
            conversation_history=None,
            semantic_memory=None,
            semantic_memory_status="not_requested",
        ):
            self.last_query_session_id = session_id
            self.last_conversation_history = list(conversation_history or [])
            self.last_semantic_memory = list(semantic_memory or [])
            self.last_semantic_memory_status = semantic_memory_status
            self.started.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("timed out waiting for test release")
            return super().query_with_trace(
                message,
                session_id=session_id,
                conversation_history=conversation_history,
                semantic_memory=semantic_memory,
                semantic_memory_status=semantic_memory_status,
            )

    fake_qa = BlockingQA()
    store = ThreadStore.in_memory()
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))
    responses = {}

    def run_query():
        responses["query"] = client.post(
            "/api/query",
            json={"message": "stale me", "session_id": "thread_race"},
        )

    query_thread = threading.Thread(target=run_query)
    query_thread.start()
    assert fake_qa.started.wait(timeout=5)

    clear_response = client.post(
        "/api/chat/clear",
        json={"session_id": "thread_race"},
    )
    fake_qa.release.set()
    query_thread.join(timeout=5)

    assert not query_thread.is_alive()
    assert clear_response.status_code == 200
    assert responses["query"].status_code == 200
    thread = client.get("/api/threads/thread_race").json()
    assert thread["messages"] == []
    assert thread["message_count"] == 0
    assert thread["latest"] is None


def test_delete_thread_blocks_stale_in_flight_query_resurrection():
    class BlockingQA(FakeQA):
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def query_with_trace(
            self,
            message,
            session_id="default",
            conversation_history=None,
            semantic_memory=None,
            semantic_memory_status="not_requested",
        ):
            self.last_query_session_id = session_id
            self.last_conversation_history = list(conversation_history or [])
            self.last_semantic_memory = list(semantic_memory or [])
            self.last_semantic_memory_status = semantic_memory_status
            self.started.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("timed out waiting for test release")
            return super().query_with_trace(
                message,
                session_id=session_id,
                conversation_history=conversation_history,
                semantic_memory=semantic_memory,
                semantic_memory_status=semantic_memory_status,
            )

    fake_qa = BlockingQA()
    store = ThreadStore.in_memory()
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))
    responses = {}

    def run_query():
        responses["query"] = client.post(
            "/api/query",
            json={"message": "stale me", "session_id": "thread_race"},
        )

    query_thread = threading.Thread(target=run_query)
    query_thread.start()
    assert fake_qa.started.wait(timeout=5)

    delete_response = client.delete("/api/threads/thread_race")
    fake_qa.release.set()
    query_thread.join(timeout=5)

    assert not query_thread.is_alive()
    assert delete_response.status_code == 200
    assert responses["query"].status_code == 200
    assert client.get("/api/threads/thread_race").status_code == 404


def test_delete_recreate_blocks_stale_in_flight_query_resurrection():
    class BlockingQA(FakeQA):
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def query_with_trace(
            self,
            message,
            session_id="default",
            conversation_history=None,
            semantic_memory=None,
            semantic_memory_status="not_requested",
        ):
            self.last_query_session_id = session_id
            self.last_conversation_history = list(conversation_history or [])
            self.last_semantic_memory = list(semantic_memory or [])
            self.last_semantic_memory_status = semantic_memory_status
            self.started.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("timed out waiting for test release")
            return super().query_with_trace(
                message,
                session_id=session_id,
                conversation_history=conversation_history,
                semantic_memory=semantic_memory,
                semantic_memory_status=semantic_memory_status,
            )

    fake_qa = BlockingQA()
    store = ThreadStore.in_memory()
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))
    responses = {}

    def run_query():
        responses["query"] = client.post(
            "/api/query",
            json={"message": "stale me", "session_id": "thread_race"},
        )

    query_thread = threading.Thread(target=run_query)
    query_thread.start()
    assert fake_qa.started.wait(timeout=5)

    delete_response = client.delete("/api/threads/thread_race")
    recreated = store.create_thread(thread_id="thread_race")
    fake_qa.release.set()
    query_thread.join(timeout=5)

    assert not query_thread.is_alive()
    assert delete_response.status_code == 200
    assert responses["query"].status_code == 200
    thread = client.get("/api/threads/thread_race").json()
    assert thread["id"] == recreated.id
    assert thread["messages"] == []
    assert thread["message_count"] == 0
    assert thread["latest"] is None


def test_query_endpoint_rejects_invalid_session_id():
    client = TestClient(web_app.create_app(FakeQA()))

    response = client.post(
        "/api/query",
        json={"message": "What is Project Phoenix?", "session_id": "../bad"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid session id."


def test_query_endpoint_allows_no_context_loop():
    qa = DocumentQA(fast_mode=True, llm_backend="mock")
    client = TestClient(web_app.create_app(qa))

    response = client.post(
        "/api/query",
        json={"message": "What can you do?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "mock response" in payload["answer"]
    assert payload["summary"]["context_provider"] == "none"
    assert payload["summary"]["document"] is None
    assert payload["summary"]["final_decision"] == "not_verified"
    assert payload["trace"]["retrieved_chunk_count"] == 0
    assert payload["trace"]["citations"] == []
    assert payload["trace"]["self_check"]["outcome"] == "not_verified"
    assert payload["timeline"]["rows"][0]["phase"] == "Context"
    assert payload["timeline"]["rows"][0]["signals"] == "no_context_provider"
    assert payload["trace"]["model_thinking"]["available"] is False
    assert payload["trace"]["model_thinking"]["content"] is None


def test_query_endpoint_rejects_blank_message():
    client = TestClient(web_app.create_app(FakeQA()))

    response = client.post("/api/query", json={"message": "  "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Message is required."


def test_clear_chat_resets_session_state():
    fake_qa = FakeQA()
    fake_qa.chat_history = [{"question": "old"}]
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post("/api/chat/clear")

    assert response.status_code == 200
    assert fake_qa.chat_history == []
    assert fake_qa.cleared_loop_session_id == "default"
    assert response.json()["timeline"]["empty"] is True


def test_clear_chat_resets_requested_session_state():
    class StaleAppendDuringClearQA(FakeQA):
        def clear_loop_session(self, session_id="default"):
            super().clear_loop_session(session_id)
            self.chat_history.append(
                {"session_id": session_id, "question": "stale in-flight"}
            )

    fake_qa = StaleAppendDuringClearQA()
    fake_qa.chat_history = [
        {"session_id": "thread_beta", "question": "old"},
        {"session_id": "thread_other", "question": "keep"},
    ]
    client = TestClient(web_app.create_app(fake_qa))

    response = client.post("/api/chat/clear", json={"session_id": "thread_beta"})

    assert response.status_code == 200
    assert fake_qa.chat_history == [
        {"session_id": "thread_other", "question": "keep"}
    ]
    assert fake_qa.cleared_loop_session_id == "thread_beta"
    assert response.json()["timeline"]["empty"] is True


def test_clear_chat_resets_persisted_thread_messages_only():
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_beta")
    store.create_thread(thread_id="thread_other")
    store.append_message("thread_beta", role="user", content="old")
    store.append_message("thread_other", role="user", content="keep")
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))

    response = client.post("/api/chat/clear", json={"session_id": "thread_beta"})

    assert response.status_code == 200
    assert client.get("/api/threads/thread_beta").json()["messages"] == []
    other_messages = client.get("/api/threads/thread_other").json()["messages"]
    assert [message["content"] for message in other_messages] == ["keep"]


def test_clear_chat_does_not_recreate_deleted_thread():
    fake_qa = FakeQA()
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_deleted")
    store.delete_thread("thread_deleted")
    client = TestClient(web_app.create_app(fake_qa, thread_store=store))

    response = client.post(
        "/api/chat/clear",
        json={"session_id": "thread_deleted"},
    )

    assert response.status_code == 200
    assert client.get("/api/threads/thread_deleted").status_code == 404


def test_clear_chat_rejects_invalid_session_id():
    client = TestClient(web_app.create_app(FakeQA()))

    response = client.post("/api/chat/clear", json={"session_id": "bad/session"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid session id."


def test_loop_contract_redacts_guardrail_blocked_draft_everywhere():
    blocked_draft = "Sensitive blocked draft should not be public."
    secret_question = "Generate unsafe content with SECRET_USER_QUESTION."
    blocked_answer = "Blocked answer with SECRET_PUBLIC_ANSWER."
    blocked_thinking = "Blocked model thinking with SECRET_MODEL_THINKING."
    loop_report = LoopReport(
        run=LoopRun(
            run_id="run_blocked",
            user_input=secret_question,
            context_provider="document",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            steps=(
                LoopStep(
                    phase=LoopPhase.DRAFT,
                    decision=LoopDecision.CONTINUE,
                    name="Draft answer",
                    input_summary=secret_question,
                    output_summary=blocked_draft,
                    verification=VerificationResult(
                        outcome=VerificationOutcome.UNSUPPORTED,
                        reasons=(blocked_draft,),
                        verifier="mock",
                        raw_response=blocked_draft,
                        metadata={"debug": blocked_draft},
                    ),
                    metadata={"draft_preview": blocked_draft},
                ),
                LoopStep(
                    phase=LoopPhase.ERROR,
                    decision=LoopDecision.BLOCK,
                    name="Guardrail decision",
                    output_summary=blocked_draft,
                    error_message=blocked_draft,
                    metadata={
                        "guardrail_decision": "block",
                        "guardrail_reason": blocked_draft,
                    },
                ),
            ),
            final_decision=LoopDecision.BLOCK,
            final_answer=blocked_answer,
            error_message=blocked_draft,
            metadata={"guardrail_detail": blocked_draft},
        )
    )
    result = QueryResult(
        answer=blocked_answer,
        trace=AnswerTrace(
            question=secret_question,
            document_name="phoenix.txt",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            retrieved_chunk_count=0,
            citations=[],
            error_message=blocked_draft,
            model_thinking=blocked_thinking,
        ),
        loop_report=loop_report,
    )

    public_payload = query_response_dict(result)
    public_json = json.dumps(public_payload)

    raw_report_json = json.dumps(loop_report.to_dict())
    assert blocked_draft in raw_report_json
    assert blocked_answer in raw_report_json
    assert secret_question in raw_report_json
    assert blocked_draft not in public_json
    assert blocked_answer not in public_json
    assert secret_question not in public_json
    assert blocked_thinking not in public_json
    assert public_payload["summary"]["last_error"] == "terminal_guardrail_decision"
    assert public_payload["timeline"]["rows"][0]["signals"] == (
        "[redacted: terminal guardrail decision]; "
        "verifier: unsupported; "
        "reasons: [redacted: terminal guardrail decision]"
    )
    assert public_payload["trace"]["question"] == (
        "[redacted: terminal guardrail decision]"
    )
    assert public_payload["answer"] == "[redacted: terminal guardrail decision]"
    assert public_payload["trace"]["answer"] == (
        "[redacted: terminal guardrail decision]"
    )
    assert public_payload["trace"]["model_thinking"] == {
        "available": False,
        "redacted": True,
        "label": "Model Thinking (unverified)",
        "content": "[redacted: terminal loop decision]",
        "note": (
            "Model-emitted thinking is useful for debugging the loop, but it is "
            "not verified evidence."
        ),
    }
    assert public_payload["trace"]["loop_report"]["public_redaction"]["applied"] is True


def test_answer_trace_redacts_model_thinking_for_refused_results_without_guardrail():
    secret_thinking = "SECRET_REFUSED_MODEL_THINKING"
    result = QueryResult(
        answer="I could not find enough relevant information in the document.",
        trace=AnswerTrace(
            question="What is unsupported?",
            document_name="phoenix.txt",
            backend="ollama",
            model_label="Ollama (nemotron-3-nano:4b)",
            retrieved_chunk_count=1,
            citations=[],
            model_thinking=secret_thinking,
        ),
        loop_report=LoopReport(
            run=LoopRun(
                run_id="run_refused",
                user_input="What is unsupported?",
                context_provider="document",
                backend="ollama",
                model_label="Ollama (nemotron-3-nano:4b)",
                final_decision=LoopDecision.REFUSE,
                final_answer="I could not find enough relevant information in the document.",
            )
        ),
    )

    trace = answer_trace_dict(result)

    assert secret_thinking not in json.dumps(trace)
    assert trace["model_thinking"] == {
        "available": False,
        "redacted": True,
        "label": "Model Thinking (unverified)",
        "content": "[redacted: terminal loop decision]",
        "note": (
            "Model-emitted thinking is useful for debugging the loop, but it is "
            "not verified evidence."
        ),
    }


def test_loop_timeline_fallback_error_rows_are_sequential():
    result = QueryResult(
        answer="A query error occurred.",
        trace=AnswerTrace(
            question="What failed?",
            document_name="demo.txt",
            backend="mock",
            model_label="MockLLM (explicit demo)",
            retrieved_chunk_count=0,
            citations=[],
            error_message="backend unavailable",
        ),
    )

    timeline = loop_timeline_dict(result)

    assert [row["index"] for row in timeline["rows"]] == [1, 2, 3]
    assert timeline["rows"][-1]["phase"] == "Error"


def test_runtime_status_dict_preserves_honest_readiness_scope():
    fake_qa = FakeQA()
    status = runtime_status_dict(fake_qa.status())

    assert status["ready_for_queries"] is False
    assert status["readiness_scope"] == "retrieval_pipeline"
    assert status["inference_validated"] is False
    assert status["max_output_tokens"] == 384
