const THREAD_STORAGE_KEY = "ai-loop-engine.threads.v1";
const DEFAULT_THREAD_TITLE = "New thread";
const MAX_THREADS = 30;
const MAX_THREAD_MESSAGES = 100;
const SESSION_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$/;

const state = {
  threads: [],
  activeThreadId: null,
  latest: null,
};

const elements = {
  backendPill: document.querySelector("#backend-pill"),
  modelPill: document.querySelector("#model-pill"),
  readyPill: document.querySelector("#ready-pill"),
  uploadForm: document.querySelector("#upload-form"),
  uploadButton: document.querySelector("#upload-button"),
  uploadStatus: document.querySelector("#upload-status"),
  fileInput: document.querySelector("#document-file"),
  textEncoding: document.querySelector("#text-encoding"),
  newThreadButton: document.querySelector("#new-thread"),
  threadList: document.querySelector("#thread-list"),
  activeThreadTitle: document.querySelector("#active-thread-title"),
  refreshStatus: document.querySelector("#refresh-status"),
  runtimeGrid: document.querySelector("#runtime-grid"),
  queryForm: document.querySelector("#query-form"),
  queryInput: document.querySelector("#query-input"),
  queryButton: document.querySelector("#query-button"),
  clearButton: document.querySelector("#clear-chat"),
  messages: document.querySelector("#messages"),
  timeline: document.querySelector("#timeline"),
  finalDecision: document.querySelector("#final-decision"),
  thinkingPanel: document.querySelector("#thinking-panel"),
  thinkingState: document.querySelector("#thinking-state"),
  thinkingNote: document.querySelector("#thinking-note"),
  thinkingContent: document.querySelector("#thinking-content"),
  summaryJson: document.querySelector("#summary-json"),
  traceJson: document.querySelector("#trace-json"),
  tabButtons: document.querySelectorAll(".tab-button"),
};

function setBusy(button, busy, label) {
  button.disabled = busy;
  if (label) {
    button.textContent = busy ? "Working..." : label;
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : { detail: await response.text() };
  if (!response.ok) {
    const message = body.detail || body.message || `Request failed: ${response.status}`;
    throw new Error(message);
  }
  return body;
}

function emptyLoopPayload() {
  return {
    timeline: { rows: [], final_decision: null, last_error: null },
    summary: {},
    trace: {},
  };
}

function safeText(value, fallback = "", maxLength = 120) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text) {
    return fallback;
  }
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

function newThreadId() {
  const randomId = globalThis.crypto?.randomUUID
    ? globalThis.crypto.randomUUID()
    : `${Date.now()}_${Math.random().toString(36).slice(2)}`;
  return `thread_${randomId.replace(/[^A-Za-z0-9_.:-]/g, "_")}`.slice(0, 96);
}

function createThread(title = DEFAULT_THREAD_TITLE) {
  const timestamp = new Date().toISOString();
  return {
    id: newThreadId(),
    title,
    messages: [],
    latest: null,
    revision: 0,
    createdAt: timestamp,
    updatedAt: timestamp,
  };
}

function sanitizeMessage(message) {
  const role = message?.role === "assistant" ? "assistant" : "user";
  const sanitized = {
    role,
    content: String(message?.content || ""),
  };
  if (role === "assistant" && message?.thinking && typeof message.thinking === "object") {
    sanitized.thinking = message.thinking;
  }
  return sanitized;
}

function normalizedRevision(value) {
  const revision = Number(value);
  return Number.isSafeInteger(revision) && revision >= 0 ? revision : 0;
}

function bumpThreadRevision(thread) {
  thread.revision = normalizedRevision(thread.revision) + 1;
  return thread.revision;
}

function sanitizeThread(rawThread) {
  const fallbackThread = createThread();
  const rawId = String(rawThread?.id || "");
  const id = SESSION_ID_PATTERN.test(rawId) ? rawId : fallbackThread.id;
  const messages = Array.isArray(rawThread?.messages)
    ? rawThread.messages.slice(-MAX_THREAD_MESSAGES).map(sanitizeMessage)
    : [];
  const latest =
    rawThread?.latest && typeof rawThread.latest === "object" ? rawThread.latest : null;
  return {
    id,
    title: safeText(rawThread?.title, DEFAULT_THREAD_TITLE, 64),
    messages,
    latest,
    revision: normalizedRevision(rawThread?.revision),
    createdAt: String(rawThread?.createdAt || fallbackThread.createdAt),
    updatedAt: String(rawThread?.updatedAt || fallbackThread.updatedAt),
  };
}

function loadThreads() {
  let payload = null;
  try {
    payload = JSON.parse(globalThis.localStorage?.getItem(THREAD_STORAGE_KEY) || "null");
  } catch {
    payload = null;
  }

  const threads = Array.isArray(payload?.threads)
    ? payload.threads.map(sanitizeThread).filter((thread) => thread.id)
    : [];
  state.threads = threads.length ? threads.slice(0, MAX_THREADS) : [createThread()];
  const activeId = String(payload?.activeThreadId || "");
  state.activeThreadId = state.threads.some((thread) => thread.id === activeId)
    ? activeId
    : state.threads[0].id;
}

function persistThreads() {
  try {
    globalThis.localStorage?.setItem(
      THREAD_STORAGE_KEY,
      JSON.stringify({
        activeThreadId: state.activeThreadId,
        threads: state.threads.slice(0, MAX_THREADS),
      }),
    );
  } catch {
    // Browser storage is a convenience. The active in-memory thread remains usable.
  }
}

function activeThread() {
  let thread = state.threads.find((item) => item.id === state.activeThreadId);
  if (!thread) {
    thread = createThread();
    state.threads.unshift(thread);
    state.activeThreadId = thread.id;
    persistThreads();
  }
  return thread;
}

function threadById(threadId) {
  return state.threads.find((thread) => thread.id === threadId) || null;
}

function touchThread(thread) {
  thread.updatedAt = new Date().toISOString();
  thread.messages = thread.messages.slice(-MAX_THREAD_MESSAGES);
  state.threads = [
    thread,
    ...state.threads.filter((item) => item.id !== thread.id),
  ].slice(0, MAX_THREADS);
  persistThreads();
}

function titleFromMessage(message) {
  return safeText(message, DEFAULT_THREAD_TITLE, 54);
}

function renderThreads() {
  elements.threadList.replaceChildren();
  for (const thread of state.threads) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "thread-button";
    button.dataset.active = String(thread.id === state.activeThreadId);
    button.addEventListener("click", () => switchThread(thread.id));

    const title = document.createElement("span");
    title.textContent = thread.title || DEFAULT_THREAD_TITLE;
    const meta = document.createElement("small");
    const messageCount = thread.messages.length;
    meta.textContent = `${messageCount} ${messageCount === 1 ? "message" : "messages"}`;

    button.append(title, meta);
    elements.threadList.append(button);
  }
}

function renderActiveThreadTitle() {
  elements.activeThreadTitle.textContent = activeThread().title || DEFAULT_THREAD_TITLE;
}

function switchThread(threadId) {
  if (!threadById(threadId)) {
    return;
  }
  state.activeThreadId = threadId;
  persistThreads();
  renderThreads();
  renderActiveThreadTitle();
  renderMessages();
  renderLoopPayload(activeThread().latest || emptyLoopPayload());
}

function startNewThread() {
  const thread = createThread();
  state.threads.unshift(thread);
  state.activeThreadId = thread.id;
  persistThreads();
  renderThreads();
  renderActiveThreadTitle();
  renderMessages();
  renderLoopPayload(emptyLoopPayload());
  elements.queryInput.focus();
}

function renderRuntimeStatus(status) {
  elements.backendPill.textContent = status.backend || "unconfigured";
  elements.modelPill.textContent = status.model || "model unavailable";
  elements.readyPill.textContent = status.ready_for_queries ? "context indexed" : "direct mode";
  elements.readyPill.dataset.ready = String(Boolean(status.ready_for_queries));

  const rows = [
    ["Indexed context", status.active_document || "none"],
    ["Query mode", status.query_mode || (status.ready_for_queries ? "contextual" : "direct")],
    ["Last attempt", status.last_attempted_document || "none"],
    ["Profile", status.profile || "unknown"],
    ["Embedding model", status.embeddings_model || "unknown"],
    ["Chunks", status.chunk_count ?? 0],
    ["Phase", status.phase || "idle"],
    ["Last error", status.last_error || "none"],
  ];

  elements.runtimeGrid.replaceChildren();
  for (const [label, value] of rows) {
    const term = document.createElement("dt");
    term.textContent = label;
    const detail = document.createElement("dd");
    detail.textContent = String(value);
    elements.runtimeGrid.append(term, detail);
  }
}

function renderMessages() {
  const messages = activeThread().messages;
  elements.messages.replaceChildren();
  if (!messages.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Ask a question, or add context when you need grounded citations.";
    elements.messages.append(empty);
    return;
  }

  for (const message of messages) {
    const bubble = document.createElement("article");
    bubble.className = `message ${message.role}`;
    const role = document.createElement("span");
    role.className = "message-role";
    role.textContent = message.role === "user" ? "You" : "Loop";
    const content = document.createElement("p");
    content.textContent = message.content;
    bubble.append(role, content);
    const thinking = renderMessageThinking(message.thinking);
    if (thinking) {
      bubble.append(thinking);
    }
    elements.messages.append(bubble);
  }
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function renderMessageThinking(thinking) {
  const data = thinking || {};
  const hasThinking = Boolean(data.available && data.content);
  const isRedacted = Boolean(data.redacted);
  if (!hasThinking && !isRedacted) {
    return null;
  }

  const details = document.createElement("details");
  details.className = "message-thinking";
  details.open = true;

  const summary = document.createElement("summary");
  const label = document.createElement("span");
  label.textContent = data.label || "Model Thinking (unverified)";
  const state = document.createElement("small");
  state.textContent = isRedacted ? "redacted" : "captured";
  state.dataset.state = isRedacted ? "redacted" : "captured";
  summary.append(label, state);

  const note = document.createElement("p");
  note.className = "message-thinking-note";
  note.textContent =
    data.note ||
    "Model-emitted thinking is useful for debugging the loop, but it is not verified evidence.";

  const content = document.createElement("pre");
  content.textContent = data.content || "[redacted]";

  details.append(summary, note, content);
  return details;
}

function renderTimeline(timeline) {
  elements.timeline.replaceChildren();
  const rows = timeline?.rows || [];
  elements.finalDecision.textContent = timeline?.final_decision || "idle";
  elements.finalDecision.dataset.decision = timeline?.final_decision || "idle";

  if (!rows.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No loop run yet.";
    elements.timeline.append(empty);
    return;
  }

  for (const row of rows) {
    const item = document.createElement("article");
    item.className = "timeline-row";
    item.dataset.phase = row.phase_key || "";

    const index = document.createElement("span");
    index.className = "timeline-index";
    index.textContent = String(row.index);

    const body = document.createElement("div");
    const heading = document.createElement("div");
    heading.className = "timeline-heading";

    const phase = document.createElement("strong");
    phase.textContent = row.phase || "Step";
    const decision = document.createElement("span");
    decision.className = "decision";
    decision.textContent = row.decision || "continue";
    heading.append(phase, decision);

    const name = document.createElement("p");
    name.className = "timeline-name";
    name.textContent = row.step || "-";

    const signals = document.createElement("p");
    signals.className = "timeline-signals";
    signals.textContent = row.signals || "-";

    body.append(heading, name, signals);
    item.append(index, body);
    elements.timeline.append(item);
  }

  if (timeline.last_error) {
    const error = document.createElement("p");
    error.className = "timeline-error";
    error.textContent = `Last error: ${timeline.last_error}`;
    elements.timeline.append(error);
  }
}

function renderModelThinking(thinking) {
  const data = thinking || {};
  const hasThinking = Boolean(data.available && data.content);
  const isRedacted = Boolean(data.redacted);
  const label = data.label || "Model Thinking (unverified)";

  elements.thinkingPanel.querySelector("summary span").textContent = label;
  elements.thinkingNote.textContent =
    data.note ||
    "Model-emitted thinking is useful for debugging the loop, but it is not verified evidence.";
  elements.thinkingState.textContent = isRedacted
    ? "redacted"
    : hasThinking
      ? "captured"
      : "not captured";
  elements.thinkingState.dataset.state = isRedacted
    ? "redacted"
    : hasThinking
      ? "captured"
      : "empty";
  elements.thinkingContent.textContent = data.content
    ? data.content
    : "No model thinking captured for this run.";
  elements.thinkingPanel.open = hasThinking || isRedacted;
}

function renderLoopPayload(payload) {
  state.latest = payload;
  renderTimeline(payload.timeline);
  renderModelThinking(payload.trace?.model_thinking);
  elements.summaryJson.textContent = JSON.stringify(payload.summary, null, 2);
  elements.traceJson.textContent = JSON.stringify(payload.trace, null, 2);
}

async function loadConfig() {
  const config = await requestJson("/api/config");
  elements.textEncoding.replaceChildren();
  for (const option of config.text_encodings) {
    const node = document.createElement("option");
    node.value = option.value;
    node.textContent = option.label;
    elements.textEncoding.append(node);
  }
}

async function refreshStatus() {
  const status = await requestJson("/api/status");
  renderRuntimeStatus(status);
}

async function uploadDocument(event) {
  event.preventDefault();
  const file = elements.fileInput.files[0];
  if (!file) {
    elements.uploadStatus.textContent = "Choose a document first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("text_encoding", elements.textEncoding.value);

  setBusy(elements.uploadButton, true, "Index Context");
  elements.uploadStatus.textContent = "Indexing context...";
  try {
    const result = await requestJson("/api/documents", {
      method: "POST",
      body: formData,
    });
    elements.uploadStatus.textContent = result.message;
    renderRuntimeStatus(result.status);
  } catch (error) {
    elements.uploadStatus.textContent = error.message;
    await refreshStatus().catch(() => {});
  } finally {
    setBusy(elements.uploadButton, false, "Index Context");
  }
}

async function runQuery(event) {
  event.preventDefault();
  const message = elements.queryInput.value.trim();
  if (!message) {
    return;
  }

  const requestThread = activeThread();
  const requestThreadId = requestThread.id;
  const requestRevision = bumpThreadRevision(requestThread);
  const isFirstUserMessage = !requestThread.messages.some(
    (item) => item.role === "user",
  );
  requestThread.messages.push({ role: "user", content: message });
  if (isFirstUserMessage) {
    requestThread.title = titleFromMessage(message);
  }
  touchThread(requestThread);
  renderThreads();
  renderActiveThreadTitle();
  renderMessages();
  elements.queryInput.value = "";

  setBusy(elements.queryButton, true, "Run Loop");
  try {
    const result = await requestJson("/api/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message, session_id: requestThreadId }),
    });
    const targetThread = threadById(requestThreadId);
    if (!targetThread || normalizedRevision(targetThread.revision) !== requestRevision) {
      return;
    }
    targetThread.messages.push({
      role: "assistant",
      content: result.answer,
      thinking: result.trace?.model_thinking || null,
    });
    targetThread.latest = result;
    touchThread(targetThread);
    renderThreads();
    if (state.activeThreadId === requestThreadId) {
      renderActiveThreadTitle();
      renderMessages();
      renderLoopPayload(result);
    }
  } catch (error) {
    const targetThread = threadById(requestThreadId);
    if (!targetThread || normalizedRevision(targetThread.revision) !== requestRevision) {
      return;
    }
    targetThread.messages.push({ role: "assistant", content: error.message });
    touchThread(targetThread);
    renderThreads();
    if (state.activeThreadId === requestThreadId) {
      renderMessages();
    }
  } finally {
    setBusy(elements.queryButton, false, "Run Loop");
  }
}

async function clearChat() {
  const thread = activeThread();
  const threadId = thread.id;
  const clearRevision = bumpThreadRevision(thread);
  thread.messages = [];
  thread.latest = null;
  touchThread(thread);
  renderThreads();
  renderActiveThreadTitle();
  renderMessages();
  try {
    const payload = await requestJson("/api/chat/clear", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ session_id: threadId }),
    });
    const targetThread = threadById(threadId);
    if (!targetThread || normalizedRevision(targetThread.revision) !== clearRevision) {
      return;
    }
    targetThread.latest = payload;
    touchThread(targetThread);
    if (state.activeThreadId === threadId) {
      renderLoopPayload(payload);
    }
  } catch (error) {
    elements.uploadStatus.textContent = error.message;
  }
}

function setupTabs() {
  for (const button of elements.tabButtons) {
    button.addEventListener("click", () => {
      for (const tab of elements.tabButtons) {
        tab.classList.toggle("active", tab === button);
      }
      elements.summaryJson.classList.toggle(
        "hidden",
        button.dataset.target !== "summary-json",
      );
      elements.traceJson.classList.toggle(
        "hidden",
        button.dataset.target !== "trace-json",
      );
    });
  }
}

async function boot() {
  loadThreads();
  setupTabs();
  renderThreads();
  renderActiveThreadTitle();
  renderMessages();
  renderLoopPayload(activeThread().latest || emptyLoopPayload());
  elements.uploadForm.addEventListener("submit", uploadDocument);
  elements.queryForm.addEventListener("submit", runQuery);
  elements.newThreadButton.addEventListener("click", startNewThread);
  elements.clearButton.addEventListener("click", clearChat);
  elements.refreshStatus.addEventListener("click", refreshStatus);
  await loadConfig();
  await refreshStatus();
}

boot().catch((error) => {
  elements.uploadStatus.textContent = error.message;
});
