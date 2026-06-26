const state = {
  messages: [],
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

function renderRuntimeStatus(status) {
  elements.backendPill.textContent = status.backend || "unconfigured";
  elements.modelPill.textContent = status.model || "model unavailable";
  elements.readyPill.textContent = status.ready_for_queries ? "indexed" : "not ready";
  elements.readyPill.dataset.ready = String(Boolean(status.ready_for_queries));

  const rows = [
    ["Active context", status.active_document || "none"],
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
  elements.messages.replaceChildren();
  if (!state.messages.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Ask a question after indexing context.";
    elements.messages.append(empty);
    return;
  }

  for (const message of state.messages) {
    const bubble = document.createElement("article");
    bubble.className = `message ${message.role}`;
    const role = document.createElement("span");
    role.className = "message-role";
    role.textContent = message.role === "user" ? "You" : "Loop";
    const content = document.createElement("p");
    content.textContent = message.content;
    bubble.append(role, content);
    elements.messages.append(bubble);
  }
  elements.messages.scrollTop = elements.messages.scrollHeight;
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

  state.messages.push({ role: "user", content: message });
  renderMessages();
  elements.queryInput.value = "";

  setBusy(elements.queryButton, true, "Run Loop");
  try {
    const result = await requestJson("/api/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message }),
    });
    state.messages.push({ role: "assistant", content: result.answer });
    renderMessages();
    renderLoopPayload(result);
  } catch (error) {
    state.messages.push({ role: "assistant", content: error.message });
    renderMessages();
  } finally {
    setBusy(elements.queryButton, false, "Run Loop");
  }
}

async function clearChat() {
  state.messages = [];
  renderMessages();
  try {
    const payload = await requestJson("/api/chat/clear", { method: "POST" });
    renderLoopPayload(payload);
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
  setupTabs();
  renderMessages();
  renderLoopPayload({
    timeline: { rows: [], final_decision: null, last_error: null },
    summary: {},
    trace: {},
  });
  elements.uploadForm.addEventListener("submit", uploadDocument);
  elements.queryForm.addEventListener("submit", runQuery);
  elements.clearButton.addEventListener("click", clearChat);
  elements.refreshStatus.addEventListener("click", refreshStatus);
  await loadConfig();
  await refreshStatus();
}

boot().catch((error) => {
  elements.uploadStatus.textContent = error.message;
});
