const ACTIVE_THREAD_STORAGE_KEY = "ai-loop-engine.active-thread.v1";
const ACTIVE_RECIPE_STORAGE_KEY = "ai-loop-engine.active-recipe.v1";
const LEGACY_THREAD_STORAGE_KEY = "ai-loop-engine.threads.v1";
const DEFAULT_THREAD_TITLE = "New thread";
const MAX_THREADS = 30;
const MAX_THREAD_MESSAGES = 100;
const SESSION_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$/;

const state = {
  threads: [],
  recipes: [],
  activeThreadId: null,
  activeRecipeId: null,
  recipeDraft: null,
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
  recipeSelect: document.querySelector("#recipe-select"),
  recipeStatus: document.querySelector("#recipe-status"),
  recipeState: document.querySelector("#recipe-state"),
  recipeEditor: document.querySelector("#recipe-editor"),
  recipeEditorState: document.querySelector("#recipe-editor-state"),
  recipeNewButton: document.querySelector("#recipe-new"),
  recipeSaveButton: document.querySelector("#recipe-save"),
  recipeDeleteButton: document.querySelector("#recipe-delete"),
  recipeExportButton: document.querySelector("#recipe-export"),
  recipeImport: document.querySelector("#recipe-import"),
  recipeName: document.querySelector("#recipe-name"),
  recipeGoal: document.querySelector("#recipe-goal"),
  recipeInstructions: document.querySelector("#recipe-instructions"),
  recipeCriteria: document.querySelector("#recipe-criteria"),
  recipeStop: document.querySelector("#recipe-stop"),
  activeThreadTitle: document.querySelector("#active-thread-title"),
  activeThreadMemory: document.querySelector("#active-thread-memory"),
  refreshStatus: document.querySelector("#refresh-status"),
  runtimeGrid: document.querySelector("#runtime-grid"),
  queryForm: document.querySelector("#query-form"),
  queryInput: document.querySelector("#query-input"),
  queryButton: document.querySelector("#query-button"),
  clearButton: document.querySelector("#clear-chat"),
  messages: document.querySelector("#messages"),
  memoryStatus: document.querySelector("#memory-status"),
  timeline: document.querySelector("#timeline"),
  runList: document.querySelector("#run-list"),
  runCount: document.querySelector("#run-count"),
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

function safeField(value, maxLength = 4000) {
  return String(value || "").slice(0, maxLength);
}

function nonNegativeInteger(value) {
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : 0;
}

function plural(count, singular, pluralForm = `${singular}s`) {
  return count === 1 ? singular : pluralForm;
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

function sanitizeLoopRun(rawRun) {
  const runId = String(rawRun?.run_id || "");
  return {
    run_id: SESSION_ID_PATTERN.test(runId) ? runId : "",
    final_decision: String(rawRun?.final_decision || "unknown"),
    context_provider: String(rawRun?.context_provider || "none"),
    backend: String(rawRun?.backend || "unknown"),
    model: String(rawRun?.model || rawRun?.model_label || "unknown"),
    recipe_id: String(rawRun?.recipe_id || ""),
    recipe_name: safeText(rawRun?.recipe_name, "General assistant loop", 96),
    step_count: Number.isSafeInteger(Number(rawRun?.step_count))
      ? Number(rawRun.step_count)
      : 0,
    started_at: String(rawRun?.started_at || ""),
    completed_at: String(rawRun?.completed_at || ""),
    created_at: String(rawRun?.created_at || ""),
  };
}

function sanitizeRecipe(rawRecipe) {
  const recipeId = String(rawRecipe?.recipe_id || "");
  const detailLoaded = Boolean(
    rawRecipe &&
      (
        Object.hasOwn(rawRecipe, "instructions") ||
        Object.hasOwn(rawRecipe, "success_criteria") ||
        Object.hasOwn(rawRecipe, "stop_condition") ||
        Object.hasOwn(rawRecipe, "metadata") ||
        Object.hasOwn(rawRecipe, "created_at")
      ),
  );
  const successCriteria = Array.isArray(rawRecipe?.success_criteria)
    ? rawRecipe.success_criteria
        .map((criterion) => safeField(criterion, 500).replace(/\s+/g, " ").trim())
        .filter(Boolean)
    : [];
  return {
    recipe_id: SESSION_ID_PATTERN.test(recipeId) ? recipeId : "",
    name: safeText(rawRecipe?.name, "Loop recipe", 96),
    description: safeField(rawRecipe?.description, 500).replace(/\s+/g, " ").trim(),
    goal: safeField(rawRecipe?.goal, 2000),
    instructions: safeField(rawRecipe?.instructions, 4000),
    success_criteria: successCriteria,
    stop_condition: safeField(rawRecipe?.stop_condition, 1000),
    context_provider: String(rawRecipe?.context_provider || "auto"),
    model_profile: String(rawRecipe?.model_profile || "quality"),
    verifier: String(rawRecipe?.verifier || "default"),
    metadata:
      rawRecipe?.metadata && typeof rawRecipe.metadata === "object"
        ? rawRecipe.metadata
        : {},
    detail_loaded: detailLoaded,
    is_default: Boolean(rawRecipe?.is_default),
    created_at: String(rawRecipe?.created_at || ""),
    updated_at: String(rawRecipe?.updated_at || ""),
  };
}

function normalizedRevision(value) {
  return nonNegativeInteger(value);
}

function bumpThreadRevision(thread) {
  thread.revision = normalizedRevision(thread.revision) + 1;
  return thread.revision;
}

function sanitizeThread(rawThread) {
  const rawId = String(rawThread?.id || "");
  const id = SESSION_ID_PATTERN.test(rawId) ? rawId : "";
  const messages = Array.isArray(rawThread?.messages)
    ? rawThread.messages.slice(-MAX_THREAD_MESSAGES).map(sanitizeMessage)
    : [];
  const latest =
    rawThread?.latest && typeof rawThread.latest === "object" ? rawThread.latest : null;
  const loopRuns = Array.isArray(rawThread?.loop_runs)
    ? rawThread.loop_runs.map(sanitizeLoopRun).filter((run) => run.run_id)
    : [];
  const rawLoopRunCount = Number(rawThread?.loop_run_count ?? rawThread?.loopRunCount);
  const rawMemoryCount = Number(rawThread?.memory_count ?? rawThread?.memoryCount);
  const loopRunCount =
    Number.isSafeInteger(rawLoopRunCount) && rawLoopRunCount >= 0
      ? rawLoopRunCount
      : loopRuns.length;
  const memoryCount =
    Number.isSafeInteger(rawMemoryCount) && rawMemoryCount >= 0
      ? rawMemoryCount
      : 0;
  const createdAt = String(
    rawThread?.created_at || rawThread?.createdAt || new Date().toISOString(),
  );
  const updatedAt = String(
    rawThread?.updated_at || rawThread?.updatedAt || createdAt,
  );
  const messageCount = Number(rawThread?.message_count);
  return {
    id,
    title: safeText(rawThread?.title, DEFAULT_THREAD_TITLE, 64),
    messages,
    loopRuns,
    loopRunCount,
    latest,
    revision: normalizedRevision(rawThread?.revision),
    createdAt,
    updatedAt,
    messageCount: Number.isSafeInteger(messageCount) && messageCount >= 0
      ? messageCount
      : messages.length,
    memoryCount,
  };
}

function applyThreadSummary(thread, summary) {
  if (!thread || !summary?.id || thread.id !== summary.id) {
    return;
  }
  thread.title = summary.title || thread.title;
  thread.updatedAt = summary.updatedAt || thread.updatedAt;
  thread.messageCount = Number.isSafeInteger(summary.messageCount)
    ? summary.messageCount
    : thread.messageCount;
  thread.loopRunCount = Number.isSafeInteger(summary.loopRunCount)
    ? summary.loopRunCount
    : thread.loopRunCount;
  thread.memoryCount = Number.isSafeInteger(summary.memoryCount)
    ? summary.memoryCount
    : thread.memoryCount;
}

function legacyActiveThreadId() {
  try {
    const payload = JSON.parse(
      globalThis.localStorage?.getItem(LEGACY_THREAD_STORAGE_KEY) || "null",
    );
    return String(payload?.activeThreadId || "");
  } catch {
    return "";
  }
}

function loadActiveThreadId() {
  const activeId = String(
    globalThis.localStorage?.getItem(ACTIVE_THREAD_STORAGE_KEY) || "",
  );
  return activeId || legacyActiveThreadId();
}

function loadActiveRecipeId() {
  return String(globalThis.localStorage?.getItem(ACTIVE_RECIPE_STORAGE_KEY) || "");
}

function persistActiveThreadId() {
  try {
    if (state.activeThreadId) {
      globalThis.localStorage?.setItem(
        ACTIVE_THREAD_STORAGE_KEY,
        state.activeThreadId,
      );
    }
  } catch {
    // Browser storage is only a selected-thread hint. Server threads remain authoritative.
  }
}

function persistActiveRecipeId() {
  try {
    if (state.activeRecipeId) {
      globalThis.localStorage?.setItem(
        ACTIVE_RECIPE_STORAGE_KEY,
        state.activeRecipeId,
      );
    }
  } catch {
    // Browser storage is only a selected-recipe hint. Server recipes remain authoritative.
  }
}

function upsertThread(thread, { moveToTop = false } = {}) {
  if (!thread.id) {
    return;
  }
  const others = state.threads.filter((item) => item.id !== thread.id);
  state.threads = moveToTop
    ? [thread, ...others].slice(0, MAX_THREADS)
    : [thread, ...others].sort((left, right) =>
        String(right.updatedAt).localeCompare(String(left.updatedAt)),
      ).slice(0, MAX_THREADS);
}

async function loadThreadDetail(threadId) {
  const detail = sanitizeThread(
    await requestJson(`/api/threads/${encodeURIComponent(threadId)}`),
  );
  upsertThread(detail);
  return detail;
}

async function loadThreads() {
  const payload = await requestJson("/api/threads");
  state.threads = Array.isArray(payload?.threads)
    ? payload.threads.map(sanitizeThread).filter((thread) => thread.id)
    : [];

  if (!state.threads.length) {
    const created = sanitizeThread(
      await requestJson("/api/threads", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ title: DEFAULT_THREAD_TITLE }),
      }),
    );
    state.threads = [created];
  }

  const activeId = loadActiveThreadId();
  state.activeThreadId = state.threads.some((thread) => thread.id === activeId)
    ? activeId
    : state.threads[0].id;
  await loadThreadDetail(state.activeThreadId);
  persistActiveThreadId();
}

function activeThread() {
  let thread = state.threads.find((item) => item.id === state.activeThreadId);
  if (!thread) {
    thread = state.threads[0] || {
      id: "",
      title: DEFAULT_THREAD_TITLE,
      messages: [],
      loopRuns: [],
      loopRunCount: 0,
      memoryCount: 0,
      latest: null,
      revision: 0,
      createdAt: "",
      updatedAt: "",
      messageCount: 0,
    };
    state.activeThreadId = thread.id;
    persistActiveThreadId();
  }
  return thread;
}

function threadById(threadId) {
  return state.threads.find((thread) => thread.id === threadId) || null;
}

function touchThread(thread) {
  thread.updatedAt = new Date().toISOString();
  thread.messages = thread.messages.slice(-MAX_THREAD_MESSAGES);
  thread.messageCount = thread.messages.length;
  upsertThread(thread, { moveToTop: true });
  persistActiveThreadId();
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
    button.addEventListener("click", () =>
      switchThread(thread.id).catch((error) => {
        elements.uploadStatus.textContent = error.message;
      }),
    );

    const title = document.createElement("span");
    title.textContent = thread.title || DEFAULT_THREAD_TITLE;
    const meta = document.createElement("small");
    const messageCount = Number.isSafeInteger(thread.messageCount)
      ? thread.messageCount
      : thread.messages.length;
    const runCount = Number.isSafeInteger(thread.loopRunCount)
      ? thread.loopRunCount
      : Array.isArray(thread.loopRuns) ? thread.loopRuns.length : 0;
    const memoryCount = Number.isSafeInteger(thread.memoryCount)
      ? thread.memoryCount
      : 0;
    meta.textContent = `${messageCount} ${
      messageCount === 1 ? "message" : "messages"
    } · ${runCount} ${runCount === 1 ? "run" : "runs"} · ${memoryCount} ${
      memoryCount === 1 ? "memory" : "memories"
    }`;

    button.append(title, meta);
    elements.threadList.append(button);
  }
}

function memoryCountLabel(count) {
  const value = nonNegativeInteger(count);
  return `${value} ${plural(value, "memory", "memories")} indexed`;
}

function runMemoryLabel(summary) {
  const data = summary && typeof summary === "object" ? summary : {};
  const recentCount = nonNegativeInteger(data.conversation_context_count);
  const semanticCount = nonNegativeInteger(data.semantic_memory_count);
  const semanticStatus = String(data.semantic_memory_status || "not_requested");
  const parts = [];
  if (recentCount) {
    parts.push(`${recentCount} recent ${plural(recentCount, "turn")}`);
  }
  if (semanticStatus === "retrieved" && semanticCount) {
    parts.push(`${semanticCount} recalled ${plural(semanticCount, "memory", "memories")}`);
  }
  if (parts.length) {
    return `last run used ${parts.join(" + ")}`;
  }
  if (semanticStatus === "retrieved") {
    return "last run found no matching memory";
  }
  if (semanticStatus === "empty") {
    return "last run found no matching memory";
  }
  if (semanticStatus === "unavailable") {
    return "last run memory unavailable";
  }
  return "last run did not use thread memory";
}

function renderActiveThreadTitle() {
  const thread = activeThread();
  elements.activeThreadTitle.textContent = thread.title || DEFAULT_THREAD_TITLE;
  elements.activeThreadMemory.textContent = `${memoryCountLabel(
    thread.memoryCount,
  )} · ${runMemoryLabel(thread.latest?.summary)}`;
}

function renderMemoryStatus(payload) {
  const thread = activeThread();
  const summary = payload?.summary || {};
  elements.memoryStatus.replaceChildren();

  const label = document.createElement("span");
  label.textContent = "Thread memory";
  const indexed = document.createElement("strong");
  indexed.textContent = memoryCountLabel(thread.memoryCount);
  const lastRun = document.createElement("span");
  lastRun.textContent = runMemoryLabel(summary);

  elements.memoryStatus.append(label, indexed, lastRun);
}

async function switchThread(threadId) {
  if (!threadById(threadId)) {
    return;
  }
  state.activeThreadId = threadId;
  persistActiveThreadId();
  await loadThreadDetail(threadId);
  renderThreads();
  renderActiveThreadTitle();
  renderMessages();
  renderRuns(activeThread().loopRuns);
  renderLoopPayload(activeThread().latest || emptyLoopPayload());
}

async function startNewThread() {
  const thread = sanitizeThread(
    await requestJson("/api/threads", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ title: DEFAULT_THREAD_TITLE }),
    }),
  );
  upsertThread(thread, { moveToTop: true });
  state.activeThreadId = thread.id;
  persistActiveThreadId();
  renderThreads();
  renderActiveThreadTitle();
  renderMessages();
  renderRuns(thread.loopRuns);
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
    ["Max output", status.max_output_tokens ? `${status.max_output_tokens} tokens` : "unknown"],
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
    const content = renderMessageContent(message.content);
    bubble.append(role, content);
    const thinking = renderMessageThinking(message.thinking);
    if (thinking) {
      bubble.append(thinking);
    }
    elements.messages.append(bubble);
  }
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function appendTextWithLineBreaks(parent, text) {
  const lines = String(text || "").split("\n");
  for (const [index, line] of lines.entries()) {
    if (line) {
      const span = document.createElement("span");
      span.textContent = line;
      parent.append(span);
    }
    if (index < lines.length - 1) {
      parent.append(document.createElement("br"));
    }
  }
}

function normalizeMessageMarkdownStructure(text) {
  const compactLabeledListPattern =
    /(?:^|\s)1[.)]\s+(?:\*\*)?[^:\n]{1,80}:(?:\*\*)?/;
  const compactNumberedListPattern =
    /(?:^|:)\s*1[.)]\s+\S[^\n]{2,}?\s+2[.)]\s+\S/s;
  const listItemPattern = /(^|\s+)(\d{1,2})([.)])\s+(?=\S)/g;
  const listStartPattern =
    /(?:^|:)\s*1[.)]\s+(?=\S)|(?:^|\s)1[.)]\s+(?=(?:\*\*)?[^:\n]{1,80}:(?:\*\*)?)/;
  const splitSequentialListItems = (line) => {
    const startMatch = line.match(listStartPattern);
    if (!startMatch || startMatch.index === undefined) {
      return line;
    }
    const markerOffset = startMatch[0].search(/1[.)]\s+/);
    if (markerOffset < 0) {
      return line;
    }
    const listStart = startMatch.index + markerOffset;
    const prefix = line.slice(0, listStart);
    const body = line.slice(listStart);
    let expectedNextItem = null;
    const normalizedBody = body.replace(
      listItemPattern,
      (marker, spacing, rawNumber, punctuation, offset) => {
        const number = Number.parseInt(rawNumber, 10);
        const shouldStartList = number === 1;
        const shouldContinueList = expectedNextItem === number;
        if (shouldStartList) {
          expectedNextItem = 2;
        } else if (shouldContinueList) {
          expectedNextItem = number + 1;
        } else {
          return marker;
        }
        const separator = offset > 0 && !spacing.includes("\n") ? "\n" : spacing;
        return `${separator}${rawNumber}${punctuation} `;
      },
    );
    const listSeparator = prefix && !prefix.endsWith("\n") ? "\n" : "";
    return `${prefix}${listSeparator}${normalizedBody}`;
  };
  return String(text || "")
    .replace(/\r\n/g, "\n")
    .split(/(`[^`\n]+`)/g)
    .map((part) =>
      part.startsWith("`")
        ? part
        : compactLabeledListPattern.test(part) ||
            compactNumberedListPattern.test(part)
          ? part.split("\n").map(splitSequentialListItems).join("\n")
          : part,
    )
    .join("");
}

function appendInlineMarkdown(parent, text) {
  const inlinePattern = /`([^`\n]+)`|\*\*([^\n]+?)\*\*/g;
  let cursor = 0;
  for (const match of text.matchAll(inlinePattern)) {
    if (match.index > cursor) {
      appendTextWithLineBreaks(parent, text.slice(cursor, match.index));
    }
    if (match[1] !== undefined) {
      const code = document.createElement("code");
      code.className = "message-inline-code";
      code.textContent = match[1];
      parent.append(code);
    } else {
      const strong = document.createElement("strong");
      strong.textContent = match[2];
      parent.append(strong);
    }
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) {
    appendTextWithLineBreaks(parent, text.slice(cursor));
  }
}

function appendParagraph(container, lines) {
  const text = lines.join("\n").trim();
  if (!text) {
    return;
  }
  const paragraph = document.createElement("p");
  paragraph.className = "message-paragraph";
  appendInlineMarkdown(paragraph, text);
  container.append(paragraph);
}

function appendList(container, items, ordered) {
  if (!items.length) {
    return;
  }
  const list = document.createElement(ordered ? "ol" : "ul");
  list.className = "message-list";
  for (const item of items) {
    const element = document.createElement("li");
    appendInlineMarkdown(element, item);
    list.append(element);
  }
  container.append(list);
}

function appendTextBlocks(container, text) {
  const lines = normalizeMessageMarkdownStructure(text).split("\n");
  let paragraph = [];
  let listItems = [];
  let orderedList = false;

  const flushParagraph = () => {
    appendParagraph(container, paragraph);
    paragraph = [];
  };
  const flushList = () => {
    appendList(container, listItems, orderedList);
    listItems = [];
    orderedList = false;
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const unorderedMatch = line.match(/^\s*[-*]\s+(.+)$/);
    const orderedMatch = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unorderedMatch || orderedMatch) {
      flushParagraph();
      const nextOrdered = Boolean(orderedMatch);
      if (listItems.length && orderedList !== nextOrdered) {
        flushList();
      }
      orderedList = nextOrdered;
      listItems.push((orderedMatch || unorderedMatch)[1]);
      continue;
    }

    flushList();
    paragraph.push(line);
  }

  flushParagraph();
  flushList();
}

function appendCodeBlock(container, language, codeText) {
  const figure = document.createElement("figure");
  figure.className = "message-code-block";

  const normalizedLanguage = String(language || "").trim();
  if (normalizedLanguage) {
    const caption = document.createElement("figcaption");
    caption.className = "message-code-language";
    caption.textContent = normalizedLanguage;
    figure.append(caption);
  }

  const pre = document.createElement("pre");
  const code = document.createElement("code");
  code.textContent = String(codeText || "").replace(/^\n/, "").replace(/\s+$/, "");
  pre.append(code);
  figure.append(pre);
  container.append(figure);
}

function renderMessageContent(text) {
  const container = document.createElement("div");
  container.className = "message-content";
  const source = String(text || "");
  const fencePattern = /```([A-Za-z0-9_+#.-]*)[ \t]*\n?([\s\S]*?)```/g;
  let cursor = 0;

  for (const match of source.matchAll(fencePattern)) {
    if (match.index > cursor) {
      appendTextBlocks(container, source.slice(cursor, match.index));
    }
    appendCodeBlock(container, match[1], match[2]);
    cursor = match.index + match[0].length;
  }

  if (cursor < source.length) {
    appendTextBlocks(container, source.slice(cursor));
  }
  if (!container.children.length) {
    appendParagraph(container, [source]);
  }
  return container;
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

function renderRuns(runs) {
  const durableRuns = Array.isArray(runs) ? runs : [];
  elements.runList.replaceChildren();
  elements.runCount.textContent = `${durableRuns.length} stored`;
  if (!durableRuns.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No durable loop runs stored for this thread yet.";
    elements.runList.append(empty);
    return;
  }

  for (const run of durableRuns) {
    const row = document.createElement("article");
    row.className = "run-row";

    const title = document.createElement("strong");
    title.textContent = `${run.final_decision || "unknown"} · ${
      run.recipe_name || "Loop recipe"
    }`;

    const meta = document.createElement("span");
    const stepCount = Number.isSafeInteger(run.step_count) ? run.step_count : 0;
    meta.textContent = `${stepCount} ${stepCount === 1 ? "step" : "steps"} · ${
      run.context_provider || "none"
    } · ${run.backend || "backend"} · ${run.run_id}`;

    row.append(title, meta);
    elements.runList.append(row);
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
  renderMemoryStatus(payload);
  renderActiveThreadTitle();
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

async function loadRecipes() {
  const payload = await requestJson("/api/recipes");
  state.recipes = Array.isArray(payload?.recipes)
    ? payload.recipes.map(sanitizeRecipe).filter((recipe) => recipe.recipe_id)
    : [];
  const savedRecipeId = loadActiveRecipeId();
  const defaultRecipeId = String(payload?.default_recipe_id || "");
  const preferredRecipeId =
    savedRecipeId && state.recipes.some((recipe) => recipe.recipe_id === savedRecipeId)
      ? savedRecipeId
      : defaultRecipeId;
  state.activeRecipeId = state.recipes.some(
    (recipe) => recipe.recipe_id === preferredRecipeId,
  )
    ? preferredRecipeId
    : state.recipes[0]?.recipe_id || "";
  persistActiveRecipeId();
  if (state.activeRecipeId) {
    await loadRecipeDetail(state.activeRecipeId).catch(() => null);
  }
  renderRecipes();
}

function activeRecipe() {
  return (
    state.recipes.find((recipe) => recipe.recipe_id === state.activeRecipeId) ||
    state.recipes[0] ||
    null
  );
}

function upsertRecipe(recipe) {
  if (!recipe.recipe_id) {
    return;
  }
  const others = state.recipes.filter((item) => item.recipe_id !== recipe.recipe_id);
  state.recipes = [recipe, ...others].sort((left, right) => {
    if (left.is_default !== right.is_default) {
      return left.is_default ? -1 : 1;
    }
    return String(left.name).localeCompare(String(right.name));
  });
}

async function loadRecipeDetail(recipeId) {
  const recipe = sanitizeRecipe(
    await requestJson(`/api/recipes/${encodeURIComponent(recipeId)}`),
  );
  upsertRecipe(recipe);
  return recipe;
}

function recipeFields() {
  return [
    elements.recipeName,
    elements.recipeGoal,
    elements.recipeInstructions,
    elements.recipeCriteria,
    elements.recipeStop,
  ];
}

function renderRecipeEditor() {
  const recipe = state.recipeDraft || activeRecipe();
  const isDraft = Boolean(state.recipeDraft);
  const isDefault = Boolean(recipe?.is_default);
  const detailLoaded = isDraft || isDefault || Boolean(recipe?.detail_loaded);
  elements.recipeEditorState.textContent = isDraft
    ? "new"
    : isDefault
      ? "default"
      : detailLoaded
        ? "custom"
        : "loading";
  elements.recipeName.value = recipe?.name || "";
  elements.recipeGoal.value = recipe?.goal || "";
  elements.recipeInstructions.value = recipe?.instructions || "";
  elements.recipeCriteria.value = Array.isArray(recipe?.success_criteria)
    ? recipe.success_criteria.join("\n")
    : "";
  elements.recipeStop.value = recipe?.stop_condition || "";
  for (const field of recipeFields()) {
    field.readOnly = !isDraft && (isDefault || !detailLoaded);
  }
  elements.recipeSaveButton.disabled =
    !recipe || (!isDraft && (isDefault || !detailLoaded));
  elements.recipeDeleteButton.disabled = !recipe || isDraft || isDefault;
  elements.recipeExportButton.disabled = !recipe || isDraft;
}

function renderRecipes() {
  elements.recipeSelect.replaceChildren();
  for (const recipe of state.recipes) {
    const option = document.createElement("option");
    option.value = recipe.recipe_id;
    option.textContent = recipe.name;
    elements.recipeSelect.append(option);
  }
  elements.recipeSelect.value = state.activeRecipeId || "";
  const recipe = activeRecipe();
  elements.recipeState.textContent = recipe?.is_default ? "default" : "custom";
  elements.recipeStatus.textContent = recipe
    ? safeText(recipe.goal || recipe.description || recipe.name, "", 180)
    : "No loop recipe available.";
  renderRecipeEditor();
}

function recipePayloadFromForm() {
  const basis = state.recipeDraft || activeRecipe() || {};
  return {
    name: elements.recipeName.value,
    goal: elements.recipeGoal.value,
    instructions: elements.recipeInstructions.value,
    success_criteria: elements.recipeCriteria.value
      .split(/\r?\n/)
      .map((criterion) => criterion.trim())
      .filter(Boolean),
    stop_condition: elements.recipeStop.value,
    context_provider: basis.context_provider || "auto",
    model_profile: basis.model_profile || "quality",
    verifier: basis.verifier || "default",
    metadata: basis.metadata || {},
  };
}

function startNewRecipe() {
  state.recipeDraft = {
    recipe_id: "",
    name: "",
    goal: "",
    instructions: "",
    success_criteria: [],
    stop_condition: "",
    context_provider: "auto",
    model_profile: "quality",
    verifier: "default",
    is_default: false,
  };
  elements.recipeEditor.open = true;
  elements.recipeStatus.textContent = "Editing new custom recipe.";
  renderRecipeEditor();
  elements.recipeName.focus();
}

async function saveRecipe() {
  let active = activeRecipe();
  const draft = state.recipeDraft;
  setBusy(elements.recipeSaveButton, true, "Save");
  try {
    if (!draft && active && !active.detail_loaded) {
      elements.recipeStatus.textContent = "Loading recipe details...";
      active = await loadRecipeDetail(active.recipe_id);
      renderRecipes();
    }
    const payload = recipePayloadFromForm();
    const saved = sanitizeRecipe(
      await requestJson(
        draft
          ? "/api/recipes"
          : `/api/recipes/${encodeURIComponent(active.recipe_id)}`,
        {
          method: draft ? "POST" : "PATCH",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        },
      ),
    );
    state.recipeDraft = null;
    upsertRecipe(saved);
    state.activeRecipeId = saved.recipe_id;
    persistActiveRecipeId();
    renderRecipes();
    elements.recipeStatus.textContent = `Saved ${saved.name}.`;
  } catch (error) {
    elements.recipeStatus.textContent = error.message;
  } finally {
    setBusy(elements.recipeSaveButton, false, "Save");
    renderRecipeEditor();
  }
}

async function deleteRecipe() {
  const recipe = activeRecipe();
  if (!recipe || recipe.is_default) {
    return;
  }
  const confirmed =
    typeof globalThis.confirm === "function"
      ? globalThis.confirm(`Delete ${recipe.name}?`)
      : true;
  if (!confirmed) {
    return;
  }
  setBusy(elements.recipeDeleteButton, true, "Delete");
  try {
    await requestJson(`/api/recipes/${encodeURIComponent(recipe.recipe_id)}`, {
      method: "DELETE",
    });
    state.recipes = state.recipes.filter(
      (item) => item.recipe_id !== recipe.recipe_id,
    );
    state.activeRecipeId = state.recipes[0]?.recipe_id || "";
    persistActiveRecipeId();
    if (state.activeRecipeId) {
      await loadRecipeDetail(state.activeRecipeId).catch(() => null);
    }
    renderRecipes();
    elements.recipeStatus.textContent = `Deleted ${recipe.name}.`;
  } catch (error) {
    elements.recipeStatus.textContent = error.message;
  } finally {
    setBusy(elements.recipeDeleteButton, false, "Delete");
    renderRecipeEditor();
  }
}

function downloadRecipe(recipe) {
  const blob = new Blob([JSON.stringify(recipe, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${recipe.recipe_id || "loop-recipe"}.json`;
  document.body?.append(link);
  link.click();
  link.remove?.();
  URL.revokeObjectURL(url);
}

async function exportRecipe() {
  const recipe = activeRecipe();
  if (!recipe) {
    return;
  }
  try {
    const exported = await requestJson(
      `/api/recipes/${encodeURIComponent(recipe.recipe_id)}/export`,
    );
    downloadRecipe(exported);
    elements.recipeStatus.textContent = `Exported ${recipe.name}.`;
  } catch (error) {
    elements.recipeStatus.textContent = error.message;
  }
}

async function importRecipe() {
  const file = elements.recipeImport.files?.[0];
  if (!file) {
    return;
  }
  try {
    const raw = JSON.parse(await file.text());
    const imported = sanitizeRecipe(
      await requestJson("/api/recipes", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(raw),
      }),
    );
    state.recipeDraft = null;
    upsertRecipe(imported);
    state.activeRecipeId = imported.recipe_id;
    persistActiveRecipeId();
    renderRecipes();
    elements.recipeStatus.textContent = `Imported ${imported.name}.`;
  } catch (error) {
    elements.recipeStatus.textContent = error.message;
  } finally {
    elements.recipeImport.value = "";
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
      body: JSON.stringify({
        message,
        session_id: requestThreadId,
        recipe_id: state.activeRecipeId || undefined,
      }),
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
    if (result.run?.run_id) {
      const run = sanitizeLoopRun(result.run);
      const priorRunCount = Number.isSafeInteger(targetThread.loopRunCount)
        ? targetThread.loopRunCount
        : Array.isArray(targetThread.loopRuns) ? targetThread.loopRuns.length : 0;
      const hadRun = Array.isArray(targetThread.loopRuns)
        ? targetThread.loopRuns.some((item) => item.run_id === run.run_id)
        : false;
      targetThread.loopRuns = [
        run,
        ...(targetThread.loopRuns || []).filter((item) => item.run_id !== run.run_id),
      ];
      targetThread.loopRunCount = Math.max(
        hadRun ? priorRunCount : priorRunCount + 1,
        targetThread.loopRuns.length,
      );
    }
    if (result.thread) {
      applyThreadSummary(targetThread, sanitizeThread(result.thread));
    }
    targetThread.latest = result;
    touchThread(targetThread);
    renderThreads();
    if (state.activeThreadId === requestThreadId) {
      renderActiveThreadTitle();
      renderMessages();
      renderRuns(targetThread.loopRuns);
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
  thread.loopRuns = [];
  thread.loopRunCount = 0;
  thread.memoryCount = 0;
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
      renderRuns(targetThread.loopRuns);
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
  setupTabs();
  await loadRecipes();
  await loadThreads();
  renderThreads();
  renderActiveThreadTitle();
  renderMessages();
  renderRuns(activeThread().loopRuns);
  renderLoopPayload(activeThread().latest || emptyLoopPayload());
  elements.uploadForm.addEventListener("submit", uploadDocument);
  elements.queryForm.addEventListener("submit", runQuery);
  elements.newThreadButton.addEventListener("click", () =>
    startNewThread().catch((error) => {
      elements.uploadStatus.textContent = error.message;
    }),
  );
  elements.clearButton.addEventListener("click", clearChat);
  elements.refreshStatus.addEventListener("click", refreshStatus);
  elements.recipeSelect.addEventListener("change", () => {
    state.activeRecipeId = elements.recipeSelect.value;
    state.recipeDraft = null;
    persistActiveRecipeId();
    loadRecipeDetail(state.activeRecipeId)
      .then(renderRecipes)
      .catch((error) => {
        elements.recipeStatus.textContent = error.message;
        renderRecipeEditor();
      });
  });
  elements.recipeNewButton.addEventListener("click", startNewRecipe);
  elements.recipeSaveButton.addEventListener("click", saveRecipe);
  elements.recipeDeleteButton.addEventListener("click", deleteRecipe);
  elements.recipeExportButton.addEventListener("click", exportRecipe);
  elements.recipeImport.addEventListener("change", importRecipe);
  await loadConfig();
  await refreshStatus();
}

boot().catch((error) => {
  elements.uploadStatus.textContent = error.message;
});
