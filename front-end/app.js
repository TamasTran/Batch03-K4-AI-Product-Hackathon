const API_BASE_URL = (window.DATASCOUT_CONFIG?.API_BASE_URL || "http://localhost:8000/api/v1").replace(/\/$/, "");
let pendingClarificationContext = null;
const HISTORY_KEY = "datascout.searchHistory.v1";
const MAX_HISTORY_ITEMS = 20;
let lastConversationHtml = null;
const CLARIFICATION_FIELD_LABELS = {
  task_type: "Tác vụ",
  modality: "Loại dữ liệu",
  domain: "Chủ đề",
  required_language: "Ngôn ngữ",
  required_labels: "Nhãn dữ liệu",
};

const elements = {
  apiState: document.querySelector("#api-state"),
  configFacts: document.querySelector("#config-facts"),
  conversation: document.querySelector("#conversation"),
  form: document.querySelector("#search-form"),
  formError: document.querySelector("#form-error"),
  historyList: document.querySelector("#history-list"),
  limit: document.querySelector("#limit"),
  limitOutput: document.querySelector("#limit-output"),
  modelName: document.querySelector("#model-name"),
  query: document.querySelector("#query"),
  sidebar: document.querySelector("#sidebar"),
  sidebarScrim: document.querySelector("#sidebar-scrim"),
  submit: document.querySelector("#submit-button"),
  webFallback: document.querySelector("#web-fallback"),
  webSearchButton: document.querySelector("#web-search-button"),
};

const escapeHtml = (value = "") =>
  String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  })[character]);

const safeUrl = (value) => {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : "#";
  } catch {
    return "#";
  }
};

const formatValue = (value, fallback = "Không rõ") =>
  value === null || value === undefined || value === "" ? fallback : escapeHtml(value);

function setConnectionState(message, online) {
  const light = elements.apiState.querySelector("span:first-child");
  elements.apiState.querySelector("span:last-child").textContent = message;
  light.className = `size-1.5 rounded-full ${online ? "bg-success shadow-[0_0_10px_#22c55e]" : "bg-signal-bright"}`;
}

async function loadConfig() {
  try {
    const response = await fetch(`${API_BASE_URL}/config`);
    if (!response.ok) throw new Error("API config unavailable");
    const config = await response.json();
    setConnectionState("API sẵn sàng", true);
    elements.modelName.textContent = config.llm_enabled ? config.llm_model : "Transparent heuristic";
    elements.configFacts.innerHTML = `
      <div class="flex justify-between gap-4"><span>LLM engine</span><b class="text-ink">${config.llm_enabled ? escapeHtml(config.llm_model) : "Heuristic"}</b></div>
      <div class="flex justify-between gap-4"><span>Kaggle API</span><b class="text-ink">${config.kaggle_configured ? "Sẵn sàng" : "Chưa có key"}</b></div>
      <div class="flex justify-between gap-4"><span>Web search</span><b class="text-ink">${config.web_search_configured ? "Sẵn sàng" : "Chưa có key"}</b></div>
    `;
  } catch {
    setConnectionState("API chưa kết nối", false);
    elements.modelName.textContent = "Backend offline";
    elements.configFacts.innerHTML = "<p>Chạy run_backend.py tại thư mục gốc rồi tải lại trang.</p>";
  }
}

function selectedSources() {
  return [...document.querySelectorAll('input[name="source"]:checked')].map((input) => input.value);
}

function readHistory() {
  try {
    const value = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

function writeHistory(items) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, MAX_HISTORY_ITEMS)));
  } catch {
    // Searching must still work when storage is blocked or full.
  }
  renderHistoryList();
}

function rememberSearch(query, data) {
  const entry = {
    id: Date.now(),
    query,
    createdAt: new Date().toISOString(),
    verified: data.verified || [],
    unverified: data.unverified || [],
  };
  writeHistory([entry, ...readHistory().filter((item) => item.query !== query)]);
}

function renderHistoryList() {
  const history = readHistory();
  elements.historyList.innerHTML = history.length
    ? history.slice(0, 6).map((item) => `
      <button type="button" class="history-item" data-history-id="${item.id}">
        <span>${escapeHtml(item.query)}</span>
        <small>${(item.verified?.length || 0) + (item.unverified?.length || 0)} kết quả</small>
      </button>`).join("")
    : '<p class="thread-empty">Chưa có lượt tìm kiếm nào.</p>';
}

function closeSidebar() {
  elements.sidebar.classList.add("-translate-x-full");
  elements.sidebarScrim.classList.add("hidden");
}

function openSidebar() {
  elements.sidebar.classList.remove("-translate-x-full");
  elements.sidebarScrim.classList.remove("hidden");
}

function resetConversation() {
  pendingClarificationContext = null;
  lastConversationHtml = null;
  elements.query.value = "";
  elements.query.placeholder = "Mô tả dataset bạn cần tìm...";
  elements.conversation.innerHTML = document.querySelector("#welcome-template").innerHTML;
  autoResizeComposer();
  setActiveView("assistant");
  elements.query.focus();
}

function setActiveView(view) {
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  closeSidebar();
}

function showHistory() {
  const history = readHistory();
  lastConversationHtml ||= elements.conversation.innerHTML;
  elements.conversation.innerHTML = `
    <div class="assistant-block">
      <div class="lane-heading"><div><h1>Lịch sử tìm kiếm</h1><p class="lane-copy">Được lưu cục bộ trên trình duyệt này.</p></div><span>${history.length}</span></div>
      ${history.length ? history.map((item) => `
        <button type="button" class="history-result" data-history-id="${item.id}">
          <span>${escapeHtml(item.query)}</span>
          <small>${new Date(item.createdAt).toLocaleString("vi-VN")} · ${(item.verified?.length || 0) + (item.unverified?.length || 0)} kết quả</small>
        </button>`).join("") : '<div class="lane-empty">Chưa có lịch sử. Hãy thực hiện một lượt tìm kiếm.</div>'}
    </div>`;
  setActiveView("history");
}

function showLibrary() {
  const saved = readHistory().flatMap((entry) => [
    ...(entry.verified || []).map((item) => ({ ...item, confidence: "verified" })),
    ...(entry.unverified || []).map((item) => ({ ...item, confidence: "unverified" })),
  ]);
  const unique = [...new Map(saved.map((item) => [item.url || item.id || item.title, item])).values()];
  lastConversationHtml ||= elements.conversation.innerHTML;
  elements.conversation.innerHTML = `
    <div class="assistant-block">
      <div class="lane-heading"><div><h1>Dataset library</h1><p class="lane-copy">Các dataset từ những lượt tìm kiếm đã lưu.</p></div><span>${unique.length}</span></div>
      ${unique.length ? unique.map((item) => item.confidence === "verified" ? verifiedCard(item) : unverifiedCard(item)).join("") : '<div class="lane-empty">Library đang trống. Kết quả tìm kiếm sẽ tự động xuất hiện tại đây.</div>'}
    </div>`;
  setActiveView("library");
}

function restoreAssistant() {
  if (lastConversationHtml) elements.conversation.innerHTML = lastConversationHtml;
  lastConversationHtml = null;
  setActiveView("assistant");
}

function openHistoryItem(id) {
  const item = readHistory().find((entry) => String(entry.id) === String(id));
  if (!item) return;
  renderResults({ verified: item.verified, unverified: item.unverified, intent: {}, guidance: {}, errors: [], tool_events: [], parse_mode: "saved", rank_mode: "saved" }, item.query);
  lastConversationHtml = elements.conversation.innerHTML;
  setActiveView("assistant");
}

function syncWebSearchButton() {
  const enabled = elements.webFallback.checked;
  elements.webSearchButton.setAttribute("aria-pressed", String(enabled));
  elements.webSearchButton.classList.toggle("active", enabled);
  elements.webSearchButton.querySelector("span").textContent = enabled ? "Web on" : "Web off";
}

function autoResizeComposer() {
  elements.query.style.height = "auto";
  elements.query.style.height = `${Math.min(elements.query.scrollHeight, 160)}px`;
}

function scoreCell(label, value) {
  return `<div class="score-cell"><span>${escapeHtml(label)}</span><b>${formatValue(value)}/5</b></div>`;
}

function sourceLinks(item) {
  const sources = (item.sources || []).filter((source) => source.url && source.url !== item.url);
  if (!sources.length) return "";
  return `<div class="dataset-meta"><span>Cũng có trên ${sources.map((source) =>
    `<a class="text-signal hover:underline" href="${safeUrl(source.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(source.source)}</a>`
  ).join(", ")}</span></div>`;
}

function duplicateNote(item) {
  if (!item.possible_duplicate_of?.length) return "";
  return `<p class="mt-2 text-[11px] leading-5 text-signal-bright">Có thể trùng với ${escapeHtml(item.possible_duplicate_of.join(", "))}. Hãy kiểm tra trước khi chọn cả hai.</p>`;
}

function verifiedCard(item) {
  return `
    <article class="dataset-card">
      <div>
        <p class="dataset-source">${escapeHtml(item.source || "Verified registry")}</p>
        <h3><a href="${safeUrl(item.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.id || item.title || item.url)}</a></h3>
        <p class="dataset-reasoning">${escapeHtml(item.reasoning || "Không có giải thích xếp hạng.")}</p>
        ${item.review_warning ? `<p class="mt-2 rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-[12px] leading-5 text-amber-200">${escapeHtml(item.review_warning)}</p>` : ""}
        ${duplicateNote(item)}
        ${sourceLinks(item)}
        <div class="dataset-meta">
          <span>License <b>${formatValue(item.license)}</b></span>
          <span>Access <b>${formatValue(item.access_type)}</b></span>
          <span>Downloads <b>${formatValue(item.downloads)}</b></span>
          <span>Likes <b>${formatValue(item.likes)}</b></span>
        </div>
        <div class="score-grid">
          ${scoreCell("Task", item.task_match)}
          ${scoreCell("Domain", item.domain_fit)}
          ${scoreCell("Nhãn", item.label_overlap)}
          ${scoreCell("Quy mô", item.size_adequacy)}
        </div>
      </div>
      <div class="dataset-score"><b>${formatValue(item.total_score)}</b><span>phù hợp</span></div>
    </article>
  `;
}

function unverifiedCard(item) {
  return `
    <article class="dataset-card">
      <div>
        <p class="dataset-source">${escapeHtml(item.source || "Web discovery")} · Unverified</p>
        <h3><a href="${safeUrl(item.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.title || item.url)}</a></h3>
        <p class="dataset-reasoning">${escapeHtml(item.snippet || "Search API không trả snippet.")}</p>
        <p class="mt-2 text-[11px] leading-5 text-accent">${escapeHtml(item.reasoning || "Kết quả này chưa được xác minh.")}</p>
        ${duplicateNote(item)}
        ${sourceLinks(item)}
        <div class="score-grid">
          ${scoreCell("Task", item.task_match)}
          ${scoreCell("Domain", item.domain_fit)}
        </div>
      </div>
      <div class="dataset-score"><b>${formatValue(item.preliminary_score)}</b><span>khớp sơ bộ</span></div>
    </article>
  `;
}

function lane(title, subtitle, items, renderer, emptyText) {
  return `
    <section class="lane">
      <div class="lane-heading">
        <div><h2>${escapeHtml(title)}</h2><p class="lane-copy">${escapeHtml(subtitle)}</p></div>
        <span>${items.length.toString().padStart(2, "0")}</span>
      </div>
      ${items.length ? items.map(renderer).join("") : `<div class="lane-empty">${escapeHtml(emptyText)}</div>`}
    </section>
  `;
}

function renderThinking(query) {
  elements.conversation.innerHTML = `
    <div class="message-user">${escapeHtml(query)}</div>
    <div class="assistant-block">
      <div class="assistant-header">
        <span class="assistant-avatar"><i class="ph ph-sparkle" aria-hidden="true"></i></span>
        <div><p class="assistant-title">DataScout</p><p class="assistant-subtitle">Đang phân tích và kiểm chứng nguồn</p></div>
      </div>
      <div class="thinking-row" role="status" aria-label="Agent đang tìm kiếm">
        <div class="thinking-line"></div><div class="thinking-line"></div><div class="thinking-line"></div>
      </div>
    </div>
  `;
}

function renderClarification(data, query) {
  const missing = (data.missing_fields || [])
    .map((field) => `<span class="clarification-field">${escapeHtml(CLARIFICATION_FIELD_LABELS[field] || field)}</span>`)
    .join("");
  elements.conversation.innerHTML = `
    <div class="message-user">${escapeHtml(query)}</div>
    <div class="assistant-block">
      <div class="assistant-header">
        <span class="assistant-avatar bg-accent"><i class="ph ph-question" aria-hidden="true"></i></span>
        <div><p class="assistant-title">DataScout</p><p class="assistant-subtitle">Cần thêm thông tin trước khi tìm kiếm</p></div>
      </div>
      <div class="clarification-box">
        <p>${escapeHtml(data.clarification_question || "Bạn có thể mô tả rõ hơn tác vụ cần thực hiện không?")}</p>
        ${missing ? `<div class="clarification-fields"><span>Còn thiếu</span>${missing}</div>` : ""}
      </div>
    </div>
  `;
  elements.query.value = "";
  elements.query.placeholder = "Trả lời câu hỏi làm rõ...";
  autoResizeComposer();
  elements.query.focus();
}

function renderResults(data, query) {
  const errors = data.errors?.length
    ? `<div class="notice error">${data.errors.length} nguồn gặp lỗi. Pipeline đã tiếp tục với các nguồn còn lại.</div>`
    : "";
  const sensitive = data.intent?.is_sensitive_domain
    ? `<div class="notice error"><b>Dữ liệu nhạy cảm.</b> ${escapeHtml(data.intent.sensitive_reason || "")} Kiểm tra ethics và IRB trước khi sử dụng.</div>`
    : "";
  const guidance = [...(data.guidance?.alternatives || []), ...(data.guidance?.registries || [])];
  const guidanceHtml = guidance.length
    ? `<div class="guidance-box"><b class="text-ink">Phương án tiếp theo</b>${guidance.map((item) => `<p class="mt-2">${escapeHtml(item)}</p>`).join("")}</div>`
    : "";
  const trace = `
    <details class="trace-box">
      <summary>Agent trace · Parse ${escapeHtml(data.parse_mode)} · Rank ${escapeHtml(data.rank_mode)}</summary>
      <pre>${escapeHtml(JSON.stringify({ intent: data.intent, tool_events: data.tool_events }, null, 2))}</pre>
    </details>
  `;

  elements.conversation.innerHTML = `
    <div class="message-user">${escapeHtml(query)}</div>
    <div class="assistant-block">
      <div class="assistant-header">
        <span class="assistant-avatar"><i class="ph ph-sparkle" aria-hidden="true"></i></span>
        <div><p class="assistant-title">DataScout</p><p class="assistant-subtitle">Search completed</p></div>
      </div>
      <p class="result-intro">Tôi đã tách kết quả thành hai evidence lane. Nguồn verified có metadata từ registry; web discovery cần bạn mở link và tự xác minh.</p>
      ${errors}
      ${sensitive}
      ${trace}
      ${lane(
        "Nguồn đã xác minh",
        "Metadata trực tiếp từ registry và được kiểm tra theo ràng buộc.",
        data.verified || [],
        verifiedCard,
        "Không tìm thấy candidate verified phù hợp trong lượt này."
      )}
      ${lane(
        "Gợi ý cần tự kiểm tra",
        "Web search chỉ cung cấp title và snippet.",
        data.unverified || [],
        unverifiedCard,
        "Catch-all chưa trả kết quả hoặc chưa được bật."
      )}
      ${guidanceHtml}
    </div>
  `;
}

function renderError(message, query) {
  elements.conversation.innerHTML = `
    <div class="message-user">${escapeHtml(query)}</div>
    <div class="assistant-block">
      <div class="assistant-header">
        <span class="assistant-avatar bg-signal-bright"><i class="ph ph-warning" aria-hidden="true"></i></span>
        <div><p class="assistant-title">Không thể hoàn tất</p><p class="assistant-subtitle">Request failed</p></div>
      </div>
      <div class="notice error">${escapeHtml(message)} Kiểm tra backend hoặc API key rồi thử lại.</div>
    </div>
  `;
}

async function submitSearch(query) {
  elements.formError.textContent = "";
  const enabledSources = selectedSources();
  if (!query) {
    elements.formError.textContent = "Hãy mô tả dataset bạn cần tìm.";
    elements.query.focus();
    return;
  }
  if (!enabledSources.length && !elements.webFallback.checked) {
    elements.formError.textContent = "Hãy bật ít nhất một nguồn dữ liệu.";
    return;
  }

  elements.submit.disabled = true;
  renderThinking(query);
  try {
    const response = await fetch(`${API_BASE_URL}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        clarification_context: pendingClarificationContext,
        enabled_sources: enabledSources,
        web_fallback_enabled: elements.webFallback.checked,
        limit: Number(elements.limit.value),
      }),
    });
    const body = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(body.detail) ? body.detail.map((item) => item.msg).join(". ") : body.detail;
      throw new Error(detail || `API trả mã ${response.status}`);
    }
    if (body.status === "clarification_required") {
      pendingClarificationContext = body.clarification_context || query;
      renderClarification(body, query);
      return;
    }
    pendingClarificationContext = null;
    elements.query.placeholder = "Mô tả dataset bạn cần tìm...";
    renderResults(body, query);
    rememberSearch(query, body);
    lastConversationHtml = elements.conversation.innerHTML;
  } catch (error) {
    renderError(error.message || "Lỗi kết nối không xác định.", query);
  } finally {
    elements.submit.disabled = false;
  }
}

document.querySelector("#open-sidebar").addEventListener("click", openSidebar);
document.querySelector("#close-sidebar").addEventListener("click", closeSidebar);
elements.sidebarScrim.addEventListener("click", closeSidebar);
document.querySelector("#new-search").addEventListener("click", resetConversation);
const openSettings = () => document.querySelector("#settings-dialog").showModal();
document.querySelector("#settings-button").addEventListener("click", openSettings);
document.querySelector("#tools-button").addEventListener("click", openSettings);
document.querySelector("#model-card").addEventListener("click", openSettings);
document.querySelector("#workspace-button").addEventListener("click", openSettings);
elements.webSearchButton.addEventListener("click", () => {
  elements.webFallback.checked = !elements.webFallback.checked;
  syncWebSearchButton();
});
elements.webFallback.addEventListener("change", syncWebSearchButton);
document.querySelector("#clear-history").addEventListener("click", () => {
  writeHistory([]);
  if (document.querySelector('[data-view="history"]').classList.contains("active")) showHistory();
});
document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => {
    if (button.dataset.view === "library") showLibrary();
    else if (button.dataset.view === "history") showHistory();
    else restoreAssistant();
  });
});
document.addEventListener("click", (event) => {
  const historyButton = event.target.closest("[data-history-id]");
  if (historyButton) openHistoryItem(historyButton.dataset.historyId);
});
elements.limit.addEventListener("input", () => { elements.limitOutput.value = elements.limit.value; });
elements.query.addEventListener("input", autoResizeComposer);
elements.query.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});
elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitSearch(elements.query.value.trim());
});
document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    elements.query.value = button.dataset.prompt;
    autoResizeComposer();
    elements.query.focus();
  });
});
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    elements.query.focus();
  }
});

document.body.insertAdjacentHTML("beforeend", `<template id="welcome-template">${document.querySelector("#conversation").innerHTML}</template>`);
renderHistoryList();
syncWebSearchButton();
loadConfig();
