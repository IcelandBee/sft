const state = {
  token: new URLSearchParams(location.search).get("token") || sessionStorage.getItem("auditToken") || "",
  records: [],
  annotations: {},
  filtered: [],
  index: 0,
  saving: false,
};

if (state.token) sessionStorage.setItem("auditToken", state.token);

const $ = (id) => document.getElementById(id);
const elements = {
  progressText: $("progressText"), progressBar: $("progressBar"), saveState: $("saveState"),
  filter: $("filterSelect"), export: $("exportButton"), counter: $("sampleCounter"),
  badge: $("groupBadge"), image: $("reviewImage"), stage: $("imageStage"), loading: $("loadingState"),
  path: $("imagePath"), openImage: $("openImageButton"), previous: $("previousButton"), next: $("nextButton"),
  goldCard: $("goldCard"), e1Card: $("e1Card"), e2Card: $("e2Card"),
  goldDecision: $("goldDecision"), goldCategories: $("goldCategories"), goldReasons: $("goldReasons"),
  e1Decision: $("e1Decision"), e1Categories: $("e1Categories"), e1Reasons: $("e1Reasons"),
  e2Decision: $("e2Decision"), e2Categories: $("e2Categories"), e2Reasons: $("e2Reasons"),
  form: $("reviewForm"), category: $("categorySelect"), notes: $("notesInput"), save: $("saveButton"),
  hint: $("completionHint"), toast: $("toast"),
};

function authHeaders(extra = {}) {
  return { "X-Audit-Token": state.token, ...extra };
}

function apiUrl(path) {
  const join = path.includes("?") ? "&" : "?";
  return `${path}${join}token=${encodeURIComponent(state.token)}`;
}

function current() { return state.filtered[state.index]; }
function annotationFor(record) { return state.annotations[String(record.row)] || {}; }
function isComplete(annotation) { return annotation.completed === true; }

function payloadText(payload, field) {
  const values = payload && Array.isArray(payload[field]) ? payload[field] : [];
  return values.length ? values.join(" · ") : "无";
}

function modelPayload(model) {
  return model && model.payload ? model.payload : { decision: "INVALID", categories: [], reasons: [] };
}

function setCard(card, decisionEl, categoriesEl, reasonsEl, payload) {
  decisionEl.textContent = payload.decision || "INVALID";
  categoriesEl.textContent = payloadText(payload, "categories");
  reasonsEl.textContent = payloadText(payload, "reasons");
  card.classList.toggle("bad", payload.decision === "BAD");
}

function setRadio(name, value) {
  document.querySelectorAll(`input[name="${name}"]`).forEach((input) => {
    input.checked = input.value === value;
  });
}

function formValue() {
  const checked = (name) => document.querySelector(`input[name="${name}"]:checked`)?.value || "";
  return {
    label_status: checked("label_status"),
    visible_severity: checked("visible_severity"),
    review_decision: checked("review_decision"),
    primary_category: elements.category.value,
    notes: elements.notes.value.trim(),
  };
}

function formComplete(value = formValue()) {
  return Boolean(value.label_status && value.visible_severity && value.review_decision);
}

function populateForm(annotation) {
  setRadio("label_status", annotation.label_status || "");
  setRadio("visible_severity", annotation.visible_severity || "");
  setRadio("review_decision", annotation.review_decision || "");
  elements.category.value = annotation.primary_category || "";
  elements.notes.value = annotation.notes || "";
  updateCompletionHint();
}

function updateCompletionHint() {
  const complete = formComplete();
  elements.hint.textContent = complete ? "必填项已完成，保存后计入进度" : "完成 3 个必填项后计入进度";
  elements.hint.style.color = complete ? "#185c45" : "";
}

function progress() {
  const done = Object.values(state.annotations).filter(isComplete).length;
  elements.progressText.textContent = `${done} / ${state.records.length} 已完成`;
  elements.progressBar.style.width = `${state.records.length ? (done / state.records.length) * 100 : 0}%`;
}

function applyFilter(keepRow = null) {
  const filter = elements.filter.value;
  state.filtered = state.records.filter((record) => {
    if (filter === "unreviewed") return !isComplete(annotationFor(record));
    if (filter === "disagreement") return record.decision_disagreement === true;
    if (filter === "both_wrong") return record.review_group === "both_wrong";
    return true;
  });
  const preserved = keepRow ? state.filtered.findIndex((record) => record.row === keepRow) : -1;
  state.index = preserved >= 0 ? preserved : Math.min(state.index, Math.max(0, state.filtered.length - 1));
  render();
}

function imageUrl(record) { return apiUrl(`/api/image/${record.row}`); }

function render() {
  progress();
  const record = current();
  if (!record) {
    elements.counter.textContent = "0 / 0";
    elements.badge.textContent = elements.filter.value === "unreviewed" ? "全部完成" : "没有匹配样本";
    elements.loading.textContent = "当前筛选条件下没有样本";
    elements.loading.style.display = "block";
    elements.image.style.display = "none";
    elements.form.hidden = true;
    return;
  }
  elements.form.hidden = false;
  elements.counter.textContent = `${state.index + 1} / ${state.filtered.length} · Dev #${record.row}`;
  const disagreement = record.decision_disagreement === true;
  elements.badge.textContent = disagreement ? "E1 / E2 决策分歧" : "E1 / E2 共同错误";
  elements.badge.className = `badge ${disagreement ? "disagreement" : "both-wrong"}`;
  elements.path.textContent = record.image_path;
  elements.path.title = record.image_path;
  elements.loading.textContent = "正在载入图片…";
  elements.loading.style.display = "block";
  elements.image.style.display = "none";
  elements.image.classList.remove("zoomed");
  elements.image.src = imageUrl(record);

  setCard(elements.goldCard, elements.goldDecision, elements.goldCategories, elements.goldReasons, record.gold);
  setCard(elements.e1Card, elements.e1Decision, elements.e1Categories, elements.e1Reasons, modelPayload(record.e1));
  setCard(elements.e2Card, elements.e2Decision, elements.e2Categories, elements.e2Reasons, modelPayload(record.e2));
  populateForm(annotationFor(record));
  elements.previous.disabled = state.index === 0;
  elements.next.disabled = state.index >= state.filtered.length - 1;
  elements.save.textContent = isComplete(annotationFor(record)) ? "更新并下一张" : "保存并下一张";
}

function move(delta) {
  if (!state.filtered.length) return;
  state.index = Math.max(0, Math.min(state.filtered.length - 1, state.index + delta));
  render();
  elements.stage.scrollTo({ top: 0, left: 0 });
}

let toastTimer;
function toast(message, error = false) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.className = `toast show${error ? " error" : ""}`;
  toastTimer = setTimeout(() => { elements.toast.className = "toast"; }, 2200);
}

async function saveCurrent(advance = true) {
  const record = current();
  if (!record || state.saving) return;
  state.saving = true;
  elements.save.disabled = true;
  elements.saveState.textContent = "正在保存…";
  try {
    const response = await fetch(apiUrl(`/api/annotation/${record.row}`), {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(formValue()),
    });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || "保存失败");
    state.annotations[String(record.row)] = value.annotation;
    elements.saveState.textContent = `Dev #${record.row} 已保存`;
    toast(value.annotation.completed ? "已保存并计入进度" : "草稿已保存，必填项尚未完成");
    progress();
    if (advance) {
      if (elements.filter.value === "unreviewed" && value.annotation.completed) {
        applyFilter();
      } else if (state.index < state.filtered.length - 1) {
        move(1);
      } else {
        render();
      }
    }
  } catch (error) {
    elements.saveState.textContent = "保存失败";
    toast(error.message, true);
  } finally {
    state.saving = false;
    elements.save.disabled = false;
  }
}

async function initialize() {
  if (!state.token) {
    elements.loading.textContent = "访问链接缺少 token，请使用服务启动时打印的完整地址。";
    elements.saveState.textContent = "未授权";
    return;
  }
  try {
    const response = await fetch("/api/state", { headers: authHeaders() });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || "载入失败");
    state.records = value.records;
    state.annotations = value.annotations || {};
    elements.saveState.textContent = "已载入";
    applyFilter();
  } catch (error) {
    elements.loading.textContent = `无法载入：${error.message}`;
    elements.saveState.textContent = "载入失败";
  }
}

elements.image.addEventListener("load", () => {
  elements.loading.style.display = "none";
  elements.image.style.display = "block";
});
elements.image.addEventListener("error", () => {
  elements.loading.textContent = "图片载入失败，请检查服务器上的原图路径。";
  elements.loading.style.display = "block";
});
elements.image.addEventListener("click", () => elements.image.classList.toggle("zoomed"));
elements.openImage.addEventListener("click", () => { if (current()) window.open(imageUrl(current()), "_blank", "noopener"); });
elements.previous.addEventListener("click", () => move(-1));
elements.next.addEventListener("click", () => move(1));
elements.filter.addEventListener("change", () => applyFilter(current()?.row));
elements.form.addEventListener("change", updateCompletionHint);
elements.form.addEventListener("submit", (event) => { event.preventDefault(); saveCurrent(true); });
elements.export.addEventListener("click", () => { location.href = apiUrl("/api/export.csv"); });
document.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter") { event.preventDefault(); saveCurrent(true); return; }
  const editing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
  if (editing) return;
  if (event.key === "ArrowLeft" || event.key.toLowerCase() === "k") move(-1);
  if (event.key === "ArrowRight" || event.key.toLowerCase() === "j") move(1);
});

initialize();
