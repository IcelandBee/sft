const query = new URLSearchParams(location.search);
const state = {
  token: query.get("token") || sessionStorage.getItem("reviewToken") || "",
  records: [], annotations: {}, filtered: [], index: 0, saving: false,
};
if (state.token) sessionStorage.setItem("reviewToken", state.token);
const $ = (id) => document.getElementById(id);
const el = {
  progressText: $("progressText"), progressBar: $("progressBar"), saveState: $("saveState"),
  filter: $("filterSelect"), export: $("exportButton"), counter: $("counter"), row: $("rowLabel"),
  image: $("image"), stage: $("stage"), loading: $("loading"), path: $("imagePath"),
  open: $("openButton"), previous: $("previousButton"), next: $("nextButton"),
  form: $("reviewForm"), notes: $("notes"), save: $("saveButton"), hint: $("hint"), toast: $("toast"),
};
const api = (path) => `${path}${path.includes("?") ? "&" : "?"}token=${encodeURIComponent(state.token)}`;
const current = () => state.filtered[state.index];
const annotation = (record) => state.annotations[String(record.row)] || {};
const complete = (value) => value?.completed === true;
const checked = (name) => document.querySelector(`input[name="${name}"]:checked`)?.value || "";

function formValue() {
  return {
    review_decision: checked("review_decision"),
    visible_severity: checked("visible_severity"),
    categories: [...document.querySelectorAll("#categories input:checked")].map((item) => item.value),
    notes: el.notes.value.trim(),
  };
}
function setRadio(name, value) {
  document.querySelectorAll(`input[name="${name}"]`).forEach((item) => { item.checked = item.value === value; });
}
function populate(value) {
  setRadio("review_decision", value.review_decision || "");
  setRadio("visible_severity", value.visible_severity || "");
  document.querySelectorAll("#categories input").forEach((item) => { item.checked = (value.categories || []).includes(item.value); });
  el.notes.value = value.notes || "";
  updateHint();
}
function updateHint() {
  const value = formValue();
  const ready = Boolean(value.review_decision && value.visible_severity);
  el.hint.textContent = ready ? "必填项已完成，保存后计入进度" : "填写两项必填内容后计入完成进度";
  el.hint.classList.toggle("ready", ready);
}
function updateProgress() {
  const done = Object.values(state.annotations).filter(complete).length;
  el.progressText.textContent = `${done} / ${state.records.length} 已完成`;
  el.progressBar.style.width = `${state.records.length ? done / state.records.length * 100 : 0}%`;
}
function applyFilter(keepRow = null) {
  const mode = el.filter.value;
  state.filtered = state.records.filter((record) => mode === "all" || (mode === "unreviewed" ? !complete(annotation(record)) : complete(annotation(record))));
  const preserved = keepRow ? state.filtered.findIndex((record) => record.row === keepRow) : -1;
  state.index = preserved >= 0 ? preserved : Math.min(state.index, Math.max(0, state.filtered.length - 1));
  render();
}
function render() {
  updateProgress();
  const record = current();
  if (!record) {
    el.counter.textContent = "0 / 0"; el.row.textContent = "当前筛选下没有样本";
    el.loading.textContent = el.filter.value === "unreviewed" ? "73 张已全部完成" : "没有匹配样本";
    el.loading.style.display = "block"; el.image.style.display = "none"; el.form.hidden = true; return;
  }
  el.form.hidden = false;
  el.counter.textContent = `${state.index + 1} / ${state.filtered.length}`;
  el.row.textContent = `盲审序号 ${record.review_order}`;
  el.path.textContent = record.image_path; el.path.title = record.image_path;
  el.loading.textContent = "正在加载图片……"; el.loading.style.display = "block";
  el.image.style.display = "none"; el.image.classList.remove("zoomed"); el.image.src = api(`/api/image/${record.row}`);
  populate(annotation(record));
  el.previous.disabled = state.index === 0; el.next.disabled = state.index >= state.filtered.length - 1;
  el.save.textContent = complete(annotation(record)) ? "更新并下一张" : "保存并下一张";
}
function move(delta) {
  if (!state.filtered.length) return;
  state.index = Math.max(0, Math.min(state.filtered.length - 1, state.index + delta));
  render(); el.stage.scrollTo({ top: 0, left: 0 });
}
let toastTimer;
function toast(message, error = false) {
  clearTimeout(toastTimer); el.toast.textContent = message; el.toast.className = `toast show${error ? " error" : ""}`;
  toastTimer = setTimeout(() => { el.toast.className = "toast"; }, 2000);
}
async function saveCurrent() {
  const record = current(); if (!record || state.saving) return;
  state.saving = true; el.save.disabled = true; el.saveState.textContent = "正在保存……";
  try {
    const response = await fetch(api(`/api/annotation/${record.row}`), {
      method: "POST", headers: { "Content-Type": "application/json", "X-Audit-Token": state.token }, body: JSON.stringify(formValue()),
    });
    const value = await response.json(); if (!response.ok) throw new Error(value.error || "保存失败");
    state.annotations[String(record.row)] = value.annotation; el.saveState.textContent = "已保存";
    toast(value.annotation.completed ? "已保存并计入进度" : "草稿已保存"); updateProgress();
    if (el.filter.value === "unreviewed" && value.annotation.completed) applyFilter();
    else if (state.index < state.filtered.length - 1) move(1); else render();
  } catch (error) { el.saveState.textContent = "保存失败"; toast(error.message, true); }
  finally { state.saving = false; el.save.disabled = false; }
}
async function initialize() {
  if (!state.token) { el.loading.textContent = "访问链接缺少 token，请使用服务启动时输出的完整地址。"; return; }
  try {
    const response = await fetch("/api/state", { headers: { "X-Audit-Token": state.token } });
    const value = await response.json(); if (!response.ok) throw new Error(value.error || "载入失败");
    state.records = value.records; state.annotations = value.annotations || {}; el.saveState.textContent = "已载入"; applyFilter();
  } catch (error) { el.loading.textContent = `无法载入：${error.message}`; el.saveState.textContent = "载入失败"; }
}
el.image.addEventListener("load", () => { el.loading.style.display = "none"; el.image.style.display = "block"; });
el.image.addEventListener("error", () => { el.loading.textContent = "图片加载失败，请检查服务器原图路径。"; el.loading.style.display = "block"; });
el.image.addEventListener("click", () => el.image.classList.toggle("zoomed"));
el.open.addEventListener("click", () => { if (current()) window.open(api(`/api/image/${current().row}`), "_blank", "noopener"); });
el.previous.addEventListener("click", () => move(-1)); el.next.addEventListener("click", () => move(1));
el.filter.addEventListener("change", () => applyFilter(current()?.row));
el.form.addEventListener("change", updateHint); el.form.addEventListener("submit", (event) => { event.preventDefault(); saveCurrent(); });
el.export.addEventListener("click", () => { location.href = api("/api/export.csv"); });
document.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter") { event.preventDefault(); saveCurrent(); return; }
  const editing = ["TEXTAREA", "SELECT"].includes(document.activeElement?.tagName); if (editing) return;
  const key = event.key.toLowerCase();
  if (key === "g" || key === "b" || key === "u") {
    const value = key === "g" ? "GOOD" : key === "b" ? "BAD" : "UNSURE";
    const input = document.querySelector(`input[name="review_decision"][value="${value}"]`); if (input) { input.checked = true; updateHint(); }
  } else if (event.key === "ArrowLeft") move(-1); else if (event.key === "ArrowRight") move(1);
});
initialize();
