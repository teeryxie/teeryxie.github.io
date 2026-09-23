"use strict";

const messages = {
  en: {
    skip: "Skip to results", eyebrow: "A benchmark for social interaction", title: "Leaderboard",
    intro: "Who is speaking. When to respond. What to say. Compare how omni models navigate social interaction through audio and video.",
    resources: "Resources", paper: "Paper", code: "Code", dataset: "Dataset", download: "Download results ↓",
    results: "Results", comparison: "Results are grouped by evaluation protocol and source. Select one group at a time; scores from different groups are not a shared ranking.",
    group: "Evaluation group", search: "Find a model", placeholder: "Model name", loading: "Loading results…",
    tableRegion: "Model results, scroll horizontally for all metrics", caption: "Model scores on a 0–100 scale. Higher is better. Select a column heading to sort.",
    model: "Model", tableHelp: "Click a metric to sort. All scores use a 0–100 scale; higher is better. — means unavailable and is placed last in either sort direction.",
    metricsTitle: "Reading the metrics", jointNote: "QEns_joint = QEns × Cov+ / 100. A high response-quality score alone does not imply reliable decisions about when to speak.",
    sourcesTitle: "Sources & comparability", sourcesText: "Paper-reported values are transcribed from the cited paper, not rerun here. New evaluations belong to separate groups unless their model settings, prompts, data and judges are confirmed to match. Each group records its source and sample counts.",
    missingText: "Missing or unfinished evaluations are not assigned a score. This page does not combine the six metrics into an overall score.",
    correction: "Report a correction ↗", updated: "Updated", source: "Source", samples: "Samples", positive: "gold-positive", pending: "Evaluation pending", partial: "Quality scoring pending",
    empty: "No published results in this group yet.", noMatch: "No models match your search.", noData: "Results are being prepared. No scores have been published here yet.",
    loadError: "Results could not be loaded. Please reload the page, or use the Download results link.", count: "models shown",
    metrics: [
      ["Who", "Accuracy in identifying who is speaking from the available audio and video."],
      ["When", "Accuracy of the YES / NO decision to respond at a fixed timestamp, using only the audio and video available up to that time."],
      ["QGold", "Mean response quality over all gold-positive items: cases where the reference says a response is appropriate. Generation is forced even when the model predicts NO."],
      ["QEns", "Mean quality of non-empty responses on gold-positive items where the model also predicts YES. Each response is scored by the evaluation’s three judges."],
      ["Cov+", "Coverage of gold-positive items: the percentage for which the model predicts YES and generates a non-empty response."],
      ["QEns_joint", "Response quality weighted by coverage. Missed gold-positive opportunities contribute zero."]
    ]
  },
  zh: {
    skip: "跳转到结果", eyebrow: "面向社会交互的多模态评测", title: "排行榜",
    intro: "谁在说话、何时回应、说什么。比较全模态模型如何利用音频与视频理解社会交互，并作出回应。",
    resources: "研究资源", paper: "论文", code: "代码", dataset: "数据集", download: "下载结果 ↓",
    results: "评测结果", comparison: "结果按评测协议和来源分组，每次查看一组。不同组的分数不混合排名。",
    group: "评测分组", search: "查找模型", placeholder: "输入模型名称", loading: "正在加载结果…",
    tableRegion: "模型结果，可横向滚动查看全部指标", caption: "模型分数均采用 0–100 标度，越高越好。点击列标题排序。",
    model: "模型", tableHelp: "点击指标名称排序。所有分数均采用 0–100 标度，越高越好。— 表示缺失值，升序和降序均置于末尾。",
    metricsTitle: "指标说明", jointNote: "QEns_joint = QEns × Cov+ / 100。回答质量高，不一定意味着模型能可靠判断何时应该开口。",
    sourcesTitle: "来源与可比性", sourcesText: "论文结果转录自所引用的论文，不代表本站重新运行的结果。新增评测单独分组，除非已确认模型设置、提示词、数据和评委一致。每组均注明来源和样本数。",
    missingText: "缺失或尚未完成的评测不赋予分数。本站不把六项指标合成为一个总分。",
    correction: "反馈勘误 ↗", updated: "更新日期", source: "来源", samples: "样本", positive: "应回应样本", pending: "评测待完成", partial: "回答质量待评分",
    empty: "此组暂未发布评测结果。", noMatch: "没有匹配的模型。", noData: "结果整理中，本站暂未发布分数。",
    loadError: "无法加载结果。请刷新页面，或使用“下载结果”链接。", count: "个模型",
    metrics: [
      ["Who", "说话者归属判断的准确率，即根据可见的音视频信息识别谁正在说话。"],
      ["When", "在固定时间点判断是否应回应的准确率，只使用该时间点及之前的音视频信息，输出 YES 或 NO。"],
      ["QGold", "所有应回应样本的平均回答质量。应回应样本指标准答案认为应当回应的情况；即使模型预测 NO，也强制生成回答进行评分。"],
      ["QEns", "在应回应且模型也预测 YES、生成非空回答的样本中，计算平均回答质量。每条回答由该评测规定的三位评委评分。"],
      ["Cov+", "应回应样本的覆盖率，即其中模型预测 YES 且生成非空回答的比例。"],
      ["QEns_joint", "结合覆盖率的回答质量。漏掉的应回应机会计为零分。"]
    ]
  }
};

const metricKeys = ["who", "when", "qgold", "qens", "cov_plus", "qens_joint"];
let language = "en";
try { language = localStorage.getItem("socialomni-language") === "zh" ? "zh" : "en"; } catch {}
let dataset;
let failed = false;
let sortKey = "model";
let descending = false;
const cohortSelect = document.getElementById("cohort");
const search = document.getElementById("search");
const status = document.getElementById("load-status");
const localized = (value) => typeof value === "string" ? value : value?.[language] || value?.en || "";
const score = (record, key) => record.status === "pending" ? null : record.metrics?.[key];
const validScore = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function sourceLink(source) {
  const link = element("a", localized(source.label));
  link.href = safeUrl(source.url);
  return link;
}

function safeUrl(value) {
  const url = new URL(value, location.href);
  if (url.protocol !== "https:" && url.protocol !== "http:") throw new Error("Unsupported link protocol");
  return url.href;
}

function render() {
  const t = messages[language];
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  document.title = `SocialOmni · ${t.title}`;
  document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t[node.dataset.i18n]; });
  document.querySelectorAll("[data-i18n-aria]").forEach((node) => node.setAttribute("aria-label", t[node.dataset.i18nAria]));
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => { node.placeholder = t[node.dataset.i18nPlaceholder]; });
  document.querySelectorAll("[data-lang]").forEach((node) => node.setAttribute("aria-pressed", String(node.dataset.lang === language)));
  document.getElementById("metric-definitions").replaceChildren(...t.metrics.flatMap(([name, definition]) => [element("dt", name), element("dd", definition)]));
  if (!dataset) {
    status.textContent = failed ? t.loadError : t.loading;
    return;
  }
  const selected = cohortSelect.value;
  cohortSelect.replaceChildren(...dataset.cohorts.map((cohort) => {
    const option = element("option", localized(cohort.label));
    option.value = cohort.id;
    return option;
  }));
  if (dataset.cohorts.some((cohort) => cohort.id === selected)) cohortSelect.value = selected;
  document.getElementById("updated").textContent = dataset.updated_at ? `${t.updated} ${dataset.updated_at}` : "";
  renderRows();
}

function renderRows() {
  const t = messages[language];
  const cohort = dataset.cohorts.find((item) => item.id === cohortSelect.value);
  const detail = document.getElementById("cohort-detail");
  const notes = document.getElementById("result-notes");
  detail.replaceChildren();
  notes.replaceChildren();
  if (cohort) {
    detail.append(element("h3", localized(cohort.label)), element("p", localized(cohort.description)));
    const counts = Object.entries(cohort.samples || {}).filter(([, value]) => value !== null).map(([key, value]) => `${key === "gold_positive" ? t.positive : key === "who" ? "Who" : key === "when" ? "When" : key}: ${value}`);
    if (counts.length) detail.append(element("p", `${t.samples} · ${counts.join(" / ")}`));
    if (cohort.source) {
      const source = element("p", `${t.source} · `);
      source.append(sourceLink(cohort.source));
      detail.append(source);
    }
    if (cohort.notes) detail.append(element("p", localized(cohort.notes)));
  }
  const query = search.value.trim().toLocaleLowerCase();
  const records = dataset.records.filter((item) => item.cohort === cohort?.id && `${item.model} ${item.variant || ""}`.toLocaleLowerCase().includes(query));
  records.sort((a, b) => {
    if (sortKey === "model") return a.model.localeCompare(b.model) * (descending ? -1 : 1);
    const av = score(a, sortKey), bv = score(b, sortKey);
    if (!validScore(av)) return validScore(bv) ? 1 : a.model.localeCompare(b.model);
    if (!validScore(bv)) return -1;
    return (av - bv) * (descending ? -1 : 1) || a.model.localeCompare(b.model);
  });
  document.getElementById("rows").replaceChildren(...records.map((record) => {
    const row = element("tr");
    const name = element("td", record.model);
    if (record.variant) name.append(element("span", record.variant, "model-meta"));
    if (record.status === "pending") name.append(element("span", t.pending, "model-meta"));
    if (record.status === "partial") name.append(element("span", t.partial, "model-meta"));
    row.append(name);
    metricKeys.forEach((key) => row.append(element("td", validScore(score(record, key)) ? score(record, key).toFixed(2) : "—")));
    if (record.notes || record.source) {
      const note = element("p");
      note.append(element("strong", `${record.model}: `), document.createTextNode(localized(record.notes)));
      if (record.source) { note.append(document.createTextNode(" "), sourceLink(record.source)); }
      notes.append(note);
    }
    return row;
  }));
  document.getElementById("table-wrap").hidden = !records.length;
  status.textContent = records.length ? `${records.length} ${language === "en" && records.length === 1 ? "model shown" : t.count}` : !cohort ? t.noData : query ? t.noMatch : t.empty;
  document.querySelectorAll("th[data-key]").forEach((header) => {
    const selected = header.dataset.key === sortKey;
    if (selected) header.setAttribute("aria-sort", descending ? "descending" : "ascending");
    else header.removeAttribute("aria-sort");
    header.querySelector(".sort-indicator").textContent = selected ? descending ? " ↓" : " ↑" : " ↕";
  });
}

document.querySelectorAll("[data-lang]").forEach((button) => button.addEventListener("click", () => {
  language = button.dataset.lang;
  try { localStorage.setItem("socialomni-language", language); } catch {}
  render();
}));
document.querySelectorAll("[data-sort]").forEach((button) => button.addEventListener("click", () => {
  const key = button.dataset.sort;
  descending = sortKey === key ? !descending : key !== "model";
  sortKey = key;
  renderRows();
}));
cohortSelect.addEventListener("change", renderRows);
search.addEventListener("input", renderRows);
render();
fetch("data.json").then((response) => {
  if (!response.ok) throw new Error("Data unavailable");
  return response.json();
}).then((data) => {
  if (!Array.isArray(data.cohorts) || !Array.isArray(data.records)) throw new Error("Invalid data format");
  dataset = data;
  Object.entries(data.links || {}).forEach(([key, value]) => {
    const link = document.getElementById(`${key}-link`);
    if (link) link.href = safeUrl(value);
  });
  cohortSelect.disabled = !data.cohorts.length;
  search.disabled = !data.records.length;
  render();
}).catch(() => {
  dataset = undefined;
  failed = true;
  cohortSelect.disabled = true;
  search.disabled = true;
  status.classList.add("error");
  document.getElementById("table-wrap").hidden = true;
  render();
});
