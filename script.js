const palette = {
  Dropout: "#bd6944",
  Enrolled: "#829e77",
  Graduate: "#2f6d58"
};
const classOrder = ["Dropout", "Enrolled", "Graduate"];
// Resolve beside script.js so a GitHub Pages repository subpath is retained.
const resultsUrl = new URL(
  "results/analysis.json",
  document.currentScript?.src || document.baseURI
);
const metricCaptions = {
  weighted_f1: "Mean training-only cross-validation score; weighted F1 is the model-selection metric.",
  accuracy: "Mean training-only cross-validation accuracy; model selection still uses weighted F1.",
  macro_f1: "Mean training-only cross-validation macro F1; model selection still uses weighted F1.",
  roc_auc: "Mean macro one-vs-rest ROC-AUC across five training-only folds; the holdout AUC is reported separately above."
};
let experiment;
let showAllImportance = false;

const byId = id => document.getElementById(id);
const percent = value => `${(value * 100).toFixed(2)}%`;
const formatNumber = value => Number(value).toLocaleString("en-US");

function setText(id, value) {
  const element = byId(id);
  if (element) element.textContent = value;
}

function showMessage(id, message) {
  const container = byId(id);
  if (!container) return;
  const paragraph = document.createElement("p");
  paragraph.className = "loading";
  paragraph.textContent = message;
  container.replaceChildren(paragraph);
}

function renderBars(containerId, rows, { maxValue, format = value => value.toFixed(2), label }) {
  const container = byId(containerId);
  if (!container) return;
  if (!rows.length) {
    showMessage(containerId, "No values are available in the generated results.");
    return;
  }
  const maximum = maxValue || Math.max(...rows.map(row => row.value), 0.00001);
  const currentRows = Array.from(container.querySelectorAll(".bar-row"));
  const canUpdateInPlace = currentRows.length === rows.length && rows.every((row, index) =>
    currentRows[index].dataset.barName === row.name
  );
  if (canUpdateInPlace) {
    rows.forEach((row, index) => {
      const item = currentRows[index];
      const fill = item.querySelector(".bar-fill");
      const value = item.querySelector(".bar-value");
      if (!fill || !value) return;
      const formatted = format(row.value);
      fill.style.width = `${Math.max(0, Math.min(100, row.value / maximum * 100))}%`;
      fill.title = `${row.name}: ${formatted}${label || ""}${row.detail ? `; ${row.detail}` : ""}`;
      let mainValue = value.querySelector(".bar-value-main");
      if (!mainValue) {
        value.replaceChildren();
        mainValue = document.createElement("span");
        mainValue.className = "bar-value-main";
        value.append(mainValue);
      }
      mainValue.textContent = formatted;
      let variation = value.querySelector(".bar-variation");
      if (row.variation) {
        if (!variation) {
          variation = document.createElement("small");
          variation.className = "bar-variation";
          value.append(variation);
        }
        variation.textContent = row.variation;
      } else {
        variation?.remove();
      }
      if (row.detail) {
        item.tabIndex = 0;
        item.setAttribute("aria-label", `${row.name}: ${formatted}${label || ""}; ${row.detail}`);
      } else {
        item.removeAttribute("tabindex");
        item.removeAttribute("aria-label");
      }
    });
    return;
  }

  const chart = document.createElement("div");
  chart.className = "bar-chart";
  chart.setAttribute("role", "list");
  rows.forEach(row => {
    const item = document.createElement("div");
    item.className = "bar-row";
    item.dataset.barName = row.name;
    item.setAttribute("role", "listitem");
    if (row.detail) {
      item.tabIndex = 0;
      item.setAttribute("aria-label", `${row.name}: ${format(row.value)}${label || ""}; ${row.detail}`);
    }
    const name = document.createElement("span");
    name.className = "bar-label";
    name.textContent = row.name;
    const track = document.createElement("span");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(0, Math.min(100, row.value / maximum * 100))}%`;
    fill.title = `${row.name}: ${format(row.value)}${label || ""}${row.detail ? `; ${row.detail}` : ""}`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "bar-value";
    const mainValue = document.createElement("span");
    mainValue.className = "bar-value-main";
    mainValue.textContent = format(row.value);
    value.append(mainValue);
    if (row.variation) {
      const variation = document.createElement("small");
      variation.className = "bar-variation";
      variation.textContent = row.variation;
      value.append(variation);
    }
    item.append(name, track, value);
    chart.append(item);
  });
  container.replaceChildren(chart);
}

function renderDistribution(data) {
  const labels = classOrder;
  renderBars("class-distribution", labels.map(name => ({
    name,
    value: data.dataset.class_distribution[name],
    detail: `${percent(data.dataset.class_distribution[name] / data.dataset.rows)} of all records`
  })), {
    maxValue: Math.max(...labels.map(name => data.dataset.class_distribution[name])),
    format: formatNumber,
    label: " students"
  });
}

function renderModelComparison(metric) {
  const rows = Object.entries(experiment.cv).map(([name, scores]) => ({
    name,
    value: scores[metric][0],
    deviation: scores[metric][1]
  }));
  renderBars("model-comparison", rows.map(row => ({
    name: row.name,
    value: row.value * 100,
    variation: `±${(row.deviation * 100).toFixed(2)} pp`,
    detail: `mean across five training-only folds; standard deviation ${(row.deviation * 100).toFixed(2)} percentage points`
  })), {
    maxValue: 100,
    format: value => `${value.toFixed(2)}%`
  });
  setText("metric-caption", metricCaptions[metric]);
}

function renderClassReport(data) {
  const body = byId("class-report")?.querySelector("tbody");
  if (!body) return;
  body.replaceChildren();
  data.classes.forEach(label => {
    const scores = data.holdout.classification_report[label];
    const row = document.createElement("tr");
    [label, percent(scores.precision), percent(scores.recall), percent(scores["f1-score"]), formatNumber(scores.support)]
      .forEach(value => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
    body.append(row);
  });
}

function renderConfusionMatrix(data) {
  const container = byId("confusion-matrix");
  if (!container) return;
  const table = document.createElement("table");
  table.className = "confusion-table";
  const caption = document.createElement("caption");
  caption.className = "sr-only";
  caption.textContent = "Confusion matrix. Rows are actual classes; columns are predicted classes.";
  table.append(caption);
  const head = document.createElement("thead");
  const headerRow = document.createElement("tr");
  ["Actual / Predicted", ...data.classes].forEach(label => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    headerRow.append(cell);
  });
  head.append(headerRow);
  table.append(head);
  const body = document.createElement("tbody");
  data.holdout.confusion_matrix.forEach((values, rowIndex) => {
    const row = document.createElement("tr");
    const heading = document.createElement("th");
    heading.scope = "row";
    heading.textContent = data.classes[rowIndex];
    row.append(heading);
    values.forEach((value, columnIndex) => {
      const cell = document.createElement("td");
      cell.textContent = formatNumber(value);
      cell.className = rowIndex === columnIndex ? "diagonal" : "off-diagonal";
      cell.setAttribute("aria-label", `${value} ${data.classes[rowIndex]} records predicted as ${data.classes[columnIndex]}`);
      row.append(cell);
    });
    body.append(row);
  });
  table.append(body);
  container.replaceChildren(table);
}

function renderImportance(data) {
  const entries = data.feature_importance || [];
  const visible = showAllImportance ? entries : entries.slice(0, 6);
  renderBars("feature-importance", visible.map(item => ({
    name: item.feature,
    value: item.importance,
    detail: `standard deviation across five permutations: ${Number(item.std).toFixed(4)}`
  })), {
    maxValue: Math.max(...visible.map(item => item.importance)),
    format: value => value.toFixed(3),
    label: " weighted F1 decrease"
  });
  const toggle = byId("importance-toggle");
  if (toggle) {
    toggle.hidden = entries.length <= 6;
    toggle.setAttribute("aria-expanded", String(showAllImportance));
    toggle.textContent = showAllImportance ? "Show top six" : `Show all ${entries.length} features`;
  }
}

function renderSilhouetteComparison(data) {
  const candidates = data.clustering.candidates || [];
  renderBars("silhouette-comparison", candidates.map(candidate => ({
    name: `k = ${candidate.k}`,
    value: candidate.silhouette,
    detail: candidate.k === data.clustering.selected_k ? "selected candidate: highest sampled silhouette score" : "candidate silhouette score"
  })), {
    format: value => value.toFixed(3),
    label: " silhouette score"
  });
}

function renderEvaluation(data) {
  const modelScores = Object.entries(data.cv);
  const ranked = [...modelScores].sort((a, b) => b[1].weighted_f1[0] - a[1].weighted_f1[0]);
  const baseline = modelScores.find(([name]) => /most.?frequent|baseline/i.test(name));
  const selected = data.cv[data.selected_model];
  const runnerUp = ranked.find(([name]) => name !== data.selected_model);
  const runnerUpGap = runnerUp ? selected.weighted_f1[0] - runnerUp[1].weighted_f1[0] : 0;
  const foldSpread = runnerUp ? Math.max(selected.weighted_f1[1], runnerUp[1].weighted_f1[1]) : 0;
  const rankingContext = runnerUpGap <= foldSpread
    ? "The lead over its closest comparator is small relative to fold-to-fold spread, so read it as an edge rather than a clearly separated winner."
    : "Its lead over the closest comparator is larger than the observed fold-to-fold spread, though this remains internal validation on one dataset. ";
  const baselineContext = baseline
    ? ` ${baseline[0]} provides a simple reference point and performs much less strongly than the leading fitted models.`
    : "";
  setText("model-interpretation", `${data.selected_model} ranked first on training-only cross-validation weighted F1, with ${runnerUp?.[0] || "the other models"} as the closest comparator. ${rankingContext}${baselineContext} Fold standard deviations are shown in Results.`);

  const distribution = data.dataset.class_distribution;
  const largest = Object.entries(distribution).sort((a, b) => b[1] - a[1])[0];
  const smallest = Object.entries(distribution).sort((a, b) => a[1] - b[1])[0];
  const weighted = data.holdout.weighted_f1;
  const macro = data.holdout.macro_f1;
  const weightedContext = weighted >= macro
    ? "The support-weighted summary is more optimistic than the equal-class summary."
    : "The equal-class summary is more optimistic than the support-weighted summary.";
  setText("metric-interpretation", `Weighted F1 follows class support, while macro F1 gives each outcome equal weight. ${weightedContext} ${largest[0]} is the largest class and ${smallest[0]} the smallest, so weighted scores can understate weaker performance on less common outcomes.`);

  const report = data.holdout.classification_report;
  const lowestRecall = data.classes
    .map((label, index) => ({ label, index, recall: report[label].recall }))
    .sort((a, b) => a.recall - b.recall)[0];
  const confusions = data.holdout.confusion_matrix[lowestRecall.index]
    .map((count, index) => ({ label: data.classes[index], count, index }))
    .filter(item => item.index !== lowestRecall.index)
    .sort((a, b) => b.count - a.count);
  const mostFrequentConfusion = confusions[0];
  setText("error-interpretation", `${lowestRecall.label} had the weakest holdout recall. Its largest off-diagonal confusion was prediction as ${mostFrequentConfusion.label}. The confusion matrix and class report in Results show the full pattern.`);
}

function renderClusters(data) {
  const container = byId("cluster-outcomes");
  if (!container) return;
  const list = document.createElement("div");
  list.className = "cluster-list";
  const legend = document.createElement("div");
  legend.className = "stacked-legend";
  classOrder.forEach(label => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.backgroundColor = palette[label];
    const text = document.createElement("span");
    text.textContent = label;
    item.append(swatch, text);
    legend.append(item);
  });
  data.clusters.forEach(cluster => {
    const item = document.createElement("div");
    item.className = "cluster-item";
    const title = document.createElement("div");
    title.className = "cluster-title";
    const name = document.createElement("span");
    name.textContent = `Cluster ${cluster.cluster + 1}`;
    const size = document.createElement("span");
    size.textContent = `${formatNumber(cluster.size)} records`;
    title.append(name, size);
    const bar = document.createElement("div");
    bar.className = "stacked-bar";
    bar.setAttribute("role", "img");
    bar.setAttribute("aria-label", `Cluster ${cluster.cluster + 1} outcome shares`);
    classOrder.forEach(label => {
      const share = cluster.outcome_share[label] || 0;
      const segment = document.createElement("span");
      segment.className = "stacked-segment";
      segment.style.width = `${share * 100}%`;
      segment.style.backgroundColor = palette[label];
      segment.title = `${label}: ${percent(share)}`;
      bar.append(segment);
    });
    const profile = document.createElement("div");
    profile.className = "cluster-profiles";
    const shares = document.createElement("span");
    shares.textContent = classOrder
      .map(label => `${label} ${percent(cluster.outcome_share[label] || 0)}`)
      .join(" · ");
    const numeric = cluster.profile.numeric_median || {};
    const medianEntry = Object.entries(numeric)[0];
    if (medianEntry) {
      const median = document.createElement("span");
      median.textContent = `Profile median — ${medianEntry[0]}: ${Number(medianEntry[1]).toFixed(1)}`;
      profile.append(shares, median);
    } else {
      profile.append(shares);
    }
    item.append(title, bar, profile);
    list.append(item);
  });
  list.append(legend);
  container.replaceChildren(list);
  setText("cluster-summary", `Selected k = ${data.selected_k}; best candidate silhouette = ${data.silhouette.toFixed(3)} (candidate k = 2–6). The silhouette suggests modest separation, not proof of natural student groups.`);
}

function renderResults(results) {
  const data = results.experiments.uci;
  experiment = data;
  const holdout = data.holdout;
  const cv = data.cv[data.selected_model];
  const correct = holdout.confusion_matrix.reduce((sum, row, index) => sum + row[index], 0);
  setText("result-model", data.selected_model);
  setText("training-cv-score", percent(cv.weighted_f1[0]));
  setText("training-cv-variation", `±${(cv.weighted_f1[1] * 100).toFixed(2)} percentage points SD across five training-only folds`);
  setText("holdout-accuracy", percent(holdout.accuracy));
  setText("holdout-precision", percent(holdout.precision_weighted));
  setText("holdout-recall", percent(holdout.recall_weighted));
  setText("holdout-weighted-f1", percent(holdout.weighted_f1));
  setText("holdout-macro-f1", percent(holdout.macro_f1));
  setText("holdout-auc", percent(holdout.roc_auc_ovr_macro));
  setText("holdout-correct", `${formatNumber(correct)} of ${formatNumber(data.split.test_records)} records correctly classified`);
  setText("missing-cells", formatNumber(data.dataset.missing_cells));
  setText("duplicate-rows", formatNumber(data.dataset.exact_duplicate_rows));
  setText("data-stat-rows", formatNumber(data.dataset.rows));
  setText("class-total", `n = ${formatNumber(data.dataset.rows)}`);
  document.querySelectorAll("[data-stat='rows']").forEach(node => node.textContent = formatNumber(data.dataset.rows));
  document.querySelectorAll("[data-stat='features']").forEach(node => node.textContent = formatNumber(data.dataset.predictor_columns));
  document.querySelectorAll("[data-stat='classes']").forEach(node => node.textContent = String(data.classes.length));
  document.querySelectorAll("[data-stat='missing']").forEach(node => node.textContent = formatNumber(data.dataset.missing_cells));
  renderDistribution(data);
  renderModelComparison("weighted_f1");
  renderClassReport(data);
  renderConfusionMatrix(data);
  renderImportance(data);
  renderClusters(data.clustering);
  renderSilhouetteComparison(data);
  renderEvaluation(data);
}

function showResultsMessage(message) {
  [
    "class-distribution",
    "model-comparison",
    "feature-importance",
    "silhouette-comparison",
    "cluster-outcomes",
    "confusion-matrix"
  ].forEach(id => showMessage(id, message));
  const body = byId("class-report")?.querySelector("tbody");
  if (body) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.textContent = message;
    row.append(cell);
    body.replaceChildren(row);
  }
}

function requireObject(value, path) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Invalid generated results: ${path} must be an object.`);
  }
  return value;
}

function requireNumber(value, path, { minimum = -Infinity, maximum = Infinity, integer = false } = {}) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum || (integer && !Number.isInteger(value))) {
    const kind = integer ? "whole number" : "finite number";
    const bounds = minimum > -Infinity && maximum < Infinity
      ? ` from ${minimum} to ${maximum}`
      : minimum > -Infinity
        ? ` greater than or equal to ${minimum}`
        : maximum < Infinity
          ? ` less than or equal to ${maximum}`
          : "";
    throw new Error(`Invalid generated results: ${path} must be a ${kind}${bounds}.`);
  }
  return value;
}

function validateResults(results) {
  const data = requireObject(results?.experiments?.uci, "experiments.uci");
  if (!Array.isArray(data.classes) || data.classes.length !== classOrder.length || data.classes.some((label, index) => label !== classOrder[index])) {
    throw new Error(`Invalid generated results: classes must be ordered as ${classOrder.join(", ")}.`);
  }

  const dataset = requireObject(data.dataset, "dataset");
  const distribution = requireObject(dataset.class_distribution, "dataset.class_distribution");
  requireNumber(dataset.rows, "dataset.rows", { minimum: 1, integer: true });
  const counts = classOrder.map(label => requireNumber(distribution[label], `dataset.class_distribution.${label}`, { minimum: 0, integer: true }));
  if (counts.reduce((total, count) => total + count, 0) !== dataset.rows) {
    throw new Error("Invalid generated results: class counts do not add up to dataset.rows.");
  }
  ["predictor_columns", "missing_cells", "exact_duplicate_rows"].forEach(key =>
    requireNumber(dataset[key], `dataset.${key}`, { minimum: 0, integer: true })
  );

  const cv = requireObject(data.cv, "cv");
  const metricNames = ["accuracy", "weighted_f1", "macro_f1", "roc_auc"];
  if (Object.keys(cv).length === 0 || !Object.hasOwn(cv, data.selected_model)) {
    throw new Error("Invalid generated results: cv must contain models and the selected_model.");
  }
  Object.entries(cv).forEach(([model, scores]) => {
    requireObject(scores, `cv.${model}`);
    metricNames.forEach(metric => {
      const pair = scores[metric];
      if (!Array.isArray(pair) || pair.length !== 2) {
        throw new Error(`Invalid generated results: cv.${model}.${metric} must contain a mean and standard deviation.`);
      }
      requireNumber(pair[0], `cv.${model}.${metric}[0]`, { minimum: 0, maximum: 1 });
      requireNumber(pair[1], `cv.${model}.${metric}[1]`, { minimum: 0, maximum: 1 });
    });
  });

  const holdout = requireObject(data.holdout, "holdout");
  ["accuracy", "precision_weighted", "recall_weighted", "weighted_f1", "macro_f1", "roc_auc_ovr_macro"].forEach(metric =>
    requireNumber(holdout[metric], `holdout.${metric}`, { minimum: 0, maximum: 1 })
  );
  const report = requireObject(holdout.classification_report, "holdout.classification_report");
  classOrder.forEach(label => {
    const scores = requireObject(report[label], `holdout.classification_report.${label}`);
    ["precision", "recall", "f1-score"].forEach(metric =>
      requireNumber(scores[metric], `holdout.classification_report.${label}.${metric}`, { minimum: 0, maximum: 1 })
    );
    requireNumber(scores.support, `holdout.classification_report.${label}.support`, { minimum: 0, integer: true });
  });
  if (!Array.isArray(holdout.confusion_matrix) || holdout.confusion_matrix.length !== classOrder.length) {
    throw new Error("Invalid generated results: holdout.confusion_matrix must be a square class-by-class matrix.");
  }
  holdout.confusion_matrix.forEach((row, index) => {
    if (!Array.isArray(row) || row.length !== classOrder.length) {
      throw new Error("Invalid generated results: holdout.confusion_matrix must be a square class-by-class matrix.");
    }
    row.forEach((value, column) => requireNumber(value, `holdout.confusion_matrix[${index}][${column}]`, { minimum: 0, integer: true }));
  });

  const split = requireObject(data.split, "split");
  requireNumber(split.test_records, "split.test_records", { minimum: 1, integer: true });
  if (!Array.isArray(data.feature_importance) || data.feature_importance.length === 0) {
    throw new Error("Invalid generated results: feature_importance must contain at least one feature.");
  }
  data.feature_importance.forEach((item, index) => {
    requireObject(item, `feature_importance[${index}]`);
    if (typeof item.feature !== "string" || item.feature.length === 0) {
      throw new Error(`Invalid generated results: feature_importance[${index}].feature must be a non-empty string.`);
    }
    requireNumber(item.importance, `feature_importance[${index}].importance`);
    requireNumber(item.std, `feature_importance[${index}].std`, { minimum: 0 });
  });

  const clustering = requireObject(data.clustering, "clustering");
  if (!Array.isArray(clustering.candidates) || clustering.candidates.length === 0) {
    throw new Error("Invalid generated results: clustering.candidates must contain silhouette scores.");
  }
  clustering.candidates.forEach((candidate, index) => {
    requireObject(candidate, `clustering.candidates[${index}]`);
    requireNumber(candidate.k, `clustering.candidates[${index}].k`, { minimum: 2, integer: true });
    requireNumber(candidate.silhouette, `clustering.candidates[${index}].silhouette`, { minimum: -1, maximum: 1 });
  });
  if (!clustering.candidates.some(candidate => candidate.k === clustering.selected_k)) {
    throw new Error("Invalid generated results: clustering.selected_k is not in the candidate scores.");
  }
  requireNumber(clustering.silhouette, "clustering.silhouette", { minimum: -1, maximum: 1 });
  requireNumber(clustering.sample_size, "clustering.sample_size", { minimum: 1, integer: true });
  if (!Array.isArray(clustering.clusters) || clustering.clusters.length === 0) {
    throw new Error("Invalid generated results: clustering.clusters must contain at least one cluster.");
  }
  clustering.clusters.forEach((cluster, index) => {
    requireObject(cluster, `clustering.clusters[${index}]`);
    requireNumber(cluster.cluster, `clustering.clusters[${index}].cluster`, { minimum: 0, integer: true });
    requireNumber(cluster.size, `clustering.clusters[${index}].size`, { minimum: 1, integer: true });
    const shares = requireObject(cluster.outcome_share, `clustering.clusters[${index}].outcome_share`);
    classOrder.forEach(label => requireNumber(shares[label], `clustering.clusters[${index}].outcome_share.${label}`, { minimum: 0, maximum: 1 }));
  });
  return results;
}

async function loadResults() {
  let results;
  try {
    if (window.location.protocol === "file:") {
      throw new Error("This page was opened directly from disk; JSON data can only be loaded when the site is served over HTTP.");
    }
    const response = await fetch(resultsUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`Result request failed with HTTP ${response.status}.`);
    results = validateResults(await response.json());
  } catch (error) {
    console.error(`Could not load generated research results from ${resultsUrl.href}.`, error);
    const filePreview = window.location.protocol === "file:";
    const nextStep = filePreview
      ? "Open a terminal in the project root, run `python -m http.server 8000`, and visit http://localhost:8000."
      : "Confirm results/analysis.json exists at this project path and was included in the deployment.";
    const message = `Could not load or validate the generated results at ${resultsUrl.pathname}. ${nextStep} Details: ${error.message}`;
    showResultsMessage(message);
    return;
  }

  try {
    renderResults(results);
  } catch (error) {
    console.error("The generated results loaded, but the page could not render them.", error);
    showResultsMessage("Generated results loaded, but could not be displayed. Check the browser console for details.");
  }
}

const menuButton = document.querySelector(".menu-toggle");
const nav = document.querySelector(".primary-nav");
if (menuButton && nav) {
  menuButton.addEventListener("click", () => {
    const open = menuButton.getAttribute("aria-expanded") !== "true";
    menuButton.setAttribute("aria-expanded", String(open));
    nav.classList.toggle("is-open", open);
  });
  nav.addEventListener("click", event => {
    if (event.target instanceof HTMLAnchorElement) {
      menuButton.setAttribute("aria-expanded", "false");
      nav.classList.remove("is-open");
    }
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") {
      menuButton.setAttribute("aria-expanded", "false");
      nav.classList.remove("is-open");
      menuButton.focus();
    }
  });
}

function updateActiveNavigation() {
  const links = [...document.querySelectorAll("[data-section-nav] a[href^='#']")];
  const sections = [...new Set(links.map(link => document.getElementById(link.hash.slice(1))).filter(Boolean))];
  const threshold = Math.min(240, window.innerHeight * 0.32);
  let active = sections[0];
  sections.forEach(section => {
    if (section.getBoundingClientRect().top <= threshold) active = section;
  });
  if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) active = sections.at(-1) || active;
  links.forEach(link => {
    if (active && link.hash === `#${active.id}`) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  });
  const currentStage = active?.id === "results" ? "evaluation" : active?.id;
  document.querySelectorAll("[data-crisp-stage]").forEach(link => {
    if (link.dataset.crispStage === currentStage) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  });
}
window.addEventListener("scroll", updateActiveNavigation, { passive: true });
window.addEventListener("resize", updateActiveNavigation);
updateActiveNavigation();

const importanceToggle = byId("importance-toggle");
if (importanceToggle) {
  importanceToggle.addEventListener("click", () => {
    showAllImportance = !showAllImportance;
    if (experiment) renderImportance(experiment);
  });
}

document.querySelectorAll("[data-metric]").forEach(button => {
  button.addEventListener("click", () => {
    if (!experiment) return;
    document.querySelectorAll("[data-metric]").forEach(option => {
      option.setAttribute("aria-pressed", String(option === button));
    });
    renderModelComparison(button.dataset.metric);
  });
});

loadResults();
