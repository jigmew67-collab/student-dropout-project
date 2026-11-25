// =========================================================
// COLORS
// =========================================================

const green = '#2c7a57';
const lime = '#b9e86b';
const ink = '#10251e';
const muted = '#dbe4dd';


// =========================================================
// CHART DEFAULT SETTINGS
// =========================================================

if (typeof Chart !== 'undefined') {
    Chart.defaults.font.family = 'Manrope';
    Chart.defaults.color = '#66766f';
}

const makeChart = (canvasId, config) => {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    if (typeof Chart === 'undefined') {
        setText('results-load-status', 'Analysis data loaded, but Chart.js did not load; charts are unavailable.');
        return null;
    }
    if (Chart.getChart(canvas)) Chart.getChart(canvas).destroy();
    return new Chart(canvas, config);
};

const uciImportanceFallback = [
    { feature: 'Curricular units 2nd sem (approved)', importance: 0.050258191468661326 },
    { feature: 'Tuition fees up to date', importance: 0.043123748227947 },
    { feature: 'Curricular units 1st sem (approved)', importance: 0.015463601956593109 },
    { feature: 'Curricular units 2nd sem (evaluations)', importance: 0.011910642372923252 },
    { feature: 'Curricular units 2nd sem (grade)', importance: 0.011832190309396728 }
];

makeChart('uciImportanceChart', {
    type: 'bar',
    data: {
        labels: uciImportanceFallback.map(row => row.feature),
        datasets: [{
            label: 'Weighted F1 decrease after shuffling',
            data: uciImportanceFallback.map(row => row.importance),
            backgroundColor: green,
            borderRadius: 5
        }]
    },
    options: {
        indexAxis: 'y',
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
            x: { beginAtZero: true, title: { display: true, text: 'Permutation importance' } },
            y: { grid: { display: false } }
        }
    }
});


// =========================================================
// SCROLL REVEAL ANIMATION
// =========================================================

const observer = new IntersectionObserver(
    (entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    },
    {
        threshold: 0.08
    }
);

document.querySelectorAll('.reveal').forEach((element) => {
    observer.observe(element);
});


// =========================================================
// MOBILE NAVIGATION MENU
// =========================================================

const btn = document.querySelector('.menu-btn');
const links = document.querySelector('.nav-links');

if (btn && links) {
    btn.addEventListener('click', () => {
        const isOpen = links.classList.toggle('open');
        btn.setAttribute('aria-expanded', String(isOpen));
        btn.setAttribute('aria-label', isOpen ? 'Close menu' : 'Open menu');
    });
}


// Close mobile menu after clicking a navigation link
document.querySelectorAll('.nav-links a').forEach((link) => {
    link.addEventListener('click', () => {
        if (links) links.classList.remove('open');
        if (btn) {
            btn.setAttribute('aria-expanded', 'false');
            btn.setAttribute('aria-label', 'Open menu');
        }
    });
});

// Generated experiment outputs drive the interactive interpretation charts.
const palette = ['#2c7a57', '#b9e86b', '#7bb6a0', '#7692a5', '#d69c59', '#a788b5'];
const percent = value => `${(value * 100).toFixed(2)}%`;

function setText(id, text) {
    const element = document.getElementById(id);
    if (element) element.textContent = text;
}

function showError(containerId, message) {
    const container = document.getElementById(containerId);
    if (container) container.textContent = message;
}

function showTableMessage(tableId, message, isBody = false) {
    const element = document.getElementById(tableId);
    const body = isBody ? element : element?.querySelector('tbody');
    if (!body) return;
    const columns = isBody
        ? Math.max(1, body.closest('table')?.querySelectorAll('thead th').length || 0)
        : Math.max(1, element.querySelectorAll('thead th').length);
    body.innerHTML = `<tr><td colspan="${columns}">${message}</td></tr>`;
}

function showOuladUnavailable(message) {
    setText('oulad-result-status', message);
    ['oulad-accuracy', 'oulad-f1', 'oulad-auc', 'oulad-model',
        'oulad-lesson-accuracy', 'oulad-lesson-detected', 'oulad-lesson-total',
        'oulad-lesson-missed', 'oulad-lesson-precision', 'oulad-lesson-recall',
        'oulad-lesson-f1'].forEach(id => setText(id, 'Unavailable'));
    showTableMessage('ouladModelTable', message);
    showTableMessage('ouladConfusion', message);
    showTableMessage('ouladClassReport', message);
    showTableMessage('ouladImputationTable', message);
    showTableMessage('ouladImportanceTable', message, true);
    setText('ouladClusterText', message);
    setText('oulad-data-quality-summary', message);
    setText('oulad-missingness-caption', message);
    setText('oulad-imputation-selection', message);
}

function renderDistribution(data) {
    const canvas = document.getElementById('distributionChart');
    if (!canvas || !data?.class_distribution) return;
    const labels = data.classes || Object.keys(data.class_distribution);
    makeChart('distributionChart', {
        type: 'bar',
        data: { labels, datasets: [{ data: labels.map(label => data.class_distribution[label]), backgroundColor: [green, '#75ad91', '#bdd2c5'], borderRadius: 8, borderSkipped: false }] },
        options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, grid: { color: '#edf0ec' } }, x: { grid: { display: false } } } }
    });
}

function renderHeroDistribution(data) {
    const canvas = document.getElementById('classChart');
    if (!canvas || !data?.class_distribution) return;
    const labels = data.classes || Object.keys(data.class_distribution);
    makeChart('classChart', {
        type: 'doughnut',
        data: { labels, datasets: [{ data: labels.map(label => data.class_distribution[label]), backgroundColor: [lime, '#57a77c', '#dce5df'], borderWidth: 0 }] },
        options: { responsive: true, maintainAspectRatio: false, cutout: '72%', plugins: { legend: { position: 'right', labels: { color: '#c5d2cc', boxWidth: 8, usePointStyle: true, font: { size: 9 } } } } }
    });
}

function renderModelComparison(canvasId, cv) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !cv) return;
    const labels = Object.keys(cv);
    makeChart(canvasId, {
        type: 'bar',
        data: { labels: labels.map(label => label === 'Baseline (Most Frequent)' ? 'Baseline' : label), datasets: [
            { label: 'Accuracy %', data: labels.map(label => cv[label].accuracy[0] * 100), backgroundColor: green, borderRadius: 6 },
            { label: 'Weighted F1 %', data: labels.map(label => cv[label].weighted_f1[0] * 100), backgroundColor: lime, borderRadius: 6 }
        ] },
        options: { responsive: true, indexAxis: 'y', plugins: { legend: { labels: { boxWidth: 9, usePointStyle: true, font: { size: 9 } } } }, scales: { x: { min: 0, max: 100, grid: { color: '#edf0ec' } }, y: { grid: { display: false } } } }
    });
}

function formatMeanStd(pair) {
    return Array.isArray(pair) && pair.length === 2
        ? `${percent(pair[0])} ± ${percent(pair[1])}`
        : 'N/A';
}

function renderModelTable(tableId, cv, selectedModel) {
    const body = document.querySelector(`#${tableId} tbody`);
    if (!body) return;
    if (!cv) {
        body.innerHTML = '<tr><td colspan="5">Cross-validation results are unavailable.</td></tr>';
        return;
    }
    body.innerHTML = Object.entries(cv).map(([name, scores]) => {
        const selected = name === selectedModel;
        return `<tr${selected ? ' class="highlight"' : ''}><th scope="row">${name}${selected ? ' · selected by CV weighted F1' : ''}</th><td>${formatMeanStd(scores.accuracy)}</td><td>${formatMeanStd(scores.weighted_f1)}</td><td>${formatMeanStd(scores.macro_f1)}</td><td>${formatMeanStd(scores.roc_auc)}</td></tr>`;
    }).join('');
}

function renderClassPerformance(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !data?.classes || !data?.holdout?.classification_report) return;
    const metrics = ['precision', 'recall', 'f1-score'];
    const labels = { precision: 'Precision', recall: 'Recall', 'f1-score': 'F1' };
    makeChart(canvasId, {
        type: 'bar',
        data: {
            labels: data.classes,
            datasets: metrics.map((metric, index) => ({
                label: labels[metric],
                data: data.classes.map(className => data.holdout.classification_report[className][metric]),
                backgroundColor: [green, lime, '#7bb6a0'][index],
                borderRadius: 4
            }))
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true, max: 1, ticks: { callback: value => `${Math.round(value * 100)}%` } },
                x: { grid: { display: false } }
            }
        }
    });
}

function renderConfusionAndReport(data) {
    const confusion = document.getElementById('uciConfusion');
    const report = document.getElementById('uciClassReport');
    if (!confusion || !data?.holdout?.confusion_matrix) return;
    const classes = data.classes;
    const matrix = data.holdout.confusion_matrix;
    confusion.innerHTML = `<tr><th>Actual \\ Predicted</th>${classes.map(label => `<th>Pred. ${label}</th>`).join('')}</tr>` +
        classes.map((label, i) => `<tr><th>True ${label}</th>${matrix[i].map(value => `<td>${value}</td>`).join('')}</tr>`).join('');
    if (report) {
        report.innerHTML = classes.map(label => {
            const row = data.holdout.classification_report[label];
            return `<tr><th>${label}</th><td>${percent(row.precision)}</td><td>${percent(row.recall)}</td><td>${percent(row['f1-score'])}</td><td>${row.support.toLocaleString()}</td></tr>`;
        }).join('');
    }
    setText('uci-selected-model', data.selected_model || 'Selected model');
}

function renderImportance(canvasId, rows, tableId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    if (!rows?.length) {
        showError(`${canvasId}-status`, 'Permutation-importance results are unavailable.');
        const body = tableId && document.getElementById(tableId);
        if (body) body.innerHTML = '<tr><td colspan="4">Permutation-importance results are unavailable.</td></tr>';
        return;
    }
    const body = tableId && document.getElementById(tableId);
    if (body) {
        body.innerHTML = rows.map((row, index) =>
            `<tr><td>${index + 1}</td><th scope="row">${row.feature}</th><td>${row.importance.toFixed(5)}</td><td>${row.std.toFixed(5)}</td></tr>`
        ).join('');
    }
    makeChart(canvasId, {
        type: 'bar',
        data: { labels: rows.map(row => row.feature), datasets: [{
            label: 'Weighted F1 decrease after shuffling',
            data: rows.map(row => row.importance), backgroundColor: green, borderRadius: 5
        }] },
        options: { indexAxis: 'y', responsive: true, plugins: { legend: { display: false } },
            scales: { x: { beginAtZero: true, title: { display: true, text: 'Permutation importance' } }, y: { grid: { display: false } } } }
    });
}

function renderClusterOutcomeDistribution(canvasId, clustering) {
    const canvas = document.getElementById(canvasId);
    const groups = clustering?.clusters;
    if (!canvas || !groups?.length) return;
    const outcomes = [...new Set(groups.flatMap(group => Object.keys(group.outcome_share || {})))];
    makeChart(canvasId, {
        type: 'bar',
        data: {
            labels: groups.map(group => `Cluster ${group.cluster + 1} (n=${group.size.toLocaleString()})`),
            datasets: outcomes.map((outcome, index) => ({
                label: outcome,
                data: groups.map(group => group.outcome_share[outcome] || 0),
                backgroundColor: palette[index % palette.length]
            }))
        },
        options: {
            responsive: true,
            indexAxis: 'y',
            scales: {
                x: { stacked: true, min: 0, max: 1, ticks: { callback: value => `${Math.round(value * 100)}%` } },
                y: { stacked: true, grid: { display: false } }
            }
        }
    });
}

function renderClusters(canvasId, textId, clustering, distributionCanvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    if (!clustering?.pca_points) {
        showError(textId, 'Clustering results are unavailable.');
        return;
    }
    const points = clustering.pca_points;
    const clusters = [...new Set(points.map(point => point.cluster))].sort((a, b) => a - b);
    makeChart(canvasId, {
        type: 'scatter',
        data: { datasets: clusters.map(cluster => ({
            label: `Cluster ${cluster + 1}`,
            data: points.filter(point => point.cluster === cluster).map(point => ({ x: point.x, y: point.y, outcome: point.outcome })),
            backgroundColor: palette[cluster % palette.length], pointRadius: 2, pointHoverRadius: 4
        })) },
        options: { responsive: true, plugins: { tooltip: { callbacks: { label: item => `${item.dataset.label}: ${item.raw.outcome}` } } },
            scales: { x: { title: { display: true, text: 'PCA component 1' } }, y: { title: { display: true, text: 'PCA component 2' } } } }
    });
    const text = document.getElementById(textId);
    const silhouette = Number.isFinite(clustering.silhouette)
        ? clustering.silhouette.toFixed(3)
        : 'not available';
    if (text) text.innerHTML = `<p class="chart-note">Selected k = ${clustering.selected_k}; silhouette = ${silhouette}. ${clustering.clusters.map(item => {
        const distribution = Object.entries(item.outcome_share).map(([label, share]) => `${label} ${(share * 100).toFixed(1)}%`).join(', ');
        const numeric = Object.entries(item.profile?.numeric_median || {}).map(([key, value]) => `${key} median ${Number(value).toFixed(1)}`);
        const categorical = Object.entries(item.profile?.categorical_mode || {}).map(([key, value]) => `${key} most common ${value}`);
        const profile = [...numeric, ...categorical].join('; ');
        return `Cluster ${item.cluster + 1} (n=${item.size.toLocaleString()}): ${distribution}. Profile: ${profile}`;
    }).join(' · ')}</p>`;
    if (distributionCanvasId) renderClusterOutcomeDistribution(distributionCanvasId, clustering);
}

function renderMissingnessChart(audit) {
    const canvas = document.getElementById('ouladMissingnessChart');
    const caption = document.getElementById('oulad-missingness-caption');
    if (!canvas) return;
    const missingFeatures = audit?.by_feature?.filter(item => item.missing_cells > 0) || [];
    if (!missingFeatures.length) {
        if (caption) caption.textContent = 'No missing predictor values were reported for OULAD.';
        return;
    }
    const displayNames = {
        imd_band: 'IMD band',
        early_assessment_mean_score: 'Early assessment mean'
    };
    makeChart('ouladMissingnessChart', {
        type: 'bar',
        data: {
            labels: missingFeatures.map(item => displayNames[item.feature] || item.feature),
            datasets: [{
                label: 'Records missing (%)',
                data: missingFeatures.map(item => item.missing_percent),
                missingCells: missingFeatures.map(item => item.missing_cells),
                backgroundColor: [green, lime],
                borderRadius: 5
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: {
                    title: items => missingFeatures[items[0].dataIndex].feature,
                    label: item =>
                        `${item.parsed.x.toFixed(2)}% missing (${item.dataset.missingCells[item.dataIndex].toLocaleString()} records)`
                } }
            },
            scales: {
                x: { beginAtZero: true, title: { display: true, text: 'Eligible registrations missing (%)' } },
                y: { grid: { display: false } }
            }
        }
    });
    if (caption) caption.textContent = 'Only predictors with missing cells are shown. Hover for the missing-record count; all other model predictors are complete.';
}

function renderPreprocessingComparison(comparison, classes) {
    const table = document.querySelector('#ouladImputationTable tbody');
    const selection = document.getElementById('oulad-imputation-selection');
    if (!table || !comparison?.variants) {
        if (selection) selection.textContent = 'Generated preprocessing comparison is unavailable.';
        if (table) table.innerHTML = '<tr><td colspan="8">Generated preprocessing comparison is unavailable.</td></tr>';
        return;
    }
    const labels = {
        A_existing: 'A · Existing handling',
        B_fold_unknown: 'B · Fold-fitted imputation',
        C_fold_unknown_indicator: 'C · Imputation + indicators'
    };
    table.innerHTML = Object.entries(comparison.variants).map(([name, result]) => {
        const cv = result.cv[result.selected_model];
        const holdout = result.holdout;
        const withdrawn = classes?.includes('Withdrawn')
            ? holdout.classification_report.Withdrawn.recall
            : null;
        return `<tr><th>${labels[name] || name}</th><td>${result.selected_model}</td><td>${percent(cv.weighted_f1[0])}</td><td>${percent(holdout.accuracy)}</td><td>${percent(holdout.weighted_f1)}</td><td>${percent(holdout.macro_f1)}</td><td>${holdout.roc_auc_ovr_macro == null ? 'N/A' : percent(holdout.roc_auc_ovr_macro)}</td><td>${withdrawn == null ? 'N/A' : percent(withdrawn)}</td></tr>`;
    }).join('');
    const selected = labels[comparison.selected_variant] || comparison.selected_variant;
    if (selection) selection.textContent = `Selected using training-only CV weighted F1: ${selected}. Holdout metrics were not used for selection.`;
}

function renderOulad(exp, preprocessingComparison) {
    const status = document.getElementById('oulad-result-status');
    if (!exp || exp.status === 'not_run') {
        showOuladUnavailable('OULAD experiment has no generated results in this build.');
        return;
    }
    const data = exp;
    const quality = data.data_quality;
    if (!quality || !data.holdout || !data.cv) {
        showOuladUnavailable('OULAD generated results are incomplete. Required data-quality, cross-validation, or holdout metrics are missing.');
        return;
    }
    const resultBox = document.getElementById('oulad-results');
    if (resultBox) resultBox.hidden = false;
    if (status) status.textContent = `${quality.registrations.toLocaleString()} eligible registrations from ${quality.unique_students.toLocaleString()} unique students; ${quality.excluded_not_registered_or_already_unregistered_by_day_28.toLocaleString()} registrations excluded because they were not in the day-28 risk set.`;
    const sourceRegistrations = quality.source_table_rows?.studentRegistration;
    if (sourceRegistrations != null) {
        setText('oulad-record-count', `${sourceRegistrations.toLocaleString()} source registrations`);
    }
    setText('oulad-eligible-registration-count', `${quality.registrations.toLocaleString()} eligible day-28 registrations`);
    setText('oulad-eligible-student-count', `${quality.unique_students.toLocaleString()} eligible distinct students`);
    const missingness = quality.missingness;
    if (missingness?.rows != null && missingness.total_missing_cells != null && Array.isArray(missingness.source_tables)) {
        const moduleMissingness = missingness.early_assessment_mean_missing_by_group?.code_module || [];
        const highestMissingModules = [...moduleMissingness]
            .sort((left, right) => right.missing_percent - left.missing_percent)
            .slice(0, 2)
            .map(item => `${item.group} (${item.missing_percent.toFixed(2)}%)`)
            .join(' and ');
        const outcomeMissingness = missingness.early_assessment_mean_missing_by_outcome || {};
        const outcomeSummary = Object.entries(outcomeMissingness)
            .map(([label, item]) => `${label} ${item.missing_percent.toFixed(2)}%`)
            .join('; ');
        setText('oulad-data-quality-summary',
            `The day-28 analytical table has ${missingness.rows.toLocaleString()} registrations, ${missingness.predictor_columns} predictors and one target. It contains ${missingness.total_missing_cells.toLocaleString()} missing predictor cells (${missingness.predictor_missing_percent.toFixed(2)}% of predictor cells; ${missingness.overall_missing_percent.toFixed(2)}% of all cells). The seven raw source tables contain ${missingness.raw_source_total_missing_cells.toLocaleString()} nulls among ${missingness.raw_source_total_cells.toLocaleString()} cells (${missingness.raw_source_overall_missing_percent.toFixed(4)}%); many are structural or outside the model predictors. Early-score missingness is highest in modules ${highestMissingModules}; by outcome it is ${outcomeSummary}. These are descriptive associations, not causes.`);
        renderMissingnessChart(missingness);
    } else {
        showError('oulad-data-quality-summary', 'Generated missing-value audit is unavailable.');
    }
    renderPreprocessingComparison(preprocessingComparison, data.classes);
    const targetCanvas = document.getElementById('ouladTargetChart');
    if (targetCanvas) {
        const targetLabels = Object.keys(quality.target_counts);
        makeChart('ouladTargetChart', { type: 'bar', data: { labels: targetLabels, datasets: [{
            label: 'Eligible registrations', data: targetLabels.map(label => quality.target_counts[label]),
            backgroundColor: targetLabels.map((_, i) => palette[i % palette.length]), borderRadius: 5
        }] }, options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } } });
    }
    setText('oulad-accuracy', percent(data.holdout.accuracy));
    setText('oulad-f1', percent(data.holdout.weighted_f1));
    setText('oulad-auc', data.holdout.roc_auc_ovr_macro == null ? 'N/A' : percent(data.holdout.roc_auc_ovr_macro));
    setText('oulad-model', data.selected_model);
    const classes = data.classes;
    const matrix = data.holdout.confusion_matrix;
    const confusionBody = document.querySelector('#ouladConfusion tbody');
    const reportBody = document.querySelector('#ouladClassReport tbody');
    if (confusionBody && Array.isArray(classes) && Array.isArray(matrix) &&
        classes.every((_, index) => Array.isArray(matrix[index]))) {
        confusionBody.innerHTML = `<tr><th>Actual \\ Predicted</th>${classes.map(label => `<th>Pred. ${label}</th>`).join('')}</tr>` +
            classes.map((label, i) => `<tr><th>True ${label}</th>${matrix[i].map(value => `<td>${value}</td>`).join('')}</tr>`).join('');
    }
    if (reportBody && Array.isArray(classes) && data.holdout.classification_report) {
        reportBody.innerHTML = classes.map(label => {
            const row = data.holdout.classification_report[label];
            return `<tr><th>${label}</th><td>${percent(row.precision)}</td><td>${percent(row.recall)}</td><td>${percent(row['f1-score'])}</td><td>${row.support.toLocaleString()}</td></tr>`;
        }).join('');
    }
    renderModelComparison('ouladModelChart', data.cv);
    renderModelTable('ouladModelTable', data.cv, data.selected_model);
    renderClassPerformance('ouladClassChart', data);
    renderImportance('ouladImportanceChart', data.feature_importance, 'ouladImportanceTable');
    renderClusters('ouladClusterChart', 'ouladClusterText', data.clustering, 'ouladClusterDistributionChart');
    const withdrawnIndex = classes.indexOf('Withdrawn');
    const withdrawnReport = data.holdout.classification_report?.Withdrawn;
    if (withdrawnIndex >= 0 && withdrawnReport) {
        const withdrawnRow = matrix[withdrawnIndex];
        const totalWithdrawn = withdrawnRow.reduce((sum, value) => sum + value, 0);
        const detected = withdrawnRow[withdrawnIndex];
        setText('oulad-lesson-accuracy', percent(data.holdout.accuracy));
        setText('oulad-lesson-detected', detected.toLocaleString());
        setText('oulad-lesson-total', totalWithdrawn.toLocaleString());
        setText('oulad-lesson-missed', (totalWithdrawn - detected).toLocaleString());
        setText('oulad-lesson-precision', percent(withdrawnReport.precision));
        setText('oulad-lesson-recall', percent(withdrawnReport.recall));
        setText('oulad-lesson-f1', percent(withdrawnReport['f1-score']));
    }
}

fetch('results/analysis.json', { cache: 'no-cache' })
    .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
    .then(results => {
        const uci = results.experiments.uci;
        if (!uci) throw new Error('UCI experiment data is missing');
        if (results.generated_at) {
            const date = new Date(`${results.generated_at}T00:00:00Z`);
            setText('latest-analysis-date', date.toLocaleDateString('en-GB', {
                day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC'
            }));
        }
        renderHeroDistribution(uci);
        renderDistribution(uci);
        renderModelComparison('modelChart', uci.cv);
        renderModelTable('uciModelTable', uci.cv, uci.selected_model);
        renderConfusionAndReport(uci);
        renderClassPerformance('uciClassChart', uci);
        renderImportance('uciImportanceChart', uci.feature_importance, 'uciImportanceTable');
        renderClusters('uciClusterChart', 'uciClusterText', uci.clustering, 'uciClusterDistributionChart');
        const selectedScores = uci.cv[uci.selected_model];
        const correct = uci.holdout.confusion_matrix.reduce((sum, row, index) => sum + row[index], 0);
        const total = uci.holdout.confusion_matrix.flat().reduce((sum, value) => sum + value, 0);
        setText('uci-cv-weighted-f1', percent(selectedScores.weighted_f1[0]));
        setText('uci-cv-selected-model', `Selected by training-CV weighted F1: ${uci.selected_model}`);
        setText('uci-holdout-accuracy', percent(uci.holdout.accuracy));
        setText('uci-holdout-correct', `${correct.toLocaleString()} of ${total.toLocaleString()} holdout predictions correct`);
        setText('uci-holdout-auc', percent(uci.holdout.roc_auc_ovr_macro));
        setText('results-load-status', typeof Chart === 'undefined'
            ? 'Analysis data loaded, but Chart.js did not load; charts are unavailable.'
            : 'Generated analysis results loaded from results/analysis.json.');
        if (results.experiments.oulad) {
            renderOulad(results.experiments.oulad, results.preprocessing_comparison?.oulad);
        } else {
            showOuladUnavailable('OULAD results are missing from the generated JSON.');
        }
    })
    .catch(error => {
        const message = `Generated results could not be loaded (${error.message}). Serve the project over HTTP and verify results/analysis.json.`;
        const status = document.getElementById('oulad-result-status');
        if (status) status.textContent = message;
        showError('uci-result-status', `Generated UCI results could not be loaded (${error.message}).`);
        showError('results-load-status', message);
        document.querySelectorAll('tbody').forEach(body => {
            const table = body.closest('table');
            const columns = Math.max(1, table.querySelectorAll('thead th').length);
            body.innerHTML = `<tr><td colspan="${columns}">Generated results could not be loaded.</td></tr>`;
        });
        setText('latest-analysis-date', 'Unavailable');
        showOuladUnavailable(message);
        console.error('Unable to load generated experiment results:', error);
    });
