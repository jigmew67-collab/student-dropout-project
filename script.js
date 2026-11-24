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

Chart.defaults.font.family = 'Manrope';
Chart.defaults.color = '#66766f';

const makeChart = (canvasId, config) => {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    if (typeof Chart === 'undefined') return null;
    if (Chart.getChart(canvas)) Chart.getChart(canvas).destroy();
    return new Chart(canvas, config);
};

// Static fallbacks are built from the verified project results before JSON
// loading completes, so a slow or failed fetch never leaves chart spaces empty.
makeChart('classChart', {
    type: 'doughnut',
    data: { labels: ['Graduate', 'Dropout', 'Enrolled'], datasets: [{
        data: [2209, 1421, 794], backgroundColor: [lime, '#57a77c', '#dce5df'], borderWidth: 0
    }] },
    options: { responsive: true, maintainAspectRatio: false, cutout: '72%',
        plugins: { legend: { position: 'right', labels: { color: '#c5d2cc', boxWidth: 8, usePointStyle: true, font: { size: 9 } } } } }
});

makeChart('distributionChart', {
    type: 'bar',
    data: { labels: ['Graduate', 'Dropout', 'Enrolled'], datasets: [{
        data: [2209, 1421, 794], backgroundColor: [green, '#75ad91', '#bdd2c5'], borderRadius: 8, borderSkipped: false
    }] },
    options: { plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, grid: { color: '#edf0ec' } }, x: { grid: { display: false } } } }
});

makeChart('modelChart', {
    type: 'bar',
    data: { labels: ['Baseline', 'Logistic Reg.', 'Decision Tree', 'Random Forest', 'Histogram Gradient Boosting'], datasets: [
        { label: 'Accuracy %', data: [49.93, 77.62, 68.01, 77.39, 77.82], backgroundColor: green, borderRadius: 6 },
        { label: 'Weighted F1 %', data: [33.25, 76.38, 68.29, 75.30, 76.97], backgroundColor: lime, borderRadius: 6 }
    ] },
    options: { indexAxis: 'y', plugins: { legend: { labels: { boxWidth: 9, usePointStyle: true, font: { size: 9 } } } },
        scales: { x: { min: 0, max: 100, grid: { color: '#edf0ec' } }, y: { grid: { display: false } } } }
});

makeChart('ouladTargetChart', {
    type: 'bar',
    data: { labels: ['Pass', 'Fail', 'Withdrawn', 'Distinction'], datasets: [{
        label: 'Eligible registrations', data: [12355, 7032, 5105, 3023],
        backgroundColor: ['#2c7a57', '#b9e86b', '#7bb6a0', '#7692a5'], borderRadius: 5
    }] },
    options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }
});

makeChart('ouladModelChart', {
    type: 'bar',
    data: { labels: ['Baseline', 'Logistic Reg.', 'Decision Tree', 'Random Forest', 'Histogram Gradient Boosting'], datasets: [
        { label: 'Accuracy', data: [81.45, 81.46, 71.17, 81.53, 81.76], backgroundColor: green },
        { label: 'Weighted F1', data: [73.12, 73.86, 71.74, 74.07, 75.60], backgroundColor: lime }
    ] },
    options: { indexAxis: 'y', scales: { x: { min: 0, max: 100 } } }
});

makeChart('ouladImportanceChart', {
    type: 'bar',
    data: { labels: ['early_assessment_mean_score', 'module_presentation_length', 'clicks_quiz', 'clicks_forumng', 'clicks_url'], datasets: [{
        label: 'Weighted F1 decrease after shuffling',
        data: [0.0081416467, 0.0032633238, 0.0031755827, 0.0018620410, 0.0016982953],
        backgroundColor: green, borderRadius: 5
    }] },
    options: { indexAxis: 'y', plugins: { legend: { display: false } },
        scales: { x: { beginAtZero: true, title: { display: true, text: 'Permutation importance' } }, y: { grid: { display: false } } } }
});

makeChart('uciImportanceChart', {
    type: 'bar',
    data: { labels: ['Tuition fees up to date', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)', 'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (evaluations)'], datasets: [{
        label: 'Weighted F1 decrease after shuffling',
        data: [0.0423245890, 0.0420098459, 0.0112158882, 0.0098498019, 0.0074703988],
        backgroundColor: green, borderRadius: 5
    }] },
    options: { indexAxis: 'y', plugins: { legend: { display: false } },
        scales: { x: { beginAtZero: true, title: { display: true, text: 'Permutation importance' } }, y: { grid: { display: false } } } }
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
        links.classList.toggle('open');
    });
}


// Close mobile menu after clicking a navigation link
document.querySelectorAll('.nav-links a').forEach((link) => {
    link.addEventListener('click', () => {
        links.classList.remove('open');
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
        options: { indexAxis: 'y', plugins: { legend: { labels: { boxWidth: 9, usePointStyle: true, font: { size: 9 } } } }, scales: { x: { min: 0, max: 100, grid: { color: '#edf0ec' } }, y: { grid: { display: false } } } }
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

function renderImportance(canvasId, rows) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    if (!rows?.length) {
        showError(`${canvasId}-status`, 'Permutation-importance results are unavailable.');
        return;
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

function renderClusters(canvasId, textId, clustering) {
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
    if (text) text.innerHTML = `<p class="chart-note">Selected k = ${clustering.selected_k}; silhouette = ${clustering.silhouette.toFixed(3)}. ${clustering.clusters.map(item => {
        const distribution = Object.entries(item.outcome_share).map(([label, share]) => `${label} ${(share * 100).toFixed(1)}%`).join(', ');
        const numeric = Object.entries(item.profile?.numeric_median || {}).map(([key, value]) => `${key} median ${Number(value).toFixed(1)}`);
        const categorical = Object.entries(item.profile?.categorical_mode || {}).map(([key, value]) => `${key} most common ${value}`);
        const profile = [...numeric, ...categorical].join('; ');
        return `Cluster ${item.cluster + 1} (n=${item.size.toLocaleString()}): ${distribution}. Profile: ${profile}`;
    }).join(' · ')}</p>`;
}

function renderOulad(exp) {
    const status = document.getElementById('oulad-result-status');
    if (!exp || exp.status === 'not_run') {
        if (status) status.textContent = 'OULAD experiment has no generated results in this build.';
        return;
    }
    const data = exp;
    const quality = data.data_quality;
    if (!quality || !data.holdout || !data.cv) {
        if (status) status.textContent = 'OULAD generated results are incomplete. Missing data-quality or holdout metrics.';
        return;
    }
    const resultBox = document.getElementById('oulad-results');
    if (resultBox) resultBox.hidden = false;
    if (status) status.textContent = `${quality.registrations.toLocaleString()} eligible registrations from ${quality.unique_students.toLocaleString()} unique students; ${quality.excluded_not_registered_or_already_unregistered_by_day_28.toLocaleString()} registrations excluded because they were not in the day-28 risk set.`;
    setText('oulad-record-count', `${quality.registrations.toLocaleString()} eligible registrations`);
    setText('oulad-student-count', `${quality.unique_students.toLocaleString()} distinct students`);
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
    document.querySelector('#ouladConfusion tbody').innerHTML = `<tr><th>Actual \\ Predicted</th>${classes.map(label => `<th>${label}</th>`).join('')}</tr>` + classes.map((label, i) => `<tr><th>${label}</th>${matrix[i].map(value => `<td>${value}</td>`).join('')}</tr>`).join('');
    document.querySelector('#ouladClassReport tbody').innerHTML = classes.map(label => {
        const row = data.holdout.classification_report[label];
        return `<tr><th>${label}</th><td>${percent(row.precision)}</td><td>${percent(row.recall)}</td><td>${percent(row['f1-score'])}</td><td>${row.support.toLocaleString()}</td></tr>`;
    }).join('');
    const modelLabels = Object.keys(data.cv);
    makeChart('ouladModelChart', { type: 'bar', data: { labels: modelLabels, datasets: [
        { label: 'Accuracy', data: modelLabels.map(label => data.cv[label].accuracy[0] * 100), backgroundColor: green },
        { label: 'Weighted F1', data: modelLabels.map(label => data.cv[label].weighted_f1[0] * 100), backgroundColor: lime }
    ] }, options: { indexAxis: 'y', scales: { x: { min: 0, max: 100 } } } });
    renderImportance('ouladImportanceChart', data.feature_importance);
    renderClusters('ouladClusterChart', 'ouladClusterText', data.clustering);
}

fetch('results/analysis.json')
    .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
    .then(results => {
        const uci = results.experiments.uci;
        if (!uci) throw new Error('UCI experiment data is missing');
        renderHeroDistribution(uci);
        renderDistribution(uci);
        renderModelComparison('modelChart', uci.cv);
        renderConfusionAndReport(uci);
        renderImportance('uciImportanceChart', uci.feature_importance);
        renderClusters('uciClusterChart', 'uciClusterText', uci.clustering);
        if (results.experiments.oulad) {
            renderOulad(results.experiments.oulad);
        } else {
            showError('oulad-result-status', 'OULAD results are missing from the generated JSON.');
        }
    })
    .catch(error => {
        const status = document.getElementById('oulad-result-status');
        if (status) status.textContent = `Generated results could not be loaded (${error.message}). Run analysis.py and serve the project over HTTP.`;
        showError('uci-result-status', `Generated UCI results could not be loaded (${error.message}).`);
        console.error('Unable to load generated experiment results:', error);
    });
