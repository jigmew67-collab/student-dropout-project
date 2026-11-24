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


// =========================================================
// CLASS DISTRIBUTION — DOUGHNUT CHART
// =========================================================

new Chart(document.getElementById('classChart'), {
    type: 'doughnut',

    data: {
        labels: [
            'Graduate',
            'Dropout',
            'Enrolled'
        ],

        datasets: [
            {
                data: [
                    2209,
                    1421,
                    794
                ],

                backgroundColor: [
                    lime,
                    '#57a77c',
                    '#dce5df'
                ],

                borderWidth: 0
            }
        ]
    },

    options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '72%',

        plugins: {
            legend: {
                position: 'right',

                labels: {
                    color: '#c5d2cc',
                    boxWidth: 8,
                    usePointStyle: true,

                    font: {
                        size: 9
                    }
                }
            }
        }
    }
});


// =========================================================
// CLASS DISTRIBUTION — BAR CHART
// =========================================================

new Chart(document.getElementById('distributionChart'), {
    type: 'bar',

    data: {
        labels: [
            'Graduate',
            'Dropout',
            'Enrolled'
        ],

        datasets: [
            {
                data: [
                    2209,
                    1421,
                    794
                ],

                backgroundColor: [
                    green,
                    '#75ad91',
                    '#bdd2c5'
                ],

                borderRadius: 8,
                borderSkipped: false
            }
        ]
    },

    options: {
        plugins: {
            legend: {
                display: false
            }
        },

        scales: {
            y: {
                beginAtZero: true,

                grid: {
                    color: '#edf0ec'
                }
            },

            x: {
                grid: {
                    display: false
                }
            }
        }
    }
});


// =========================================================
// MODEL PERFORMANCE CHART
// =========================================================

new Chart(document.getElementById('modelChart'), {
    type: 'bar',

    data: {
        labels: [
            'Baseline',
            'Logistic Reg.',
            'Decision Tree',
            'Random Forest',
            'Histogram Gradient Boosting'
        ],

        datasets: [
            {
                label: 'Accuracy %',

                data: [
                    49.93,
                    77.62,
                    68.01,
                    77.39,
                    77.82
                ],

                backgroundColor: green,
                borderRadius: 6
            },

            {
                label: 'Weighted F1 %',

                data: [
                    33.25,
                    76.38,
                    68.29,
                    75.30,
                    76.97
                ],

                backgroundColor: '#b9e86b',
                borderRadius: 6
            }
        ]
    },

    options: {
        indexAxis: 'y',

        plugins: {
            legend: {
                labels: {
                    boxWidth: 9,
                    usePointStyle: true,

                    font: {
                        size: 9
                    }
                }
            }
        },

        scales: {
            x: {
                min: 0,
                max: 100,

                grid: {
                    color: '#edf0ec'
                }
            },

            y: {
                grid: {
                    display: false
                }
            }
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

btn.addEventListener('click', () => {
    links.classList.toggle('open');
});


// Close mobile menu after clicking a navigation link
document.querySelectorAll('.nav-links a').forEach((link) => {
    link.addEventListener('click', () => {
        links.classList.remove('open');
    });
});

// Generated experiment outputs drive the interactive interpretation charts.
const palette = ['#2c7a57', '#b9e86b', '#7bb6a0', '#7692a5', '#d69c59', '#a788b5'];
const percent = value => `${(value * 100).toFixed(2)}%`;

function renderImportance(canvasId, rows) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !rows?.length) return;
    new Chart(canvas, {
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
    if (!canvas || !clustering?.pca_points) return;
    const points = clustering.pca_points;
    const clusters = [...new Set(points.map(point => point.cluster))].sort((a, b) => a - b);
    new Chart(canvas, {
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
    const resultBox = document.getElementById('oulad-results');
    if (resultBox) resultBox.hidden = false;
    if (status) status.textContent = `${quality.registrations.toLocaleString()} eligible registrations from ${quality.unique_students.toLocaleString()} unique students; ${quality.excluded_not_registered_or_already_unregistered_by_day_28.toLocaleString()} registrations excluded because they were not in the day-28 risk set.`;
    document.getElementById('oulad-record-count').textContent = `${quality.registrations.toLocaleString()} eligible registrations`;
    document.getElementById('oulad-student-count').textContent = `${quality.unique_students.toLocaleString()} distinct students`;
    const targetCanvas = document.getElementById('ouladTargetChart');
    if (targetCanvas) {
        const targetLabels = Object.keys(quality.target_counts);
        new Chart(targetCanvas, { type: 'bar', data: { labels: targetLabels, datasets: [{
            label: 'Eligible registrations', data: targetLabels.map(label => quality.target_counts[label]),
            backgroundColor: targetLabels.map((_, i) => palette[i % palette.length]), borderRadius: 5
        }] }, options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } } });
    }
    document.getElementById('oulad-accuracy').textContent = percent(data.holdout.accuracy);
    document.getElementById('oulad-f1').textContent = percent(data.holdout.weighted_f1);
    document.getElementById('oulad-auc').textContent = data.holdout.roc_auc_ovr_macro == null ? 'N/A' : percent(data.holdout.roc_auc_ovr_macro);
    document.getElementById('oulad-model').textContent = data.selected_model;
    const classes = data.classes;
    const matrix = data.holdout.confusion_matrix;
    document.querySelector('#ouladConfusion tbody').innerHTML = `<tr><th>Actual \\ Predicted</th>${classes.map(label => `<th>${label}</th>`).join('')}</tr>` + classes.map((label, i) => `<tr><th>${label}</th>${matrix[i].map(value => `<td>${value}</td>`).join('')}</tr>`).join('');
    document.querySelector('#ouladClassReport tbody').innerHTML = classes.map(label => {
        const row = data.holdout.classification_report[label];
        return `<tr><th>${label}</th><td>${percent(row.precision)}</td><td>${percent(row.recall)}</td><td>${percent(row['f1-score'])}</td><td>${row.support.toLocaleString()}</td></tr>`;
    }).join('');
    const modelLabels = Object.keys(data.cv);
    new Chart(document.getElementById('ouladModelChart'), { type: 'bar', data: { labels: modelLabels, datasets: [
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
        renderImportance('uciImportanceChart', uci.feature_importance);
        renderClusters('uciClusterChart', 'uciClusterText', uci.clustering);
        renderOulad(results.experiments.oulad);
    })
    .catch(error => {
        const status = document.getElementById('oulad-result-status');
        if (status) status.textContent = `Generated results could not be loaded (${error.message}). Run analysis.py and serve the project over HTTP.`;
        console.error('Unable to load generated experiment results:', error);
    });
