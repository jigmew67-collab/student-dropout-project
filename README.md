# Student Dropout Prediction Using Machine Learning in Portuguese Higher Education

ICT324 research project for Rangsit University International College. The active study uses only the UCI Portuguese higher-education dataset included in `data.csv`.

## Reproduce the analysis

The analysis was verified with Python 3.11.9 and the pinned packages in `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
python analysis.py
```

`analysis.py` reads the included semicolon-delimited CSV and regenerates:

- `results/analysis.json` — aggregate metrics and chart data used by the static website.
- `results/analysis-data.js` — the same generated results for browsers that open the site directly from disk, where `fetch()` cannot read local JSON files.
- `results.txt` — a readable run summary.

The script does not download data or modify `data.csv`. Model selection uses stratified five-fold cross-validation on the training partition; the stratified 20% holdout is reserved for final evaluation. Preprocessing is fitted within each model pipeline.

## Research scope

The dataset contains 4,424 records, 36 predictors and three recorded final outcomes: Dropout, Enrolled and Graduate. It includes first- and second-semester features, so the task is retrospective final-outcome classification, not prospective early-warning prediction.

The project compares a most-frequent baseline, Logistic Regression, Decision Tree, Random Forest and Histogram Gradient Boosting. It reports class-specific holdout performance, permutation importance and target-excluded exploratory K-Means clusters. Feature importance and cluster outcome shares are descriptive; they do not establish causation.

See `RESEARCH_PAPER.md` for the study narrative, preparation details, evaluation, limitations and references. The website is static and can be served from the repository root on GitHub Pages. When previewing locally, use a small web server (for example, `python -m http.server`) so the browser can load the generated JSON.
