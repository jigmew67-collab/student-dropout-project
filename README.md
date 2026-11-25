# Student Dropout Prediction research project

The project keeps two independent studies. Experiment 1 reproduces the Portuguese UCI final-outcome classification task. Experiment 2 studies subsequent withdrawal among Open University registrations active at a day-28 prediction landmark. Their outcomes, features and evaluation populations differ; the datasets are never concatenated.

## Reproduce

Use Python 3.11.9 and the pinned dependencies in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
python scripts/download_oulad.py
python analysis.py
```

The downloader retrieves the original 46,750,706-byte OULAD archive from its Figshare file record and verifies MD5 `7412686fd77cf0e0ee1e8c3e9b354308`. The archive stays under ignored `data/raw/`; it is not committed. The complete analysis writes aggregate-only `results/analysis.json` and `results.txt` after both experiments succeed. The original UCI `data.csv` is included in the repository. If OULAD is unavailable, the run stops without replacing existing complete outputs with partial results.

## Experiments

**UCI Portugal.** 4,424 rows, 36 predictors, and Dropout / Enrolled / Graduate outcomes. The published features include first- and second-semester results, so this is retrospective final-outcome classification rather than an early-warning estimate. A stratified 20% holdout is fixed before model selection. Five-fold stratified cross-validation on the remaining records selects the model by weighted F1. Categorical codes are one-hot encoded inside each fold.

**OULAD United Kingdom.** The analysis joins the original seven relational tables at the student–module–presentation level. It defines a day-28 risk set: registrations must have started by day 28 and not ended by then. The target is subsequent `Withdrawn`; `Fail`, `Pass`, and `Distinction` remain non-withdrawal outcomes. Assessment submission signals are limited to submissions by day 28, and VLE clicks are aggregated through day 28. Final outcome, student ID, unregistration date, and post-cutoff activity do not enter the predictors. A stratified group split keeps every registration belonging to an individual student in only one partition; the holdout is untouched during five-fold training-only CV.

Both studies compare five model configurations: four classifiers (Logistic Regression, Decision Tree, Random Forest, and Histogram Gradient Boosting) and one majority-class baseline (DummyClassifier/Most Frequent). Weighted F1 selects among classifiers. Accuracy, weighted precision/recall/F1, macro F1, ROC-AUC, confusion matrices, and per-class scores are reported where defined. Permutation importance is calculated on the holdout and measures predictive reliance, not causal effect. K-Means uses standardized and one-hot encoded predictors without target labels; silhouette scores guide candidate k, with a sampled PCA view for display. Outcome distributions across clusters are descriptive only.

## Missing values and preprocessing comparison

The included UCI CSV has 4,424 rows, 36 predictors and one target, with no missing cells. The OULAD day-28 analytical table has 27,515 eligible registrations, 31 predictors and one target. Its missing predictor values are audited by feature and student-group context; the raw seven source tables are reported separately because many raw nulls are structural or not model inputs. In particular, an unavailable early assessment mean is not a score of zero, and unregistration-date nulls are used only to define eligibility.

The analysis runs three preprocessing variants on the same splits and training-only folds:

- **A — Existing:** median-impute numeric predictors in the pipeline; retain the existing explicit `Unknown` handling for OULAD categorical gaps.
- **B — Fold-fitted imputation:** median-impute numeric predictors and use an explicit `Unknown` category for categorical gaps inside each fold's pipeline.
- **C — Fold-fitted imputation with missingness indicators:** use B's imputation and add indicators for predictor missingness observed in that training fold.

The selected variant is determined only by the best model's mean training-CV weighted F1; holdout results are evaluation-only. No missing value is automatically replaced with zero. Exact audit counts, each variant's CV and holdout metrics, class reports, confusion matrices, and the selected variant are written to `results/analysis.json` and `results.txt`. The final selected approach and any performance change are described in the research paper and website after a complete UCI/OULAD run.

## Data sources

- UCI: [Predict Students' Dropout and Academic Success](https://doi.org/10.24432/C5MC89), Portugal. The data file is published under CC BY 4.0.
- OULAD: [Open University dataset page](https://analyse.kmi.open.ac.uk/open_dataset), original [Figshare record](https://doi.org/10.6084/m9.figshare.5081998.v1), CC BY 4.0. Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics dataset*. Scientific Data, 4, 170171. https://doi.org/10.1038/sdata.2017.171

See [RESEARCH_PAPER.md](RESEARCH_PAPER.md) for the study narrative, dataset limitations, references, and generated findings. The static site is served from the repository root for GitHub Pages.
