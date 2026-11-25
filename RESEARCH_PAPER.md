# Student Dropout Prediction Using Machine Learning in Portuguese Higher Education

**ICT324 research project · Rangsit University International College**

## Abstract

This study examines how machine-learning models classify final student outcomes in the UCI Machine Learning Repository's Portuguese higher-education dataset. The included CSV contains 4,424 records, 36 predictors and three recorded outcomes: Dropout, Enrolled and Graduate. It has no missing cells or exact duplicate rows. A stratified 20% holdout is reserved before five-fold stratified cross-validation on the training partition. Histogram Gradient Boosting is selected by mean training-fold weighted F1. The generated analysis reports 75.37% holdout accuracy, 0.7475 weighted F1, 0.6861 macro F1 and 0.8794 macro one-vs-rest ROC-AUC. Enrolled has the weakest class recall (0.4277), showing why overall accuracy is not a sufficient summary. A separate 300-tree Random Forest is used to calculate holdout permutation importance, and target-excluded K-Means provides descriptive clusters with modest silhouette separation. The dataset includes academic features from both semesters; these results are retrospective classification of recorded final outcomes, not prospective early-warning predictions. Predictive importance does not establish cause. This work's contribution is a transparent, bounded reproduction and discussion of one dataset, not a novel algorithm or validated decision system.

## 1. Introduction and research context

Student attrition can interrupt educational progress and has academic, social and financial consequences. Educational data mining and learning analytics investigate how records can help describe student outcomes and inform support. Such analysis must distinguish predictive association from causation and from an intervention that demonstrably improves outcomes (Romero & Ventura, 2020).

The UCI dataset introduced by Realinho et al. (2021) combines demographic, socioeconomic and academic attributes with three final outcome labels. Academic features include results from the first and second semesters. Therefore, a model using the complete feature set answers a retrospective classification question; it does not establish that dropout can be predicted early enough to intervene. The associated 2022 study reports predictive experiments using this dataset (Realinho et al., 2022).

## 2. Problem, significance and research question

Reported dropout-prediction performance is meaningful only in context: the label definition, feature availability, population and evaluation design all matter. This project makes the prediction target and timing explicit, describes preparation from the original CSV, compares standard classifiers to a simple baseline, and reports class-specific errors.

The research question is: **How well can machine-learning models classify the final outcome recorded in the Portuguese student dataset, and which observed features matter most to those predictions?**

The contribution is methodological and educational. It is a reproducible single-dataset comparison with transparent preprocessing and limitations. The study does not claim methodological novelty, causal discovery, prospective validity, or publication readiness.

## 3. Dataset and data understanding

### 3.1 Source and structure

The study uses only the UCI Machine Learning Repository dataset *Predict Students' Dropout and Academic Success* (Realinho et al., 2021; Realinho et al., 2022). The project includes the original `data.csv` file and does not change its records or labels. The file is semicolon-delimited and has 4,424 rows, 36 predictors and one target column (37 columns total).

The target counts in the included file are:

| Recorded outcome | Records |
|---|---:|
| Dropout | 1,421 |
| Enrolled | 794 |
| Graduate | 2,209 |

Predictors cover demographic and socioeconomic characteristics, application and qualification fields, tuition and scholarship status, curricular-unit participation and results in both semesters, and macroeconomic indicators. Seventeen integer-coded fields are handled as categorical features; the remaining 19 features are numeric. The feature categories reflect the analysis code, not an independent recoding of the source data.

### 3.2 Quality checks

The analysis checks every cell for missingness and checks exact complete-row duplicates before modeling. In the included file, 0 of 163,688 cells are missing and there are 0 exact duplicate rows. No records are removed or modified. No imputation is required for this copy of the dataset. Median and most-frequent imputers are nevertheless defined within the model pipelines, so they would be fitted on training data if missing predictor values were encountered.

## 4. Data preparation

Preparation is intentionally explicit because the released table already contains ready-to-use columns but still requires careful feature typing and evaluation safeguards.

1. **Read and inspect.** `pandas.read_csv` loads `data.csv` with `sep=";"`; surrounding whitespace is removed from column names. The script verifies that the target and expected categorical columns are present and that the three observed labels match the stated classes.
2. **Audit the raw table.** Missing cells and exact duplicate records are counted. The checks do not trigger dropping or altering rows in this dataset.
3. **Separate target.** `X` contains the 36 predictors, while `y` is the `Target` column retained as the strings Dropout, Enrolled and Graduate. Target values are not encoded as predictor features.
4. **Handle feature types in pipelines.** The 17 integer-coded categorical fields are mode-imputed if necessary and one-hot encoded with unseen categories ignored. The 19 numeric features are median-imputed if necessary. Logistic Regression uses standardized numeric inputs; the tree-based estimators use unscaled numeric inputs. Since this CSV has no missing cells, no actual values are imputed in the reported run.
5. **Split and cross-validate.** A stratified 80/20 split with random state 42 creates 3,539 training and 885 holdout records. Five-fold stratified cross-validation is conducted only within the training partition. Weighted F1 selects the model; the holdout is evaluated after selection.

The transformations are components of each model pipeline. Each fold fits its own imputer, encoder and scaler using only that fold's fitting records. The holdout is not used to fit preprocessing or select a model. These choices reduce preprocessing leakage; they do not address every source of bias or replace external validation.

Illustrative operations from `analysis.py`:

```python
dataset = pd.read_csv(ROOT / "data.csv", sep=";")
dataset.columns = [column.replace("\ufeff", "").strip() for column in dataset.columns]
X = dataset.drop(columns="Target")
y = dataset["Target"].astype(str)

train_idx, test_idx = train_test_split(
    np.arange(len(X)),
    test_size=0.20,
    random_state=42,
    stratify=y,
)
```

For example, `Application mode` and `Tuition fees up to date` are one-hot encoded as categorical fields. `Admission grade` and `Curricular units 2nd sem (approved)` remain numeric inputs. Category codes are treated as labels; the analysis does not assume the numeric codes have ordered magnitude.

## 5. Methodology

The research journey follows the six CRISP-DM phases (Chapman et al., 2000): Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment/Research Implications.

### 5.1 Classification

Five model configurations use random state 42:

- Most Frequent `DummyClassifier` baseline
- Logistic Regression with `max_iter=3000`
- Decision Tree
- Random Forest with 300 trees
- Histogram Gradient Boosting

Weighted F1 on training-only cross-validation is the model-selection metric. The holdout is used for final estimates only. No resampling or threshold search is performed. Accuracy, weighted precision, weighted recall, weighted and macro F1, macro one-vs-rest ROC-AUC, a confusion matrix and per-class precision, recall, F1 and support are reported. Weighted scores reflect class frequencies, whereas macro F1 gives each class equal weight.

### 5.2 Feature interpretation

A separate 300-tree Random Forest is fitted on the training partition. Permutation importance is computed on the untouched holdout using five repeats and weighted F1. A feature's score is the measured reduction in this metric when that feature is shuffled. It describes predictive reliance for this fitted model and evaluation sample; it is not a causal effect and can be affected by correlated predictors.

### 5.3 Exploratory clustering

The target is excluded before clustering. Categorical variables are one-hot encoded and numeric variables standardized. K-Means is fit for k=2 through k=6 with ten initializations and random state 42. A seeded sample of up to 2,000 records is used to calculate candidate silhouette scores. Outcome composition is examined only after clustering and is descriptive, not an individual prediction or evidence of natural student types.

## 6. Results

### 6.1 Training-only model comparison

The following are means and standard deviations across five folds of the training partition. Histogram Gradient Boosting has the highest mean weighted F1 in the generated report.

| Model | Accuracy | Weighted F1 | Macro F1 | Macro OvR ROC-AUC |
|---|---:|---:|---:|---:|
| Most Frequent baseline | 0.4993 ± 0.0006 | 0.3325 ± 0.0007 | 0.2220 ± 0.0002 | 0.5000 ± 0.0000 |
| Logistic Regression | 0.7759 ± 0.0080 | 0.7635 ± 0.0090 | 0.7014 ± 0.0144 | 0.8884 ± 0.0088 |
| Decision Tree | 0.6801 ± 0.0064 | 0.6829 ± 0.0108 | 0.6183 ± 0.0143 | 0.7277 ± 0.0104 |
| Random Forest | 0.7739 ± 0.0068 | 0.7530 ± 0.0102 | 0.6819 ± 0.0157 | 0.8866 ± 0.0083 |
| Histogram Gradient Boosting | **0.7782 ± 0.0116** | **0.7697 ± 0.0128** | **0.7110 ± 0.0168** | **0.8891 ± 0.0109** |

### 6.2 Untouched holdout

The 885-record holdout results for the selected Histogram Gradient Boosting model are accuracy 0.7537, weighted precision 0.7446, weighted recall 0.7537, weighted F1 0.7475, macro F1 0.6861, and macro OvR ROC-AUC 0.8794.

| Actual class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Dropout | 0.7707 | 0.7218 | 0.7455 | 284 |
| Enrolled | 0.5000 | 0.4277 | 0.4610 | 159 |
| Graduate | 0.8157 | 0.8914 | 0.8519 | 442 |

Confusion matrix, with rows as actual classes and columns in the order Dropout, Enrolled, Graduate:

|  | Predicted Dropout | Predicted Enrolled | Predicted Graduate |
|---|---:|---:|---:|
| Actual Dropout | 205 | 40 | 39 |
| Actual Enrolled | 41 | 68 | 50 |
| Actual Graduate | 20 | 28 | 394 |

Enrolled is the most difficult class: 68 of 159 are identified correctly, while 41 are classified as Dropout and 50 as Graduate. Class imbalance is one relevant context, but the confusion matrix alone does not establish why these errors occur. The result supports reporting class-specific performance alongside aggregate metrics.

### 6.3 Feature importance and clusters

The generated result file ranks second-semester approved curricular units and tuition-fee status among the leading permutation-importance features. These features may be strongly associated with the final recorded outcomes, but their ranking does not establish that they cause those outcomes. The full ranked list and scores are provided in `results/analysis.json`.

The selected exploratory K-Means solution uses k=3 and has a silhouette score around 0.23 in the generated run, indicating modest separation under this encoded feature space. One cluster has a high observed share of Dropout records. Cluster outcome shares are descriptive summaries after the unsupervised fit; they are not validation of a student typology or evidence of causation.

## 7. Discussion and limitations

The selected model improves substantially on the most-frequent baseline under weighted F1, but its performance is uneven across classes. Enrolled recall is 0.4277, lower than Dropout (0.7218) and Graduate (0.8914). A model with these errors should not be treated as an adequate standalone basis for support decisions.

Most importantly, second-semester features make this an analysis of recorded final outcomes, not a prospective early-warning estimate. The evaluation is a random stratified holdout from the same included dataset, not an external institution, later cohort or prospective deployment. Results may not generalize to other populations or time periods. The study does not measure intervention effectiveness, establish causal mechanisms, or evaluate calibration and fairness. Permutation importance can also be unstable when predictors are correlated, and a modest silhouette score does not demonstrate naturally distinct student groups.

Future research should define an explicit prediction landmark and use only information available at that time, evaluate on newer and external cohorts, investigate class-sensitive metrics and calibration, and study any support intervention with educators and students. Predictions, if evaluated in future, should be advisory and subject to human oversight rather than used to restrict educational opportunities.

## 8. Conclusion

This project provides a reproducible comparison of five model configurations for three recorded final outcomes in the UCI Portuguese higher-education dataset. The selected model achieves 0.7475 weighted F1 on the holdout, while the Enrolled class remains materially harder to identify. Transparent data preparation, class-specific errors and prediction timing are essential to interpreting that result. It is evidence about retrospective classification on this dataset—not proof of causation, prospective early-warning ability or readiness for deployment.

## References

Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS.

Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously. *Journal of Machine Learning Research, 20*(177), 1–81. https://jmlr.org/papers/v20/18-760.html

MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 281–297).

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data, 7*(11), 146. https://doi.org/10.3390/data7110146

Romero, C., & Ventura, S. (2020). Educational data mining and learning analytics: An updated survey. *WIREs Data Mining and Knowledge Discovery, 10*(3), e1355. https://doi.org/10.1002/widm.1355

Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics, 20*, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7
