# Student Dropout Prediction Using Machine Learning: Separate Studies of Portuguese Higher Education and UK Online Learning

## Abstract

This project evaluates student outcome classification in two public higher-education datasets with different populations, labels, and prediction times. The UCI Portugal dataset contains 4,424 student records and supports retrospective three-class classification of Dropout, Enrolled, and Graduate using 36 predictors. A stratified 20% holdout was reserved before five-fold stratified cross-validation on the training partition. Histogram Gradient Boosting achieved 75.37% holdout accuracy, 0.748 weighted F1, 0.686 macro F1, and 0.879 macro one-vs-rest ROC-AUC. The Open University Learning Analytics Dataset (OULAD) experiment instead predicts subsequent withdrawal among registrations still active at day 28. It includes 27,515 eligible registrations from 24,826 students after applying the landmark risk-set definition. A stratified group split prevents a student's registrations from crossing partitions. The selected Histogram Gradient Boosting model achieved 81.68% holdout accuracy and 0.750 weighted F1, but only 5.58% recall for Withdrawn. This imbalance shows why accuracy alone is inadequate. Holdout permutation importance ranked tuition-fee status and second-semester approved units highly for UCI, and early assessment mean score and early VLE activity for OULAD. Target-excluded K-Means produced exploratory clusters with modest silhouette separation. These predictive associations and profiles are descriptive, not causal. Because the tasks differ, scores are not direct measures of national or institutional dropout risk.

## 1. Introduction and research background

Student withdrawal can interrupt educational progress and carry academic, social, and financial costs. Institutions may use student records to identify patterns that help prioritize timely support. Such analysis must preserve the distinction between a statistical association and an intervention that improves student outcomes. Educational data mining studies use demographic, academic, assessment, and digital engagement information, but reported performance depends on the population, prediction horizon, label definition, and evaluation design (Romero & Ventura, 2020; Realinho et al., 2022).

The UCI dataset introduced by Realinho et al. (2022) combines demographic and socioeconomic data with academic results recorded at the end of both semesters. It labels final outcomes as Dropout, Enrolled, or Graduate. OULAD provides anonymized student demographics, assessment records, registrations, course metadata, and daily VLE interactions for seven Open University modules (Kuzilek et al., 2017). It records four final results: Withdrawn, Fail, Pass, and Distinction. OULAD's longitudinal activity data permits a bounded prediction landmark, whereas the UCI dataset as distributed includes later academic performance.

## 2. Research problem, gap, and significance

Many dropout prediction reports are difficult to compare because the term “dropout” can refer to different outcomes and because features may be observed at different points in a student's study. A model using second-semester results answers a different question from an early-warning model using only the first four weeks of activity. The present study addresses this methodological gap by retaining each dataset separately, stating the prediction time and target for each, and preventing student overlap across OULAD partitions.

The work compares a consistent set of classification algorithms within each dataset, interprets predictors using holdout permutation importance, and explores unlabeled student profiles with K-Means. The intended significance is methodological: to show which performance claims each dataset can support and where further prospective validation is necessary. These models are not clinical or administrative decision systems and should not be used to deny services or determine an individual's academic future.

## 3. Objectives and research questions

The objectives are to (1) reproduce the UCI multiclass experiment, (2) construct a day-28 OULAD withdrawal experiment from the original relational tables, (3) compare five classifiers within each task, (4) rank predictive features without causal claims, and (5) examine target-excluded clusters and their outcome composition.

The research questions are:

1. How do Logistic Regression, Decision Tree, Random Forest, and Histogram Gradient Boosting compare with a most-frequent baseline within each dataset's own evaluation design?
2. Which available predictors have the greatest permutation importance on each held-out partition?
3. Do standardized student records form distinguishable exploratory clusters, and how are outcomes distributed among those clusters?
4. How do label definition, prediction time, feature structure, and split design constrain comparison across the datasets?

## 4. Literature review

Realinho et al. (2022) documented the Portuguese student dataset and its three-class final outcome. Their description makes clear that academic features include information calculated at the end of the first and second semesters; therefore, results from the complete feature set should not be described as a pre-enrolment warning. Kuzilek et al. (2017) introduced OULAD as linked student, course, assessment, registration, and VLE records. Its clickstream summaries support research into time-sensitive learner activity.

Educational data mining and learning analytics research has emphasized the use of educational data to understand learning and support decision making (Romero & Ventura, 2020). Model evaluation on imbalanced labels should include class-specific precision and recall rather than relying on accuracy alone. This report provides weighted and macro F1 and confusion matrices to show the trade-off between common and less common outcomes. Permutation importance evaluates how much predictive performance changes when a feature's values are shuffled, but importance does not establish a causal mechanism (Fisher, Rudin, & Dominici, 2019). K-Means partitions feature vectors by distance; the silhouette coefficient offers one measure of separation, not proof that a particular number of natural student types exists (MacQueen, 1967; Rousseeuw, 1987).

## 5. Dataset descriptions

### 5.1 UCI Portugal

The UCI Machine Learning Repository's *Predict Students' Dropout and Academic Success* data contain 4,424 records, 36 input features, and the target classes Dropout (1,421), Enrolled (794), and Graduate (2,209). The setting is Portuguese higher education. Features include admission and qualification data; demographic and socioeconomic codes; tuition, debt, and scholarship indicators; curricular-unit evaluation, approval, and grade counts for both semesters; and macroeconomic indicators. In this copy there are no missing values or exact duplicate rows. Seventeen integer-coded variables are treated as categorical and one-hot encoded. The full source feature set spans both semesters, so the task is retrospective final-outcome classification, not an early-warning prediction at a predeclared point.

### 5.2 OULAD United Kingdom

OULAD's original 2017 release contains seven linked CSV tables: studentInfo, studentRegistration, courses, assessments, studentAssessment, vle, and studentVle. The release covers 32,593 student–module–presentation registrations, 28,785 unique students, seven modules, and 22 presentations. Its interaction table contains 10,655,280 daily activity records. This project downloaded the Figshare archive and verified its published MD5 checksum. The full raw archive is excluded from Git; the download script and source citation are included.

The prediction unit is one student–module–presentation registration. The target is **subsequent Withdrawn** versus **Not withdrawn**. Fail is a negative completed outcome, not a dropout label; Pass and Distinction are also retained as non-withdrawal outcomes. The day-28 risk set excludes registrations that began after day 28 and records already unregistered by then. This leaves 27,515 eligible registrations and 24,826 distinct students; 5,078 records are excluded by this eligibility rule. Among eligible records, final results are Pass 12,355, Fail 7,032, Withdrawn 5,105, and Distinction 3,023.

The 31 model features include demographic, course, registration, and course-design fields; submitted assessment count and mean score available by day 28; total VLE clicks by day 28; and clicks by activity type. `id_student`, final result, and `date_unregistration` are never predictors. The registration date and unregistration date are used only to define day-28 eligibility. Late or absent assessment observations produce missing values for early mean score, which is imputed inside each training fold. No student identifiers or row-level records are published. The 28,785 source students can have multiple registrations, which motivates grouping by student in evaluation.

## 6. Data preprocessing

For UCI, the target is kept as a string class, exact duplicate rows and missing cells are checked, and categorical integer codes are one-hot encoded. Numeric values are standardized for Logistic Regression and left unscaled for tree models. Transformations are fit inside model pipelines so each validation fold learns preprocessing only from its training part.

For OULAD, joins use module, presentation, student, and assessment keys. VLE events after day 28 are discarded before aggregating clicks by activity type. Assessment score features include only unbanked submissions submitted by day 28. Registration dates define the active risk set, while final result and unregistration date are removed from the predictor matrix. Missing numeric values are imputed by the median and categorical values by the most frequent training-fold value; categorical fields are one-hot encoded with unseen categories ignored. The source data's distinct table sizes and joins are reported in the generated result file. Seven identical predictor rows occur after student IDs are omitted; these are feature-equivalent observations, not duplicated registration keys.

## 7. Research methodology

The workflow follows CRISP-DM (Chapman et al., 2000): define the educational prediction tasks, inspect each source, prepare task-specific features, compare classifiers, evaluate on held-out records, and publish aggregate results with limitations. The datasets are never merged.

### 7.1 Classification and evaluation

Both experiments compare a most-frequent DummyClassifier, Logistic Regression, Decision Tree, Random Forest (300 trees), and Histogram Gradient Boosting. Random seed 42 is fixed. UCI uses a stratified 80/20 split; five-fold stratified cross-validation is run on the 80% training set only. OULAD uses a stratified group fold split: one fold is reserved as the approximately 20% holdout and all five cross-validation folds are formed from the remaining records, grouped by `id_student`. The overlap audit reports zero students shared by OULAD train and holdout. Weighted F1 selects the model; the holdout is evaluated once.

The evaluation reports accuracy, weighted precision, weighted recall, weighted F1, macro F1, ROC-AUC where defined, a confusion matrix, and per-class precision, recall, F1, and support. Weighted metrics reflect class frequency; macro F1 weights the classes equally. No threshold search or resampling is performed.

### 7.2 Feature interpretation

A 300-tree Random Forest is fitted on each training partition. Permutation importance is measured on the untouched holdout with five shuffles per feature and weighted F1 as the score. Scores quantify predictive reliance under this fitted model and sample; they neither establish causation nor guarantee stable importance in another institution or prediction period.

### 7.3 Clustering

The target is removed before clustering. Numeric features are standardized and categorical features one-hot encoded. K-Means is fit for k=2 through k=6 with ten initializations and random seed 42. The highest silhouette among these candidates selects k. Principal component analysis projects a seeded sample of up to 2,000 records into two dimensions for display. Outcome shares are computed after clustering solely to describe clusters. Cluster profiles and outcome compositions are descriptive and do not predict an individual outcome.

## 8. Results

### 8.1 UCI classification

The selected model was Histogram Gradient Boosting, with mean training-only CV weighted F1 of 0.7697 (SD 0.0128), accuracy 0.7782 (SD 0.0116), macro F1 0.7110 (SD 0.0168), and macro OvR AUC 0.8891 (SD 0.0109). The untouched holdout contained 885 records. Holdout accuracy was 0.7537, weighted precision 0.7446, weighted recall 0.7537, weighted F1 0.7475, macro F1 0.6861, and macro OvR AUC 0.8794.

| Actual class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Dropout | 0.7707 | 0.7218 | 0.7455 | 284 |
| Enrolled | 0.5000 | 0.4277 | 0.4610 | 159 |
| Graduate | 0.8157 | 0.8914 | 0.8519 | 442 |

The confusion matrix (actual rows; predicted columns in Dropout, Enrolled, Graduate order) is `[[205, 40, 39], [41, 68, 50], [20, 28, 394]]`. Enrolled is the weakest class by recall and F1. The holdout permutation ranking begins with tuition-fee status, second-semester approved-unit count, second-semester grade, first-semester approved-unit count, and second-semester evaluation count. These features include later academic information and should not be interpreted as an early-warning signal or as causes of withdrawal.

### 8.2 OULAD day-28 classification

The selected model was Histogram Gradient Boosting. Its mean CV weighted F1 was 0.7560 (SD 0.0022), accuracy 0.8176 (SD 0.0012), macro F1 0.5151 (SD 0.0059), and ROC-AUC 0.6960 (SD 0.0098). The grouped holdout had 5,503 registrations from students absent from training. Holdout accuracy was 0.8168, weighted precision 0.7738, weighted recall 0.8168, weighted F1 0.7503, macro F1 0.4998, and ROC-AUC 0.7094.

| Actual class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Not withdrawn | 0.8215 | 0.9902 | 0.8980 | 4,482 |
| Withdrawn | 0.5644 | 0.0558 | 0.1016 | 1,021 |

The confusion matrix (actual rows, Not withdrawn then Withdrawn; predicted columns in the same order) is `[[4438, 44], [964, 57]]`. The model misses most later withdrawals at its default decision threshold. Its accuracy is close to the majority outcome rate, so accuracy alone would give a misleading impression of usefulness. The largest permutation-importance value is for early mean assessment score, followed by module presentation length and quiz, forum, and URL click totals; the weighted-F1 importance values are small and should be interpreted accordingly. Future work should prioritize recall/precision trade-offs and calibration with a threshold chosen only inside training data, before any deployment claim.

### 8.3 Feature importance and clustering

UCI's first- and second-semester achievement measures and fee-status feature had the largest holdout permutation effects. For OULAD, early assessment and a small number of early activity aggregates ranked highest. These are predictive associations in their respective tasks and observation windows; they are not evidence that changing a feature changes a student's outcome.

K-Means selected k=3 for UCI (silhouette 0.231) and k=2 for OULAD (silhouette 0.365). These modest silhouette scores indicate limited separation under the chosen feature representation. UCI cluster outcome shares and OULAD cluster profiles are provided in `results/analysis.json` and visualized on the website. No shared cluster naming scheme is imposed across datasets. The cluster solution is exploratory, sensitive to encoding and distance, and not a validated risk model.

### 8.4 Comparative findings

The UCI holdout estimates three-class final-outcome performance on student records, while OULAD estimates day-28 subsequent-withdrawal detection on course registrations with unseen students. The OULAD grouped holdout is intentionally more demanding with respect to student independence, and the OULAD positive class is less frequent in the analytic risk set. Features also differ: UCI includes semester grades and approvals, whereas OULAD uses early online engagement and assessment signals. Consequently, the numerical scores are not a direct comparison of country performance or underlying dropout rates. OULAD's low withdrawal recall suggests the current default-threshold classifier is not adequate for early warning despite respectable accuracy.

## 9. Limitations

UCI is one institution and its outcome classes reflect final study status; second-semester predictors limit prospective use. OULAD covers seven selected modules and historical 2013–14 presentations at a distance-teaching university; its day-28 cohort excludes students not registered by then and withdrawals already recorded by then. The OULAD target intentionally distinguishes withdrawal from fail, but non-withdrawal combines unsuccessful and successful outcomes. Its grouped split estimates generalization to students unseen in training, not necessarily to future cohorts or entirely new modules. VLE activity and assessment availability vary by course. Class imbalance, default probability thresholds, and no temporal validation constrain performance claims. Permutation importance can distribute credit among correlated variables. K-Means and PCA depend on preprocessing and do not establish natural student types. Anonymized aggregate results cannot replace local validation, fairness evaluation, governance, or evidence that interventions help students.

## 10. Conclusion and future work

This project provides two reproducible analyses while preserving their distinct target definitions and prediction times. The UCI experiment reproduces a useful retrospective multiclass baseline and confirms that the Enrolled class is harder to identify. OULAD shows that an apparently strong accuracy can coexist with near-zero recall for later withdrawal under class imbalance. Assessment and engagement features carry predictive information in the OULAD experiment, but the current model is not ready to support intervention. Future work should evaluate earlier temporal landmarks, temporal and module-held-out validation, calibrated probabilities, threshold selection within training data, class-sensitive objectives, fairness across protected and socioeconomic groups, and prospective intervention utility. Any operational use should be reviewed with educators and students and should support, not replace, human judgment.

## References

1. Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS.
2. Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously. *Journal of Machine Learning Research, 20*(177), 1–81.
3. Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data, 4*, 170171. https://doi.org/10.1038/sdata.2017.171
4. Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics Dataset* [Data set]. Figshare. https://doi.org/10.6084/m9.figshare.5081998.v1
5. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 281–297).
6. Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data, 7*(11), 146. https://doi.org/10.3390/data7110146
7. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics, 20*, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7
8. Romero, C., & Ventura, S. (2020). Educational data mining and learning analytics: An updated survey. *WIREs Data Mining and Knowledge Discovery, 10*(3), e1355. https://doi.org/10.1002/widm.1355
9. UCI Machine Learning Repository. *Predict Students' Dropout and Academic Success* [Data set]. https://doi.org/10.24432/C5MC89
10. Open University. *Open University Learning Analytics Dataset (OULAD)*. https://analyse.kmi.open.ac.uk/open_dataset
