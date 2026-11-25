"""Reproduce the Portuguese higher-education outcome analysis."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
SEED = 42
CLASSES = ["Dropout", "Enrolled", "Graduate"]
CATEGORICAL = [
    "Marital status", "Application mode", "Course",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "International",
]


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def preprocessor(frame, categorical, numeric, scale):
    numeric_categorical = [
        column for column in categorical
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    string_categorical = [
        column for column in categorical if column not in numeric_categorical
    ]
    transformers = []
    if string_categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            string_categorical,
        ))
    if numeric_categorical:
        transformers.append((
            "coded_categories",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            numeric_categorical,
        ))
    transformers.append((
        "numeric",
        Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler() if scale else "passthrough"),
        ]),
        numeric,
    ))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def make_models(frame, categorical, numeric):
    scaled = preprocessor(frame, categorical, numeric, scale=True)
    unscaled = preprocessor(frame, categorical, numeric, scale=False)
    return {
        "Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([
            ("preprocess", scaled),
            ("model", LogisticRegression(max_iter=3000, random_state=SEED)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocess", unscaled),
            ("model", DecisionTreeClassifier(random_state=SEED)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", unscaled),
            ("model", RandomForestClassifier(
                n_estimators=300, random_state=SEED, n_jobs=-1
            )),
        ]),
        "Histogram Gradient Boosting": Pipeline([
            ("preprocess", unscaled),
            ("model", HistGradientBoostingClassifier(random_state=SEED)),
        ]),
    }


def evaluate(X, y, classes, categorical):
    numeric = [column for column in X.columns if column not in categorical]
    models = make_models(X, categorical, numeric)
    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = {}

    for name, model in models.items():
        fold_scores = {"accuracy": [], "weighted_f1": [], "macro_f1": [], "roc_auc": []}
        for fit_idx, validation_idx in splitter.split(X_train, y_train):
            fold_model = model
            fold_model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
            actual = y_train.iloc[validation_idx]
            predicted = fold_model.predict(X_train.iloc[validation_idx])
            fold_scores["accuracy"].append(accuracy_score(actual, predicted))
            fold_scores["weighted_f1"].append(
                f1_score(actual, predicted, average="weighted")
            )
            fold_scores["macro_f1"].append(
                f1_score(actual, predicted, average="macro")
            )
            probabilities = fold_model.predict_proba(X_train.iloc[validation_idx])
            fold_scores["roc_auc"].append(roc_auc_score(
                actual,
                probabilities,
                labels=fold_model.classes_,
                multi_class="ovr",
                average="macro",
            ))
        cv_scores[name] = {
            metric: [float(np.mean(scores)), float(np.std(scores))]
            for metric, scores in fold_scores.items()
        }

    selected_name = max(
        cv_scores, key=lambda name: cv_scores[name]["weighted_f1"][0]
    )
    selected_model = models[selected_name].fit(X_train, y_train)
    predicted = selected_model.predict(X_test)
    probabilities = selected_model.predict_proba(X_test)
    report = classification_report(
        y_test, predicted, labels=classes, output_dict=True, zero_division=0
    )
    return {
        "selected_model": selected_name,
        "classes": classes,
        "cv": cv_scores,
        "holdout": {
            "accuracy": accuracy_score(y_test, predicted),
            "precision_weighted": precision_score(
                y_test, predicted, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_test, predicted, average="weighted", zero_division=0
            ),
            "weighted_f1": f1_score(y_test, predicted, average="weighted"),
            "macro_f1": f1_score(y_test, predicted, average="macro"),
            "roc_auc_ovr_macro": roc_auc_score(
                y_test,
                probabilities,
                labels=selected_model.classes_,
                multi_class="ovr",
                average="macro",
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predicted, labels=classes
            ).tolist(),
            "classification_report": {
                label: report[label] for label in classes
            },
        },
        "split": {
            "train_records": int(len(train_idx)),
            "test_records": int(len(test_idx)),
            "stratified": True,
            "grouped_by_student": False,
        },
        "_model": selected_model,
        "_X_test": X_test,
        "_y_test": y_test,
        "_X_train": X_train,
        "_y_train": y_train,
    }


def permutation_importance(result, X, y, categorical):
    from sklearn.inspection import permutation_importance as measure_importance

    numeric = [column for column in X.columns if column not in categorical]
    model = Pipeline([
        ("preprocess", preprocessor(X, categorical, numeric, scale=False)),
        ("model", RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1
        )),
    ])
    model.fit(result["_X_train"], result["_y_train"])
    scores = measure_importance(
        model,
        result["_X_test"],
        result["_y_test"],
        scoring="f1_weighted",
        n_repeats=5,
        random_state=SEED,
        n_jobs=-1,
    )
    ranked = sorted(
        zip(X.columns, scores.importances_mean, scores.importances_std),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"feature": feature, "importance": float(value), "std": float(deviation)}
        for feature, value, deviation in ranked[:15]
    ]


def cluster(X, y, categorical, sample_size=2000):
    numeric = [column for column in X.columns if column not in categorical]
    encoded = preprocessor(X, categorical, numeric, scale=True).fit_transform(X)
    rng = np.random.default_rng(SEED)
    sample_idx = np.sort(rng.choice(
        len(X), min(sample_size, len(X)), replace=False
    ))
    candidates = []
    fitted = {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(encoded)
        score = silhouette_score(encoded[sample_idx], model.labels_[sample_idx])
        candidates.append({"k": k, "silhouette": float(score)})
        fitted[k] = model

    selected_k = max(candidates, key=lambda item: item["silhouette"])["k"]
    labels = y.astype(str).to_numpy()
    numeric_profiles = sorted(
        numeric,
        key=lambda column: float(pd.to_numeric(X[column], errors="coerce").var()),
        reverse=True,
    )[:4]
    categorical_profiles = [
        column for column in categorical if X[column].nunique(dropna=True) <= 12
    ][:4]
    clusters = []
    for cluster_id in range(selected_k):
        rows = np.flatnonzero(fitted[selected_k].labels_ == cluster_id)
        outcome_share = pd.Series(labels[rows]).value_counts(normalize=True)
        categorical_mode = {}
        for column in categorical_profiles:
            modes = X.iloc[rows][column].mode(dropna=True)
            if not modes.empty:
                categorical_mode[column] = str(modes.iloc[0])
        clusters.append({
            "cluster": cluster_id,
            "size": int(len(rows)),
            "outcome_share": {
                str(label): float(value) for label, value in outcome_share.items()
            },
            "profile": {
                "numeric_median": {
                    column: float(pd.to_numeric(X.iloc[rows][column]).median())
                    for column in numeric_profiles
                },
                "categorical_mode": categorical_mode,
            },
        })
    return {
        "selection_metric": "silhouette score",
        "candidates": candidates,
        "selected_k": selected_k,
        "silhouette": next(
            item["silhouette"] for item in candidates if item["k"] == selected_k
        ),
        "sample_size": int(len(sample_idx)),
        "clusters": clusters,
    }


def main():
    dataset = pd.read_csv(ROOT / "data.csv", sep=";")
    dataset.columns = [column.replace("\ufeff", "").strip() for column in dataset.columns]
    if "Target" not in dataset:
        raise ValueError("The UCI dataset is missing its Target column.")
    missing_columns = sorted(set(CATEGORICAL) - set(dataset.columns))
    if missing_columns:
        raise ValueError(f"Expected categorical features are missing: {missing_columns}")

    X = dataset.drop(columns="Target")
    y = dataset["Target"].astype(str)
    if sorted(y.unique()) != sorted(CLASSES):
        raise ValueError(f"Unexpected target labels: {sorted(y.unique())}")
    result = evaluate(X, y, CLASSES, CATEGORICAL)
    result["dataset"] = {
        "source": "UCI Machine Learning Repository",
        "title": "Predict Students' Dropout and Academic Success",
        "rows": int(len(dataset)),
        "predictor_columns": int(X.shape[1]),
        "target_column": "Target",
        "class_distribution": {
            label: int(y.value_counts().get(label, 0)) for label in CLASSES
        },
        "missing_cells": int(dataset.isna().sum().sum()),
        "exact_duplicate_rows": int(dataset.duplicated().sum()),
        "duplicate_feature_rows": int(X.duplicated().sum()),
        "features": {
            "categorical": CATEGORICAL,
            "numeric": [column for column in X.columns if column not in CATEGORICAL],
        },
    }
    result["feature_importance"] = permutation_importance(
        result, X, y, CATEGORICAL
    )
    result["clustering"] = cluster(X, y, CATEGORICAL)
    output_result = {
        key: value for key, value in result.items() if not key.startswith("_")
    }
    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "random_state": SEED,
        },
        "protocol": {
            "holdout_fraction": 0.20,
            "cross_validation": "5-fold stratified; training partition only",
            "selection_metric": "mean cross-validation weighted F1",
            "categorical_encoding": "One-hot encoding inside each model pipeline",
            "numeric_handling": (
                "Median imputation inside each pipeline; standardization for "
                "Logistic Regression only"
            ),
            "missingness": (
                "The included dataset has no missing cells; imputers are fitted "
                "within training folds and do not alter this dataset."
            ),
            "feature_importance": (
                "300-tree Random Forest; holdout permutation importance; "
                "five repeats; weighted F1 scoring"
            ),
            "clustering": (
                "K-Means on standardized/one-hot encoded predictors; target "
                "excluded; k=2..6 selected by sampled silhouette score"
            ),
        },
        "experiments": {"uci": output_result},
    }
    payload = clean(payload)

    classes = result["classes"]
    holdout = result["holdout"]
    report = [
        "STUDENT DROPOUT PREDICTION — PORTUGUESE HIGHER EDUCATION",
        f"Generated: {payload['generated_at']}",
        (
            f"Environment: Python {platform.python_version()}, pandas "
            f"{pd.__version__}, NumPy {np.__version__}, scikit-learn "
            f"{sklearn.__version__}; random state {SEED}"
        ),
        "",
        "DATA AUDIT",
        f"Rows: {len(dataset)}; predictors: {X.shape[1]}; target: Target",
        f"Missing cells: {dataset.isna().sum().sum()}",
        f"Exact duplicate rows: {dataset.duplicated().sum()}",
        "Target counts: " + ", ".join(
            f"{label} {int(y.value_counts().get(label, 0))}" for label in CLASSES
        ),
        "",
        "MODEL SELECTION — STRATIFIED 5-FOLD CV ON TRAINING PARTITION",
    ]
    for name, scores in result["cv"].items():
        report.append(
            f"{name}: accuracy {scores['accuracy'][0]:.4f} ± {scores['accuracy'][1]:.4f}; "
            f"weighted F1 {scores['weighted_f1'][0]:.4f} ± {scores['weighted_f1'][1]:.4f}; "
            f"macro F1 {scores['macro_f1'][0]:.4f} ± {scores['macro_f1'][1]:.4f}; "
            f"macro OvR ROC-AUC {scores['roc_auc'][0]:.4f} ± {scores['roc_auc'][1]:.4f}"
        )
    report.extend([
        "",
        f"Selected model: {result['selected_model']}",
        f"Split: {result['split']['train_records']} train / "
        f"{result['split']['test_records']} untouched holdout records",
        (
            f"Holdout: accuracy {holdout['accuracy']:.4f}; weighted precision "
            f"{holdout['precision_weighted']:.4f}; weighted recall "
            f"{holdout['recall_weighted']:.4f}; weighted F1 "
            f"{holdout['weighted_f1']:.4f}; macro F1 {holdout['macro_f1']:.4f}; "
            f"macro OvR ROC-AUC {holdout['roc_auc_ovr_macro']:.4f}"
        ),
        "Per-class (precision, recall, F1, support):",
    ])
    for label in classes:
        scores = holdout["classification_report"][label]
        report.append(
            f"  {label}: {scores['precision']:.4f}, {scores['recall']:.4f}, "
            f"{scores['f1-score']:.4f}, {int(scores['support'])}"
        )
    report.extend([
        f"Confusion matrix (class order {classes}): "
        f"{holdout['confusion_matrix']}",
        "",
        "Top holdout permutation features (predictive reliance, not causation):",
    ])
    report.extend(
        f"  {item['feature']}: {item['importance']:.6f}"
        for item in result["feature_importance"][:5]
    )
    report.extend([
        "",
        f"Exploratory K-Means: k={result['clustering']['selected_k']}; "
        f"silhouette={result['clustering']['silhouette']:.4f}; "
        f"sample={result['clustering']['sample_size']}",
        "Clustering excludes the target; cluster outcome shares are descriptive.",
    ])

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (ROOT / "results.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
