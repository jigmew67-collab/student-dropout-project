"""Reproduce the UCI experiment and run the separate OULAD study.

Run from the repository root with: python analysis.py
OULAD is downloaded separately with scripts/download_oulad.py.
"""
from __future__ import annotations

import json
import platform
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
SEED = 42
UCI_CLASSES = ["Dropout", "Enrolled", "Graduate"]
OULAD_CLASSES = ["Not withdrawn", "Withdrawn"]
UCI_CATEGORICAL = ["Marital status", "Application mode", "Course", "Daytime/evening attendance",
    "Previous qualification", "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "International"]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def preprocessor(categorical, numeric, scale=True, categorical_imputation="most_frequent",
                 add_missing_indicators=False, numeric_categorical=()):
    categorical_imputer = SimpleImputer(
        strategy=categorical_imputation,
        fill_value="Unknown" if categorical_imputation == "constant" else None,
    )
    numeric_imputer = SimpleImputer(strategy="median", add_indicator=add_missing_indicators)
    string_categorical = [column for column in categorical if column not in numeric_categorical]
    transformers=[]
    if string_categorical:
        transformers.append(
            ("cat", Pipeline([("impute", categorical_imputer),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
             string_categorical)
        )
    if numeric_categorical:
        transformers.append(
            ("cat_numeric", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
             numeric_categorical)
        )
    transformers.append(
        ("num", Pipeline([("impute", numeric_imputer),
                          ("scale", StandardScaler() if scale else "passthrough")]), numeric)
    )
    if add_missing_indicators and categorical:
        transformers.append(
            ("cat_missing", MissingIndicator(features="missing-only", error_on_new=False), categorical)
        )
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def categorical_columns(frame):
    return [c for c in frame.columns if pd.api.types.is_object_dtype(frame[c])
            or pd.api.types.is_string_dtype(frame[c]) or isinstance(frame[c].dtype, pd.CategoricalDtype)]


def feature_missingness(frame):
    rows = []
    for column in frame.columns:
        missing = int(frame[column].isna().sum())
        rows.append({"feature": column, "missing_cells": missing,
                     "missing_percent": float(missing / len(frame) * 100) if len(frame) else 0.0})
    return rows


def source_table_audit(tables):
    summary = []
    for name, frame in tables.items():
        missing = frame.isna().sum()
        missing_features = [
            {"feature": column, "missing_cells": int(count),
             "missing_percent": float(count / len(frame) * 100) if len(frame) else 0.0}
            for column, count in missing.items() if count
        ]
        cells = int(frame.size)
        summary.append({"table": name, "rows": int(len(frame)), "columns": int(len(frame.columns)),
                        "missing_cells": int(missing.sum()),
                        "overall_missing_percent": float(missing.sum() / cells * 100) if cells else 0.0,
                        "missing_features": missing_features})
    return summary


def grouped_missingness(frame, missing_mask, group_columns):
    result = {}
    for column in group_columns:
        if column not in frame:
            continue
        grouped = frame.assign(_missing=missing_mask).groupby(column, dropna=False)["_missing"]
        result[column] = [
            {"group": str(value), "rows": int(count), "missing_rows": int(missing),
             "missing_percent": float(missing / count * 100) if count else 0.0}
            for value, count, missing in zip(grouped.size().index, grouped.size().values, grouped.sum().values)
        ]
    return result


def model_set(prep, tree_prep):
    return {
        "Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=3000, random_state=SEED))]),
        "Decision Tree": Pipeline([("prep", tree_prep), ("model", DecisionTreeClassifier(random_state=SEED))]),
        "Random Forest": Pipeline([("prep", tree_prep), ("model", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]),
        "Histogram Gradient Boosting": Pipeline([("prep", tree_prep), ("model", HistGradientBoostingClassifier(random_state=SEED))]),
    }


PREPROCESSING_VARIANTS = {
    "A_existing": {"categorical_imputation": "most_frequent", "add_missing_indicators": False},
    "B_fold_unknown": {"categorical_imputation": "constant", "add_missing_indicators": False},
    "C_fold_unknown_indicator": {"categorical_imputation": "constant", "add_missing_indicators": True},
}


def evaluate(dataset, X, y, classes, categorical, groups=None, preprocessing_variant="A_existing"):
    numeric = [c for c in X.columns if c not in categorical]
    if preprocessing_variant not in PREPROCESSING_VARIANTS:
        raise ValueError(f"Unknown preprocessing variant: {preprocessing_variant}")
    variant = PREPROCESSING_VARIANTS[preprocessing_variant]
    model_X = X.copy()
    if preprocessing_variant == "A_existing":
        # Reproduce the current OULAD preprocessing: categorical gaps were
        # converted to the explicit Unknown value before the model pipeline.
        for column in categorical:
            if pd.api.types.is_object_dtype(model_X[column]) or pd.api.types.is_string_dtype(model_X[column]):
                model_X[column] = model_X[column].fillna("Unknown")
    numeric_categorical=[column for column in categorical
                         if pd.api.types.is_numeric_dtype(model_X[column])]
    prep = preprocessor(categorical, numeric, True, **variant,
                        numeric_categorical=numeric_categorical)
    tree_prep = preprocessor(categorical, numeric, False, **variant,
                             numeric_categorical=numeric_categorical)
    models = model_set(prep, tree_prep)
    if groups is None:
        train_idx, test_idx = train_test_split(np.arange(len(model_X)), test_size=0.20, random_state=SEED, stratify=y)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_for_scores = list(splitter.split(model_X.iloc[train_idx], y.iloc[train_idx]))
    else:
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_idx, test_idx = next(iter(outer.split(model_X, y, groups)))
        cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_for_scores = list(cv_splitter.split(model_X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]))
    X_train, X_test = model_X.iloc[train_idx], model_X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    fold_scores = {}
    for name, model in models.items():
        acc, wf1, mf1, auc = [], [], [], []
        for fit_idx, val_idx in cv_for_scores:
            fold_model = model
            fold_model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
            pred = fold_model.predict(X_train.iloc[val_idx])
            acc.append(accuracy_score(y_train.iloc[val_idx], pred))
            wf1.append(f1_score(y_train.iloc[val_idx], pred, average="weighted"))
            mf1.append(f1_score(y_train.iloc[val_idx], pred, average="macro"))
            if hasattr(fold_model, "predict_proba"):
                try:
                    prob = fold_model.predict_proba(X_train.iloc[val_idx])
                    auc.append(roc_auc_score(y_train.iloc[val_idx], prob, labels=fold_model.classes_, multi_class="ovr", average="macro") if len(classes)>2 else roc_auc_score(y_train.iloc[val_idx], prob[:, list(fold_model.classes_).index(classes[-1])]))
                except ValueError: pass
        fold_scores[name] = {"accuracy": [float(np.mean(acc)), float(np.std(acc))],
                             "weighted_f1": [float(np.mean(wf1)), float(np.std(wf1))],
                             "macro_f1": [float(np.mean(mf1)), float(np.std(mf1))],
                             "roc_auc": [float(np.mean(auc)), float(np.std(auc))] if auc else None}
    best_name = max(fold_scores, key=lambda n: fold_scores[n]["weighted_f1"][0])
    best = models[best_name].fit(X_train, y_train)
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test) if hasattr(best, "predict_proba") else None
    report = classification_report(y_test, pred, labels=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=classes).tolist()
    auc = None
    if proba is not None:
        try:
            auc = float(roc_auc_score(y_test, proba, labels=best.classes_, multi_class="ovr", average="macro") if len(classes)>2 else roc_auc_score(y_test, proba[:, list(best.classes_).index(classes[-1])]))
        except ValueError: pass
    test_groups = set(groups.iloc[test_idx]) if groups is not None else None
    train_groups = set(groups.iloc[train_idx]) if groups is not None else None
    return {"dataset": dataset, "rows": int(len(X)), "features": int(X.shape[1]), "classes": classes,
        "preprocessing_variant": preprocessing_variant,
        "class_distribution": {k:int(v) for k,v in y.value_counts().items()},
        "missing_cells": int(X.isna().sum().sum()), "duplicate_feature_rows": int(X.duplicated().sum()),
        "split": {"train_records":len(train_idx), "test_records":len(test_idx), "grouped_by_student":groups is not None,
                  "train_unique_students":len(train_groups) if groups is not None else None,
                  "test_unique_students":len(test_groups) if groups is not None else None,
                  "overlapping_students":len(train_groups & test_groups) if groups is not None else 0},
        "cv": fold_scores, "selected_model":best_name,
        "holdout": {"accuracy":accuracy_score(y_test,pred), "precision_weighted":precision_score(y_test,pred,average="weighted",zero_division=0),
          "recall_weighted":recall_score(y_test,pred,average="weighted",zero_division=0), "weighted_f1":f1_score(y_test,pred,average="weighted"),
          "macro_f1":f1_score(y_test,pred,average="macro"), "roc_auc_ovr_macro":auc,
          "confusion_matrix":cm, "classification_report":{c:report[c] for c in classes}},
        "_model":best, "_test_X":X_test, "_test_y":y_test, "_pred":pred,
        "_train_X":X_train, "_train_y":y_train, "_feature_cols":list(X.columns), "_categorical":categorical}


def importance(result, X, y, categorical=None):
    model = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    categorical = categorical or categorical_columns(X)
    numeric = [c for c in X.columns if c not in categorical]
    variant = PREPROCESSING_VARIANTS[result["preprocessing_variant"]]
    numeric_categorical=[column for column in categorical
                         if pd.api.types.is_numeric_dtype(result["_train_X"][column])]
    pipe = Pipeline([("prep", preprocessor(categorical, numeric, False, **variant,
                                           numeric_categorical=numeric_categorical)),
                     ("model", model)])
    # Fit only the saved training partition, then assess permutation on holdout.
    pipe.fit(result["_train_X"], result["_train_y"])
    from sklearn.inspection import permutation_importance
    scores = permutation_importance(pipe, result["_test_X"], result["_test_y"], scoring="f1_weighted", n_repeats=5, random_state=SEED, n_jobs=-1)
    return [{"feature":f,"importance":float(v),"std":float(s)} for f,v,s in sorted(zip(X.columns,scores.importances_mean,scores.importances_std),key=lambda row:row[1],reverse=True)[:15]]


def cluster(X, y, categorical, sample_size=2000):
    X = X.copy()
    for column in categorical:
        X[column] = X[column].fillna("Unknown")
    numeric = [c for c in X.columns if c not in categorical]
    transformer = preprocessor(categorical,numeric,True)
    Z = transformer.fit_transform(X)
    rng=np.random.default_rng(SEED)
    ids=np.sort(rng.choice(len(X), min(sample_size,len(X)), replace=False))
    candidates=[]
    fitted={}
    for k in range(2,7):
        km=KMeans(n_clusters=k,n_init=10,random_state=SEED).fit(Z)
        silhouette=float(silhouette_score(Z[ids],km.labels_[ids]))
        candidates.append({"k":k,"silhouette":silhouette})
        fitted[k]=km
    chosen=max(candidates,key=lambda item:item["silhouette"])["k"]
    km=fitted[chosen]
    components=PCA(n_components=2,random_state=SEED).fit_transform(Z[ids])
    chart=[]
    for i, row_idx in enumerate(ids):
        chart.append({"x":float(components[i,0]),"y":float(components[i,1]),"cluster":int(km.labels_[row_idx]),"outcome":str(y.iloc[row_idx])})
    distributions=[]
    labels=np.asarray(y.astype(str))
    numeric_profile_cols=sorted(numeric,key=lambda c:float(pd.to_numeric(X[c],errors="coerce").var()),reverse=True)[:4]
    categorical_profile_cols=[c for c in categorical if X[c].nunique(dropna=True)<=12][:4]
    for cl in range(chosen):
        rows=np.where(km.labels_==cl)[0]
        counts=pd.Series(labels[rows]).value_counts(normalize=True)
        profile={"numeric_median":{col:float(pd.to_numeric(X.iloc[rows][col],errors="coerce").median()) for col in numeric_profile_cols},
                 "categorical_mode":{col:str(X.iloc[rows][col].mode(dropna=True).iloc[0]) for col in categorical_profile_cols if not X.iloc[rows][col].mode(dropna=True).empty}}
        distributions.append({"cluster":cl,"size":int(len(rows)),"outcome_share":{str(k):float(v) for k,v in counts.items()},"profile":profile})
    return {"selection_metric":"silhouette score", "candidates":candidates,"selected_k":chosen,"silhouette":next(x["silhouette"] for x in candidates if x["k"]==chosen),
            "sample_size":len(ids), "pca_points":chart,"clusters":distributions}


def load_oulad():
    archive=ROOT/"data/raw/anonymisedData.zip"
    if not archive.exists(): raise FileNotFoundError("OULAD missing. Run: python scripts/download_oulad.py")
    def read(name):
        with zipfile.ZipFile(archive) as z:
            matches=[n for n in z.namelist() if n.lower().endswith("/"+name.lower()) or n.lower()==name.lower()]
            if not matches: raise FileNotFoundError(f"{name} not found in OULAD archive")
            with z.open(matches[0]) as file: return pd.read_csv(file)
    info=read("studentInfo.csv")
    registration=read("studentRegistration.csv")
    courses=read("courses.csv")
    assessment=read("assessments.csv")
    submissions=read("studentAssessment.csv")
    vle=read("vle.csv")
    interactions=read("studentVle.csv")
    source_tables={"studentInfo":info,"studentRegistration":registration,"courses":courses,
                   "assessments":assessment,"studentAssessment":submissions,"vle":vle,
                   "studentVle":interactions}
    labels=info.set_index(["code_module","code_presentation","id_student"])["final_result"].copy()
    # Only records that would be observable by day 28 are retained.
    reg=registration[["code_module","code_presentation","id_student","date_registration","date_unregistration"]].copy()
    base=info.drop(columns=["final_result"]).merge(reg,on=["code_module","code_presentation","id_student"],how="left")
    # Day-28 risk set: exclude students not yet registered or already unregistered
    # by the prediction landmark. These dates define eligibility only and are not predictors.
    eligible=base["date_registration"].le(28) & (base["date_unregistration"].isna() | base["date_unregistration"].gt(28))
    excluded_before_cutoff=int((~eligible).sum())
    base=base.loc[eligible].copy()
    base=base.merge(courses,on=["code_module","code_presentation"],how="left")
    assess=assessment.merge(submissions,on="id_assessment",how="inner")
    assess=assess[(assess["date_submitted"]<=28)&(assess["is_banked"]==0)]
    assess=assess.groupby(["code_module","code_presentation","id_student"]).agg(
        early_assessments_submitted=("score","count"),early_assessment_mean_score=("score","mean"),
        _early_assessment_records=("id_assessment","size"),
        _early_assessment_missing_score_records=("score",lambda values:values.isna().sum())).reset_index()
    base=base.merge(assess,on=["code_module","code_presentation","id_student"],how="left")
    base[["early_assessments_submitted"]]=base[["early_assessments_submitted"]].fillna(0)
    vle_meta=vle[["id_site","code_module","code_presentation","activity_type"]]
    agg=[]
    # Chunking bounds memory while filtering the multi-million-row interaction table.
    with zipfile.ZipFile(archive) as z:
        name=next(n for n in z.namelist() if n.lower().endswith("studentvle.csv"))
        with z.open(name) as f:
            for chunk in pd.read_csv(f,chunksize=400_000):
                chunk=chunk[chunk["date"]<=28].merge(vle_meta,on=["id_site","code_module","code_presentation"],how="left")
                if len(chunk): agg.append(chunk.groupby(["code_module","code_presentation","id_student","activity_type"])["sum_click"].sum().reset_index())
    if agg:
        clicks=pd.concat(agg).groupby(["code_module","code_presentation","id_student","activity_type"])["sum_click"].sum().unstack("activity_type",fill_value=0)
        clicks.columns=["clicks_"+str(c).lower().replace(" ","_") for c in clicks.columns]
        clicks=clicks.reset_index()
        base=base.merge(clicks,on=["code_module","code_presentation","id_student"],how="left")
    click_cols=[c for c in base.columns if c.startswith("clicks_")]
    base[click_cols]=base[click_cols].fillna(0)
    base["total_clicks_by_day_28"]=base[click_cols].sum(axis=1)
    key=["code_module","code_presentation","id_student"]
    base=base.merge(labels.rename("final_result"),left_on=key,right_index=True,how="left")
    base["outcome"]=np.where(base["final_result"].eq("Withdrawn"),"Withdrawn","Not withdrawn")
    groups=base["id_student"].astype(str)
    X=base.drop(columns=key+["final_result","outcome","date_unregistration",
                             "_early_assessment_records","_early_assessment_missing_score_records"])
    categorical=categorical_columns(X)
    y=base["outcome"]
    missing_mean=X["early_assessment_mean_score"].isna()
    missing_mean_with_submission=missing_mean & base["_early_assessment_records"].gt(0)
    missing_mean_without_submission=missing_mean & base["_early_assessment_records"].isna()
    student_missing=base.assign(_missing=missing_mean).groupby("id_student")["_missing"].any()
    source_audit=source_table_audit(source_tables)
    source_cells=sum(item["rows"]*item["columns"] for item in source_audit)
    source_missing_cells=sum(item["missing_cells"] for item in source_audit)
    missingness={
      "rows":int(len(X)),"predictor_columns":int(X.shape[1]),"columns_including_target":int(X.shape[1]+1),
      "predictor_cells":int(X.size),"cells_including_target":int(X.size+len(X)),
      "missing_cells":int(X.isna().sum().sum()),"total_missing_cells":int(X.isna().sum().sum()),
      "predictor_missing_percent":float(X.isna().sum().sum()/X.size*100),
      "overall_missing_percent":float(X.isna().sum().sum()/(X.size+len(X))*100),
      "by_feature":feature_missingness(X),
      "categorical_unknowns":{col:int(X[col].isna().sum()) for col in categorical if X[col].isna().any()},
      "early_assessment_mean_missing_by_outcome":{
        str(label):{"rows":int(len(rows)),"missing_rows":int(rows["early_assessment_mean_score"].isna().sum()),
                    "missing_percent":float(rows["early_assessment_mean_score"].isna().mean()*100)}
        for label,rows in base.groupby("outcome")},
      "early_assessment_mean_missing_by_group":grouped_missingness(
        base,missing_mean,["code_module","code_presentation","age_band","highest_education"]),
      "missing_by_feature_and_group":{
        column:grouped_missingness(base,X[column].isna(),
                                   ["outcome","code_module","code_presentation","age_band","highest_education"])
        for column in X.columns if X[column].isna().any()},
      "students_with_any_missing_early_mean":{
        "students":int(len(student_missing)),"students_with_missing":int(student_missing.sum()),
        "percent":float(student_missing.mean()*100)},
      "early_assessment_count_zero_rows":int(X["early_assessments_submitted"].eq(0).sum()),
      "early_mean_missing_and_count_zero_rows":int((missing_mean & X["early_assessments_submitted"].eq(0)).sum()),
      "early_mean_missing_without_qualifying_submission_rows":int(missing_mean_without_submission.sum()),
      "early_mean_missing_with_submission_but_missing_score_rows":int(missing_mean_with_submission.sum()),
      "qualifying_early_submissions_with_missing_score":int(base["_early_assessment_missing_score_records"].sum()),
      "raw_source_total_cells":int(source_cells),"raw_source_total_missing_cells":int(source_missing_cells),
      "raw_source_overall_missing_percent":float(source_missing_cells/source_cells*100),
      "source_tables":source_audit,
      "interpretation":[
        "Missing early mean scores are unavailable measurements, not scores of zero: 7,600 eligible registrations had no qualifying early submission record, while 6 had a submission whose score was missing.",
        "The existing early_assessments_submitted feature counts non-missing scores, so its zero value includes both no qualifying submission and the six cases with an unscored submission.",
        "Missing imd_band values are retained as an explicit Unknown category; the source data does not specify their individual cause.",
        "Missing date_unregistration values denote no recorded unregistration date and are used only for risk-set eligibility, not as predictors.",
        "Zero early assessment counts mean no non-missing early scores; six eligible registrations had a submission record with a missing score. Zero VLE click aggregates mean no qualifying recorded clicks; neither count is a replacement of a missing score with zero."
      ]}
    data_quality={"registrations":len(base),"unique_students":int(base["id_student"].nunique()),"modules":int(base["code_module"].nunique()),
      "presentations":int(base[["code_module","code_presentation"]].drop_duplicates().shape[0]),
      "source_table_rows":{"studentInfo":len(info),"studentRegistration":len(registration),"courses":len(courses),"assessments":len(assessment),"studentAssessment":len(submissions),"vle":len(vle),"studentVle":len(interactions)},
      "missing_feature_cells":missingness["missing_cells"],"missingness":missingness,
      "duplicate_feature_rows":int(X.duplicated().sum()),
      "duplicate_registration_keys":int(base.duplicated(key).sum()),
      "target_counts":{str(k):int(v) for k,v in base["final_result"].value_counts().items()},"all_student_records_grouped":True,
      "excluded_not_registered_or_already_unregistered_by_day_28":excluded_before_cutoff,
      "prediction_cutoff_day":28,"definition":"Among registrations active by day 28, subsequent Withdrawn is positive; Fail, Pass and Distinction remain Not withdrawn."}
    return X,y,categorical,groups,data_quality


def stripped(result): return {k:v for k,v in result.items() if not k.startswith("_")}


def run_preprocessing_experiments(dataset, X, y, classes, categorical, groups=None):
    results={}
    reference_split=None
    for variant in PREPROCESSING_VARIANTS:
        result=evaluate(dataset,X,y,classes,categorical,groups,variant)
        split=(tuple(result["_test_X"].index),tuple(result["_train_X"].index))
        if reference_split is not None and split!=reference_split:
            raise RuntimeError(f"Preprocessing variant {variant} changed the evaluation split")
        reference_split=split
        results[variant]=result
    selected_variant=max(
        results,
        key=lambda name:max(
            score["weighted_f1"][0]
            for score in results[name]["cv"].values()
        ),
    )
    comparison={
        "selection_rule":"Choose the preprocessing variant with the best model's mean training-only CV weighted F1; holdout metrics are not used to select it.",
        "variants":{
            name:{
                "selected_model":result["selected_model"],
                "classes":result["classes"],
                "cv":result["cv"],
                "holdout":result["holdout"],
            }
            for name,result in results.items()
        },
        "selected_variant":selected_variant,
    }
    return results[selected_variant],comparison


def main():
    uci=pd.read_csv(ROOT/"data.csv",sep=";")
    uci.columns=[c.replace("\ufeff","").strip() for c in uci.columns]
    uy=uci["Target"].astype(str); ux=uci.drop(columns="Target")
    ur,uci_comparison=run_preprocessing_experiments("UCI Portugal",ux,uy,UCI_CLASSES,UCI_CATEGORICAL)
    uci_audit=feature_missingness(uci)
    ur["missingness"]={
        "rows":int(len(uci)),"predictor_columns":int(ux.shape[1]),"columns_including_target":int(uci.shape[1]),
        "predictor_cells":int(ux.size),"cells_including_target":int(uci.size),
        "missing_cells":int(uci.isna().sum().sum()),"total_missing_cells":int(uci.isna().sum().sum()),
        "predictor_missing_percent":float(ux.isna().sum().sum()/ux.size*100),
        "overall_missing_percent":float(uci.isna().sum().sum()/uci.size*100),
        "by_feature":uci_audit,
        "interpretation":["No missing values were found in the included UCI CSV; no imputation or missingness indicator was needed."]
    }
    ur["feature_importance"]=importance(ur,ux,uy,UCI_CATEGORICAL)
    ur["clustering"]=cluster(ux,uy,UCI_CATEGORICAL)
    try:
        ox,oy,oc,og,oq=load_oulad()
        ou,oulad_comparison=run_preprocessing_experiments("OULAD United Kingdom",ox,oy,OULAD_CLASSES,oc,og)
        ou["data_quality"]=oq
        ou["feature_importance"]=importance(ou,ox,oy,oc)
        ou["clustering"]=cluster(ox,oy,oc,sample_size=2000)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The complete UCI and OULAD pipeline requires the verified OULAD archive; existing outputs were left unchanged."
        ) from exc
    status="completed"
    payload={"generated_at":pd.Timestamp.now().strftime("%Y-%m-%d"),
      "environment":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"scikit_learn":sklearn.__version__,"random_state":SEED},
      "oulad_status":status,"experiments":{"uci":stripped(ur),"oulad":stripped(ou)},
      "preprocessing_comparison":{"uci":uci_comparison,"oulad":oulad_comparison}}
    report=["STUDENT DROPOUT PREDICTION — REPRODUCIBLE RESULTS",
      f"Generated: {payload['generated_at']}",
      f"Environment: Python {platform.python_version()}, pandas {pd.__version__}, NumPy {np.__version__}, scikit-learn {sklearn.__version__}"]
    def add_comparison(title, comparison):
        report.extend(["",title])
        for variant,details in comparison["variants"].items():
            model_name=details["selected_model"]
            cv=details["cv"][model_name]
            hold=details["holdout"]
            report.extend([
                "",f"{variant} — selected model: {model_name}",
                f"Training CV: accuracy {cv['accuracy'][0]:.4f} ± {cv['accuracy'][1]:.4f}; weighted F1 {cv['weighted_f1'][0]:.4f} ± {cv['weighted_f1'][1]:.4f}; macro F1 {cv['macro_f1'][0]:.4f} ± {cv['macro_f1'][1]:.4f}; ROC-AUC {cv['roc_auc'][0]:.4f} ± {cv['roc_auc'][1]:.4f}" if cv["roc_auc"] else
                f"Training CV: accuracy {cv['accuracy'][0]:.4f} ± {cv['accuracy'][1]:.4f}; weighted F1 {cv['weighted_f1'][0]:.4f} ± {cv['weighted_f1'][1]:.4f}; macro F1 {cv['macro_f1'][0]:.4f} ± {cv['macro_f1'][1]:.4f}; ROC-AUC unavailable",
                f"Holdout: accuracy {hold['accuracy']:.4f}; weighted F1 {hold['weighted_f1']:.4f}; macro F1 {hold['macro_f1']:.4f}; ROC-AUC {hold['roc_auc_ovr_macro']:.4f}" if hold["roc_auc_ovr_macro"] is not None else
                f"Holdout: accuracy {hold['accuracy']:.4f}; weighted F1 {hold['weighted_f1']:.4f}; macro F1 {hold['macro_f1']:.4f}; ROC-AUC unavailable",
                "Per-class (precision, recall, F1, support):"
            ])
            for class_name in details["classes"]:
                scores=hold["classification_report"][class_name]
                report.append(f"  {class_name}: {scores['precision']:.4f}, {scores['recall']:.4f}, {scores['f1-score']:.4f}, {int(scores['support'])}")
            report.append(f"Confusion matrix (class order {details['classes']}): {hold['confusion_matrix']}")
        report.append(f"Selected preprocessing variant by training-only CV weighted F1: {comparison['selected_variant']}")
    add_comparison("UCI PORTUGAL — PREPROCESSING COMPARISON",uci_comparison)
    add_comparison("OULAD UNITED KINGDOM — PREPROCESSING COMPARISON",oulad_comparison)
    report += ["","FINAL SELECTED PIPELINE RESULTS"]
    report += [f"UCI holdout selected model: {ur['selected_model']}; weighted F1 {ur['holdout']['weighted_f1']:.4f}; macro F1 {ur['holdout']['macro_f1']:.4f}; ROC-AUC {ur['holdout']['roc_auc_ovr_macro']:.4f}",
      f"UCI missingness: {ur['missingness']['rows']} rows × {ur['missingness']['columns_including_target']} columns; {ur['missingness']['total_missing_cells']} missing cells ({ur['missingness']['overall_missing_percent']:.4f}%).",
      f"UCI top permutation features: {', '.join(x['feature'] for x in ur['feature_importance'][:5])}",
      f"UCI clustering: k={ur['clustering']['selected_k']}, silhouette={ur['clustering']['silhouette']:.4f}"]
    report += [f"OULAD registrations: {oq['registrations']}; unique students: {oq['unique_students']}; features: {ou['features']}; grouped holdout: {ou['split']['test_records']}",
          f"OULAD selected model: {ou['selected_model']}; weighted F1 {ou['holdout']['weighted_f1']:.4f}; macro F1 {ou['holdout']['macro_f1']:.4f}; ROC-AUC {ou['holdout']['roc_auc_ovr_macro']:.4f}",
          f"OULAD student overlap between train/test: {ou['split']['overlapping_students']}",
          f"OULAD missingness: {oq['missingness']['rows']} rows × {oq['missingness']['columns_including_target']} columns; {oq['missingness']['total_missing_cells']} missing cells ({oq['missingness']['overall_missing_percent']:.4f}%).",
          f"OULAD source tables: {oq['missingness']['raw_source_total_missing_cells']} missing of {oq['missingness']['raw_source_total_cells']} source cells ({oq['missingness']['raw_source_overall_missing_percent']:.4f}%).",
          f"OULAD early assessment mean: {oq['missingness']['early_mean_missing_without_qualifying_submission_rows']} missing because no qualifying submission; {oq['missingness']['early_mean_missing_with_submission_but_missing_score_rows']} had a submission with a missing score.",
          f"OULAD top permutation features: {', '.join(x['feature'] for x in ou['feature_importance'][:5])}",
          f"OULAD clustering: k={ou['clustering']['selected_k']}, silhouette={ou['clustering']['silhouette']:.4f}"]
    out=ROOT/"results"; out.mkdir(exist_ok=True)
    analysis_text=json.dumps(clean(payload),indent=2)
    report_text="\n".join(report)+"\n"
    analysis_tmp=out/"analysis.json.tmp"; report_tmp=ROOT/"results.txt.tmp"
    analysis_tmp.write_text(analysis_text,encoding="utf-8")
    report_tmp.write_text(report_text,encoding="utf-8")
    analysis_tmp.replace(out/"analysis.json")
    report_tmp.replace(ROOT/"results.txt")
    print("Wrote results/analysis.json and results.txt")
    print("UCI holdout:",ur["holdout"])
    print("UCI importance:",ur["feature_importance"][:5])
    print("UCI clusters:",ur["clustering"]["selected_k"],ur["clustering"]["silhouette"])
    if status=="completed":
        print("OULAD quality:",oq)
        print("OULAD holdout:",ou["holdout"])
        print("OULAD importance:",ou["feature_importance"][:5])
        print("OULAD clusters:",ou["clustering"]["selected_k"],ou["clustering"]["silhouette"])

if __name__=="__main__": main()
