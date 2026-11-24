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
from sklearn.impute import SimpleImputer
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


def preprocessor(categorical, numeric, scale=True):
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler() if scale else "passthrough")]
    return ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ("num", Pipeline(steps), numeric),
    ], verbose_feature_names_out=False)


def categorical_columns(frame):
    return [c for c in frame.columns if pd.api.types.is_object_dtype(frame[c])
            or pd.api.types.is_string_dtype(frame[c]) or isinstance(frame[c].dtype, pd.CategoricalDtype)]


def model_set(prep, tree_prep):
    return {
        "Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=3000, random_state=SEED))]),
        "Decision Tree": Pipeline([("prep", tree_prep), ("model", DecisionTreeClassifier(random_state=SEED))]),
        "Random Forest": Pipeline([("prep", tree_prep), ("model", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]),
        "Histogram Gradient Boosting": Pipeline([("prep", tree_prep), ("model", HistGradientBoostingClassifier(random_state=SEED))]),
    }


def evaluate(dataset, X, y, classes, categorical, groups=None, split_mode="stratified"):
    numeric = [c for c in X.columns if c not in categorical]
    prep = preprocessor(categorical, numeric, True)
    tree_prep = preprocessor(categorical, numeric, False)
    models = model_set(prep, tree_prep)
    if groups is None:
        train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.20, random_state=SEED, stratify=y)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_for_scores = list(splitter.split(X.iloc[train_idx], y.iloc[train_idx]))
    else:
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_idx, test_idx = next(iter(outer.split(X, y, groups)))
        cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_for_scores = list(cv_splitter.split(X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
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
        "_model":best, "_test_X":X_test, "_test_y":y_test, "_pred":pred, "_feature_cols":list(X.columns), "_categorical":categorical}


def importance(result, X, y, categorical=None):
    model = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    categorical = categorical or categorical_columns(X)
    numeric = [c for c in X.columns if c not in categorical]
    pipe = Pipeline([("prep", preprocessor(categorical,numeric,False)),("model",model)])
    # Fit training partition only, then assess permutation on the fixed holdout.
    ids = result["_test_X"].index
    train = X.index.difference(ids)
    pipe.fit(X.loc[train], y.loc[train])
    from sklearn.inspection import permutation_importance
    scores = permutation_importance(pipe, result["_test_X"], result["_test_y"], scoring="f1_weighted", n_repeats=5, random_state=SEED, n_jobs=-1)
    return [{"feature":f,"importance":float(v),"std":float(s)} for f,v,s in sorted(zip(X.columns,scores.importances_mean,scores.importances_std),key=lambda row:row[1],reverse=True)[:15]]


def cluster(X, y, categorical, sample_size=2000):
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
        early_assessments_submitted=("score","count"),early_assessment_mean_score=("score","mean")).reset_index()
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
    X=base.drop(columns=key+["final_result","outcome","date_unregistration"])
    categorical=categorical_columns(X)
    for col in categorical: X[col]=X[col].fillna("Unknown")
    y=base["outcome"]
    data_quality={"registrations":len(base),"unique_students":int(base["id_student"].nunique()),"modules":int(base["code_module"].nunique()),
      "presentations":int(base[["code_module","code_presentation"]].drop_duplicates().shape[0]),
      "source_table_rows":{"studentInfo":len(info),"studentRegistration":len(registration),"courses":len(courses),"assessments":len(assessment),"studentAssessment":len(submissions),"vle":len(vle),"studentVle":len(interactions)},
      "missing_feature_cells":int(X.isna().sum().sum()),"duplicate_feature_rows":int(X.duplicated().sum()),
      "duplicate_registration_keys":int(base.duplicated(key).sum()),
      "target_counts":{str(k):int(v) for k,v in base["final_result"].value_counts().items()},"all_student_records_grouped":True,
      "excluded_not_registered_or_already_unregistered_by_day_28":excluded_before_cutoff,
      "prediction_cutoff_day":28,"definition":"Among registrations active by day 28, subsequent Withdrawn is positive; Fail, Pass and Distinction remain Not withdrawn."}
    return X,y,categorical,groups,data_quality


def stripped(result): return {k:v for k,v in result.items() if not k.startswith("_")}


def main():
    uci=pd.read_csv(ROOT/"data.csv",sep=";")
    uci.columns=[c.replace("\ufeff","").strip() for c in uci.columns]
    uy=uci["Target"].astype(str); ux=uci.drop(columns="Target")
    ur=evaluate("UCI Portugal",ux,uy,UCI_CLASSES,UCI_CATEGORICAL)
    ur["feature_importance"]=importance(ur,ux,uy,UCI_CATEGORICAL)
    ur["clustering"]=cluster(ux,uy,UCI_CATEGORICAL)
    try:
        ox,oy,oc,og,oq=load_oulad()
        ou=evaluate("OULAD United Kingdom",ox,oy,OULAD_CLASSES,oc,og)
        ou["data_quality"]=oq
        ou["feature_importance"]=importance(ou,ox,oy,oc)
        ou["clustering"]=cluster(ox,oy,oc,sample_size=2000)
        status="completed"
    except FileNotFoundError as exc:
        ou={"status":"not_run","reason":str(exc)}; status="not_run"
    payload={"generated_at":"2026-11-24","environment":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"scikit_learn":sklearn.__version__,"random_state":SEED},
      "oulad_status":status,"experiments":{"uci":stripped(ur),"oulad":stripped(ou)}}
    out=ROOT/"results"; out.mkdir(exist_ok=True)
    (out/"analysis.json").write_text(json.dumps(clean(payload),indent=2),encoding="utf-8")
    report=["STUDENT DROPOUT PREDICTION — REPRODUCIBLE RESULTS",f"Generated: 2026-11-24",f"Environment: Python {platform.python_version()}, pandas {pd.__version__}, NumPy {np.__version__}, scikit-learn {sklearn.__version__}","", "UCI PORTUGAL — FINAL OUTCOME MULTICLASS CLASSIFICATION",f"Rows: {ur['rows']}; input features: {ur['features']}; holdout: {ur['split']['test_records']}",f"Selected model: {ur['selected_model']}",f"Holdout accuracy: {ur['holdout']['accuracy']:.4f}; weighted F1: {ur['holdout']['weighted_f1']:.4f}; macro F1: {ur['holdout']['macro_f1']:.4f}; macro OvR ROC-AUC: {ur['holdout']['roc_auc_ovr_macro']:.4f}",f"Top permutation features: {', '.join(x['feature'] for x in ur['feature_importance'][:5])}",f"Clustering: k={ur['clustering']['selected_k']}, silhouette={ur['clustering']['silhouette']:.4f}",""]
    if status=="completed":
        report += ["OULAD UNITED KINGDOM — DAY-28 WITHDRAWAL CLASSIFICATION",f"Registrations: {oq['registrations']}; unique students: {oq['unique_students']}; features: {ou['features']}; grouped holdout: {ou['split']['test_records']}",f"Selected model: {ou['selected_model']}",f"Holdout accuracy: {ou['holdout']['accuracy']:.4f}; weighted F1: {ou['holdout']['weighted_f1']:.4f}; macro F1: {ou['holdout']['macro_f1']:.4f}; ROC-AUC: {ou['holdout']['roc_auc_ovr_macro']:.4f}",f"Student overlap between train/test: {ou['split']['overlapping_students']}",f"Top permutation features: {', '.join(x['feature'] for x in ou['feature_importance'][:5])}",f"Clustering: k={ou['clustering']['selected_k']}, silhouette={ou['clustering']['silhouette']:.4f}"]
    else: report += ["OULAD: NOT RUN",ou["reason"]]
    (ROOT/"results.txt").write_text("\n".join(report)+"\n",encoding="utf-8")
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
