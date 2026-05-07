"""
src/train_models.py
All-in-one training script. Run this first before launching the app.
Usage: python src/train_models.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_raw, build_master
from src.preprocessing import clean, encode_categoricals, build_features, get_xy, scale, FEATURE_COLS, MODELS_DIR

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("  F1 ML PROJECT — MODEL TRAINING")
    print("=" * 60)

    print("\n[1/6] Loading raw CSVs ...")
    dfs = load_raw()
    master = build_master(dfs, year_cutoff=2000)
    print(f"      Master shape: {master.shape}")

    print("[2/6] Cleaning ...")
    master = clean(master)
    print(f"      After clean: {master.shape}")

    print("[3/6] Encoding categoricals ...")
    master, _ = encode_categoricals(master, fit=True)

    print("[4/6] Engineering features ...")
    master = build_features(master)
    master.to_csv(os.path.join(PROCESSED_DIR, "master.csv"), index=False)

    # also save featured version for DVC pipeline
    master.to_csv(os.path.join(PROCESSED_DIR, "featured_master.csv"), index=False)
    print("      Processed data saved.")

    print("[5/6] Splitting train/test ...")
    X, y_cls, y_reg, df_clean = get_xy(master)
    idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
    X_train, X_test = X[idx_tr], X[idx_te]
    y_cls_tr, y_cls_te = y_cls[idx_tr], y_cls[idx_te]
    y_reg_tr, y_reg_te = y_reg[idx_tr], y_reg[idx_te]
    X_train_s, X_test_s, _ = scale(X_train, X_test, fit=True)
    print(f"      Train: {X_train.shape}  Test: {X_test.shape}")

    print("\n[6/6] Training models ...\n")
    metrics = {}

    # ── Classification ──────────────────────────────────────
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM":                 SVC(probability=True, random_state=42),
        "Naive Bayes":         GaussianNB(),
    }
    for name, model in classifiers.items():
        print(f"  >> {name}")
        model.fit(X_train_s, y_cls_tr)
        y_pred = model.predict(X_test_s)
        cm = confusion_matrix(y_cls_te, y_pred).tolist()
        metrics[name] = {
            "type": "classification",
            "accuracy":  round(accuracy_score(y_cls_te, y_pred), 4),
            "precision": round(precision_score(y_cls_te, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_cls_te, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_cls_te, y_pred, zero_division=0), 4),
            "confusion_matrix": cm,
        }
        fname = name.replace(" ", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{fname}.pkl"))
        print(f"     Accuracy={metrics[name]['accuracy']}  F1={metrics[name]['f1']}")

    # ── Regression ──────────────────────────────────────────
    regressors = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Lasso Regression":  Lasso(alpha=0.1, max_iter=2000),
    }
    for name, model in regressors.items():
        print(f"  >> {name}")
        model.fit(X_train_s, y_reg_tr)
        y_pred = model.predict(X_test_s)
        metrics[name] = {
            "type": "regression",
            "mae": round(mean_absolute_error(y_reg_te, y_pred), 4),
            "mse": round(mean_squared_error(y_reg_te, y_pred), 4),
            "r2":  round(r2_score(y_reg_te, y_pred), 4),
        }
        fname = name.replace(" ", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{fname}.pkl"))
        print(f"     MAE={metrics[name]['mae']}  R2={metrics[name]['r2']}")

    # ── Clustering ──────────────────────────────────────────
    print("  >> K-Means Clustering (k=5)")
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    km.fit(X_train_s)
    joblib.dump(km, os.path.join(MODELS_DIR, "KMeans.pkl"))
    metrics["K-Means"] = {"type": "clustering", "k": 5, "inertia": round(float(km.inertia_), 2)}
    print(f"     Inertia={km.inertia_:.2f}")

    # ── Save artifacts ──────────────────────────────────────
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    rf = joblib.load(os.path.join(MODELS_DIR, "Random_Forest.pkl"))
    fi = dict(zip(FEATURE_COLS, [round(float(v), 6) for v in rf.feature_importances_]))
    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
        json.dump(fi, f, indent=2)

    # meta for UI dropdowns
    meta = {
        "drivers":          sorted(df_clean["driver_name"].dropna().unique().tolist()),
        "constructors":     sorted(df_clean["team_name"].dropna().unique().tolist()),
        "circuits":         sorted(df_clean["circuit_name"].dropna().unique().tolist()),
        "driverRef_map":    df_clean.groupby("driver_name")["driverRef"].first().to_dict(),
        "constructorRef_map": df_clean.groupby("team_name")["constructorRef"].first().to_dict(),
        "circuitRef_map":   df_clean.groupby("circuit_name")["circuitRef"].first().to_dict(),
        "years":            sorted([int(y) for y in df_clean["year"].unique().tolist()]),
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # strategy table
    strat = (
        df_clean.groupby("circuitRef")
        .agg(avg_pit_count=("pit_stop_count","mean"),
             avg_lap_ms=("avg_lap_ms","mean"),
             circuit_name=("circuit_name","first"),
             country=("country","first"))
        .reset_index()
    )
    strat["avg_pit_count"] = strat["avg_pit_count"].round(2)
    strat["avg_lap_ms"]    = strat["avg_lap_ms"].round(0)
    strat.to_csv(os.path.join(PROCESSED_DIR, "strategy_table.csv"), index=False)

    print("\n" + "=" * 60)
    print("  ALL MODELS TRAINED AND SAVED")
    print("  Run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
