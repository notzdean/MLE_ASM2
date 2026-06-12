"""
Model training: joins feature store + label store, temporal train/test split,
trains LR / RF / XGBoost, picks best AUC, saves champion to model_store/.
"""

import os
import glob
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# All ML-ready columns produced by the gold feature store
FEATURE_COLS = (
    [f"fe_{i}" for i in range(1, 21)]          # clickstream (20)
    + [
        "Age", "Occupation",                    # attributes (2)
        "Annual_Income", "Monthly_Inhand_Salary",
        "Num_Bank_Accounts", "Num_Credit_Card",
        "Interest_Rate", "Num_of_Loan", "num_loan_types",
        "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries",
        "Credit_Mix", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "credit_history_months",
        "Payment_of_Min_Amount", "Total_EMI_per_month",
        "Amount_invested_monthly", "Payment_Behaviour",
        "Monthly_Balance",                      # financials (22)
    ]
)


def _gini(y_true, y_score):
    return 2 * roc_auc_score(y_true, y_score) - 1


def _ks(y_true, y_score):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    stat, _ = ks_2samp(pos, neg)
    return float(stat)


def _load_labeled_dataset(gold_feature_store_dir: str, gold_label_store_dir: str) -> pd.DataFrame:
    """
    Join feature store with label store using the 6-month MOB offset.

    Label store partition for date L contains customers whose loan originated
    at L - 6 months (mob = 6 at date L). We pair each label partition with
    the corresponding feature partition at origination month.
    """
    label_files = sorted(glob.glob(os.path.join(gold_label_store_dir, "*.parquet")))
    if not label_files:
        return pd.DataFrame()

    records = []
    for lf in label_files:
        # Extract date from filename: gold_label_store_YYYY_MM_DD.parquet
        basename = os.path.basename(lf)
        date_part = basename.replace("gold_label_store_", "").replace(".parquet", "")
        label_date_str = date_part.replace("_", "-")            # e.g. 2023-07-01

        try:
            label_date = pd.Timestamp(label_date_str)
        except Exception:
            continue

        # Feature snapshot = origination month = label date − 6 months
        feature_date = label_date - relativedelta(months=6)
        feature_date_str = feature_date.strftime("%Y_%m_%d")    # e.g. 2023_01_01

        feature_file = os.path.join(
            gold_feature_store_dir,
            f"gold_feature_store_{feature_date_str}.parquet",
        )
        if not os.path.exists(feature_file):
            continue

        labels = pd.read_parquet(lf)[["Customer_ID", "label"]]
        features = pd.read_parquet(feature_file)

        merged = features.merge(labels, on="Customer_ID", how="inner")
        merged["feature_snapshot_date"] = feature_date.strftime("%Y-%m-%d")
        records.append(merged)

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


def train_models(
    gold_feature_store_dir: str,
    gold_label_store_dir: str,
    model_store_dir: str,
    train_end_date: str = "2024-03-01",
):
    """
    Train candidate models and persist the champion.

    Temporal split (no shuffle to prevent leakage):
      train: feature_snapshot_date ≤ train_end_date
      test:  feature_snapshot_date >  train_end_date
    """
    print("[training] loading labeled dataset ...")
    df = _load_labeled_dataset(gold_feature_store_dir, gold_label_store_dir)

    if df.empty:
        print("[training] no labeled data available yet — skipping")
        return None

    df["label"] = df["label"].astype(int)
    df["feature_snapshot_date"] = pd.to_datetime(df["feature_snapshot_date"])

    available_features = [c for c in FEATURE_COLS if c in df.columns]

    train_df = df[df["feature_snapshot_date"] <= train_end_date]
    test_df  = df[df["feature_snapshot_date"] >  train_end_date]

    if len(train_df) < 50:
        print(f"[training] only {len(train_df)} training rows — skipping until more data")
        return None

    print(f"[training] train rows: {len(train_df)} | test rows: {len(test_df)}")

    X_train = train_df[available_features]
    y_train = train_df["label"]

    # Define candidate pipelines
    candidates = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
            )),
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     xgb.XGBClassifier(
                n_estimators=200, random_state=42, eval_metric="logloss",
                verbosity=0, scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            )),
        ]),
    }

    results = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)

        # Evaluate on test set if available, else fall back to train
        eval_df = test_df if len(test_df) >= 10 else train_df
        X_eval  = eval_df[available_features]
        y_eval  = eval_df["label"]

        y_prob  = pipeline.predict_proba(X_eval)[:, 1]
        auc     = float(roc_auc_score(y_eval, y_prob)) if y_eval.nunique() > 1 else 0.5
        gini    = _gini(y_eval, y_prob) if y_eval.nunique() > 1 else 0.0
        ks      = _ks(y_eval.values, y_prob)

        results[name] = {"auc": auc, "gini": gini, "ks": ks, "pipeline": pipeline}
        print(f"  {name:25s}  AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

    # Champion = highest AUC on eval set
    champion_name = max(results, key=lambda k: results[k]["auc"])
    champion      = results[champion_name]
    print(f"[training] champion: {champion_name}")

    # Training score distribution — used as PSI baseline
    train_scores = champion["pipeline"].predict_proba(X_train)[:, 1]
    pct_keys     = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    metadata = {
        "model_name":    champion_name,
        "train_date":    datetime.now().isoformat(),
        "train_end_date": train_end_date,
        "features":      available_features,
        "metrics": {
            "auc":  champion["auc"],
            "gini": champion["gini"],
            "ks":   champion["ks"],
        },
        "all_models": {
            n: {"auc": r["auc"], "gini": r["gini"], "ks": r["ks"]}
            for n, r in results.items()
        },
        "train_score_distribution": {
            "mean":        float(train_scores.mean()),
            "std":         float(train_scores.std()),
            "percentiles": {str(p): float(np.percentile(train_scores, p)) for p in pct_keys},
        },
    }

    os.makedirs(model_store_dir, exist_ok=True)
    model_path = os.path.join(model_store_dir, "champion_model.pkl")
    meta_path  = os.path.join(model_store_dir, "model_metadata.json")

    with open(model_path, "wb") as f:
        pickle.dump(champion["pipeline"], f)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[training] model  → {model_path}")
    print(f"[training] meta   → {meta_path}")
    return metadata
