"""
Model training: joins feature store + label store, 3-way temporal split
(train / test / OOT), RandomizedSearchCV tuning on XGBoost, picks best
champion by OOT AUC, saves versioned artifact to model_store/.

Temporal split (no shuffle — prevents data leakage):
  Train : feature_snapshot_date <= TRAIN_END_DATE   (Jan 2023 – Dec 2023)
  Test  : TRAIN_END_DATE < date <= TEST_END_DATE     (Jan 2024 – Mar 2024)
  OOT   : date > TEST_END_DATE                       (Apr 2024 – Jun 2024)
"""

import os
import glob
import json
import pickle
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


TRAIN_END_DATE = "2023-12-01"   # last training month (inclusive)
TEST_END_DATE  = "2024-03-01"   # last test month (inclusive); after this = OOT
TOP_N_FEATURES = 10             # features tracked for CSI baseline

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


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _eval_metrics(y_true, y_score):
    """Return AUC, Gini, KS for a set of predictions. Handles edge cases."""
    if len(y_true) < 5 or y_true.nunique() < 2:
        return {"auc": None, "gini": None, "ks": None}
    auc  = float(roc_auc_score(y_true, y_score))
    gini = float(2 * auc - 1)
    pos  = y_score[y_true == 1]
    neg  = y_score[y_true == 0]
    ks   = float(ks_2samp(pos, neg)[0]) if (len(pos) > 0 and len(neg) > 0) else None
    return {"auc": auc, "gini": gini, "ks": ks}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_labeled_dataset(gold_feature_store_dir: str, gold_label_store_dir: str) -> pd.DataFrame:
    """
    Join feature store with label store using the 6-month MOB offset.

    Label partition for date L contains customers whose loan originated at
    L − 6 months (mob = 6 at L). We pair with the feature partition for
    the origination month to avoid any future-data leakage.
    """
    label_files = sorted(glob.glob(os.path.join(gold_label_store_dir, "*.parquet")))
    if not label_files:
        return pd.DataFrame()

    records = []
    for lf in label_files:
        basename       = os.path.basename(lf)
        date_part      = basename.replace("gold_label_store_", "").replace(".parquet", "")
        label_date_str = date_part.replace("_", "-")

        try:
            label_date = pd.Timestamp(label_date_str)
        except Exception:
            continue

        feature_date     = label_date - relativedelta(months=6)
        feature_date_tag = feature_date.strftime("%Y_%m_%d")
        feature_file     = os.path.join(
            gold_feature_store_dir,
            f"gold_feature_store_{feature_date_tag}.parquet",
        )
        if not os.path.exists(feature_file):
            continue

        labels   = pd.read_parquet(lf)[["Customer_ID", "label"]]
        features = pd.read_parquet(feature_file)
        merged   = features.merge(labels, on="Customer_ID", how="inner")
        merged["feature_snapshot_date"] = feature_date.strftime("%Y-%m-%d")
        records.append(merged)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_models(
    gold_feature_store_dir: str,
    gold_label_store_dir: str,
    model_store_dir: str,
    train_end_date: str = TRAIN_END_DATE,
    test_end_date: str  = TEST_END_DATE,
):
    print("[training] loading labeled dataset ...")
    df = _load_labeled_dataset(gold_feature_store_dir, gold_label_store_dir)

    if df.empty:
        print("[training] no labeled data available yet — skipping")
        return None

    df["label"]                 = df["label"].astype(int)
    df["feature_snapshot_date"] = pd.to_datetime(df["feature_snapshot_date"])
    available_features          = [c for c in FEATURE_COLS if c in df.columns]

    # 3-way temporal split
    train_df = df[df["feature_snapshot_date"] <= train_end_date]
    test_df  = df[(df["feature_snapshot_date"] >  train_end_date) &
                  (df["feature_snapshot_date"] <= test_end_date)]
    oot_df   = df[df["feature_snapshot_date"] >  test_end_date]

    if len(train_df) < 50:
        print(f"[training] only {len(train_df)} training rows — skipping")
        return None

    print(
        f"[training] split  →  train: {len(train_df)}  "
        f"test: {len(test_df)}  OOT: {len(oot_df)}"
    )
    print(
        f"[training] label prevalence  →  "
        f"train: {train_df['label'].mean():.3f}  "
        f"test: {test_df['label'].mean():.3f}  "
        f"OOT: {oot_df['label'].mean():.3f if len(oot_df) else 'N/A'}"
    )

    X_train = train_df[available_features]
    y_train = train_df["label"]
    X_test  = test_df[available_features] if len(test_df) >= 10 else X_train
    y_test  = test_df["label"]            if len(test_df) >= 10 else y_train
    X_oot   = oot_df[available_features]  if len(oot_df)  >= 10 else pd.DataFrame()
    y_oot   = oot_df["label"]             if len(oot_df)  >= 10 else pd.Series(dtype=int)

    # Imputer fitted once on training data, shared across all pipelines
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    X_train_imp = pd.DataFrame(imputer.transform(X_train), columns=available_features)
    X_test_imp  = pd.DataFrame(imputer.transform(X_test),  columns=available_features)
    X_oot_imp   = pd.DataFrame(imputer.transform(X_oot),   columns=available_features) if not X_oot.empty else pd.DataFrame()

    # Scaler fitted on training data
    scaler = StandardScaler()
    scaler.fit(X_train_imp)
    X_train_sc = scaler.transform(X_train_imp)
    X_test_sc  = scaler.transform(X_test_imp)
    X_oot_sc   = scaler.transform(X_oot_imp) if not X_oot_imp.empty else np.array([])

    # ------------------------------------------------------------------
    # 1. Logistic Regression (interpretable baseline)
    # ------------------------------------------------------------------
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_sc, y_train)

    # ------------------------------------------------------------------
    # 2. Random Forest
    # ------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42,
        n_jobs=-1, class_weight="balanced"
    )
    rf.fit(X_train_sc, y_train)

    # ------------------------------------------------------------------
    # 3. XGBoost with RandomizedSearchCV (matches professor's approach)
    # ------------------------------------------------------------------
    print("[training] running RandomizedSearchCV for XGBoost ...")
    xgb_base = xgb.XGBClassifier(
        eval_metric="logloss", random_state=42, verbosity=0,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
    )
    param_dist = {
        "n_estimators":    [25, 50, 100, 200],
        "max_depth":       [2, 3, 4],
        "learning_rate":   [0.01, 0.05, 0.1],
        "subsample":       [0.6, 0.8, 1.0],
        "colsample_bytree":[0.6, 0.8, 1.0],
        "gamma":           [0, 0.1, 0.2],
        "min_child_weight":[1, 3, 5],
        "reg_alpha":       [0, 0.1, 1.0],
        "reg_lambda":      [1.0, 1.5, 2.0],
    }
    auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
    rscv = RandomizedSearchCV(
        estimator=xgb_base, param_distributions=param_dist,
        scoring=auc_scorer, n_iter=10, cv=3,
        verbose=1, random_state=42, n_jobs=-1,
    )
    rscv.fit(X_train_sc, y_train)
    xgb_best = rscv.best_estimator_
    best_hp   = rscv.best_params_
    print(f"[training] XGBoost best params: {best_hp}")

    # ------------------------------------------------------------------
    # Evaluate all three on train / test / OOT
    # ------------------------------------------------------------------
    candidates = {
        "logistic_regression": lr,
        "random_forest":       rf,
        "xgboost":             xgb_best,
    }
    results = {}
    for name, clf in candidates.items():
        train_prob = clf.predict_proba(X_train_sc)[:, 1]
        test_prob  = clf.predict_proba(X_test_sc)[:, 1]

        m_train = _eval_metrics(y_train, pd.Series(train_prob))
        m_test  = _eval_metrics(y_test,  pd.Series(test_prob))
        m_oot   = _eval_metrics(y_oot, pd.Series(clf.predict_proba(X_oot_sc)[:, 1])) \
                  if len(X_oot_sc) > 0 else {"auc": None, "gini": None, "ks": None}

        results[name] = {
            "train": m_train, "test": m_test, "oot": m_oot,
            "clf": clf,
        }
        print(
            f"  {name:25s}  "
            f"Train AUC={m_train['auc']:.4f}  "
            f"Test AUC={m_test['auc']:.4f}  "
            f"OOT AUC={m_oot['auc']:.4f if m_oot['auc'] else 'N/A'}"
        )

    # Champion = best OOT AUC (fall back to test AUC if no OOT)
    def _score(name):
        r = results[name]
        return r["oot"]["auc"] or r["test"]["auc"] or 0.0

    champion_name = max(results, key=_score)
    champion      = results[champion_name]
    print(f"[training] champion: {champion_name}  OOT AUC={_score(champion_name):.4f}")

    # ------------------------------------------------------------------
    # Feature importance from XGBoost (for CSI baseline in monitoring)
    # ------------------------------------------------------------------
    xgb_importances = {}
    if hasattr(xgb_best, "feature_importances_"):
        importances = xgb_best.feature_importances_
        xgb_importances = dict(
            sorted(
                zip(available_features, importances.tolist()),
                key=lambda x: x[1], reverse=True
            )[:TOP_N_FEATURES]
        )

    # Training score distribution — PSI baseline
    train_scores = champion["clf"].predict_proba(X_train_sc)[:, 1]
    pct_keys     = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    # Feature baseline stats for CSI (mean + std of top features on training data)
    top_features = list(xgb_importances.keys()) if xgb_importances else available_features[:TOP_N_FEATURES]
    feature_baseline_stats = {}
    for feat in top_features:
        if feat in X_train_imp.columns:
            col_vals = X_train_imp[feat].dropna()
            feature_baseline_stats[feat] = {
                "mean": float(col_vals.mean()),
                "std":  float(col_vals.std()),
                "p25":  float(col_vals.quantile(0.25)),
                "p75":  float(col_vals.quantile(0.75)),
            }

    # ------------------------------------------------------------------
    # Build and persist artifact
    # ------------------------------------------------------------------
    version_tag   = datetime.now().strftime("%Y_%m_%d")
    model_version = f"credit_model_{version_tag}"

    artifact = {
        "model_name":    champion_name,
        "model_version": model_version,
        "train_date":    datetime.now().isoformat(),
        "data_dates": {
            "train_start": str(train_df["feature_snapshot_date"].min().date()),
            "train_end":   train_end_date,
            "test_start":  str(test_df["feature_snapshot_date"].min().date()) if len(test_df) else None,
            "test_end":    test_end_date,
            "oot_start":   str(oot_df["feature_snapshot_date"].min().date()) if len(oot_df) else None,
            "oot_end":     str(oot_df["feature_snapshot_date"].max().date()) if len(oot_df) else None,
        },
        "data_stats": {
            "n_train":               len(X_train),
            "n_test":                len(X_test),
            "n_oot":                 len(X_oot),
            "label_prevalence_train": float(y_train.mean()),
            "label_prevalence_test":  float(y_test.mean()) if len(y_test) else None,
            "label_prevalence_oot":   float(y_oot.mean())  if len(y_oot)  else None,
        },
        "features": available_features,
        "top_features_by_importance": xgb_importances,
        "metrics": {
            name: {"train": r["train"], "test": r["test"], "oot": r["oot"]}
            for name, r in results.items()
        },
        "hp_params": best_hp,
        "train_score_distribution": {
            "mean":        float(train_scores.mean()),
            "std":         float(train_scores.std()),
            "percentiles": {str(p): float(np.percentile(train_scores, p)) for p in pct_keys},
        },
        "feature_baseline_stats": feature_baseline_stats,
        "preprocessing": {
            "imputer_strategy": "median",
            "scaler":           "StandardScaler",
        },
    }

    # Bundle model + preprocessors so inference applies same transforms
    model_bundle = {
        "pipeline":  champion["clf"],
        "imputer":   imputer,
        "scaler":    scaler,
        "features":  available_features,
        "metadata":  artifact,
    }

    os.makedirs(model_store_dir, exist_ok=True)
    champion_path  = os.path.join(model_store_dir, "champion_model.pkl")
    versioned_path = os.path.join(model_store_dir, f"{model_version}.pkl")
    meta_path      = os.path.join(model_store_dir, "model_metadata.json")

    with open(champion_path, "wb") as f:
        pickle.dump(model_bundle, f)

    # Keep a timestamped copy for the model history / challenger comparison
    shutil.copy2(champion_path, versioned_path)

    with open(meta_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)

    print(f"[training] champion  → {champion_path}")
    print(f"[training] versioned → {versioned_path}")
    print(f"[training] metadata  → {meta_path}")
    return artifact
