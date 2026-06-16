"""
Model inference: loads champion model (and challenger if in shadow mode),
applies the same preprocessing used at training time, scores the feature
snapshot for a given month, and writes predictions to datamart/gold/predictions/.

Shadow mode: if challenger_model.pkl exists alongside champion_model.pkl, both
models score the same customers. The champion score is used for business
decisions (predicted_label). The challenger score is recorded for comparison
in model_monitoring so it can be evaluated against the champion over time.

Scoring population: ALL visitors in the origination-month feature store (~8974/month).
This gives a stable same-size population for rolling PSI comparisons.
The monitoring task inner-joins predictions with the label store to compute AUC,
so performance metrics are computed only on the ~530 labeled loan customers.
"""

import glob
import json
import os
import pickle

import pandas as pd
from dateutil.relativedelta import relativedelta

from utils.model_training import _engineer_features


def _score_bundle(bundle: dict, df: pd.DataFrame) -> pd.Series:
    """Apply a model bundle's preprocessing and return predicted probabilities."""
    clf          = bundle["pipeline"]
    imputer      = bundle["imputer"]
    scaler       = bundle["scaler"]
    feature_cols = bundle["features"]

    available = [c for c in feature_cols if c in df.columns]
    X         = df[available]
    X_imputed = pd.DataFrame(imputer.transform(X), columns=available)
    X_scaled  = scaler.transform(X_imputed)
    return pd.Series(clf.predict_proba(X_scaled)[:, 1], index=df.index)


def run_inference(
    snapshot_date_str: str,
    gold_feature_store_dir: str,
    predictions_dir: str,
    model_store_dir: str,
    gold_label_store_dir: str = None,
):
    champion_path = os.path.join(model_store_dir, "champion_model.pkl")
    meta_path     = os.path.join(model_store_dir, "model_metadata.json")

    if not os.path.exists(champion_path):
        print(f"[inference] no champion model found — skipping {snapshot_date_str}")
        return

    with open(champion_path, "rb") as f:
        champion_bundle = pickle.load(f)
    with open(meta_path) as f:
        metadata = json.load(f)

    model_version = metadata.get("model_version", "unknown")

    date_tag = snapshot_date_str.replace("-", "_")

    # Load features from ORIGINATION month (snapshot_date - 6 months).
    # The model was trained on origination-time features (see model_training.py
    # _load_labeled_dataset which uses label_date - 6 months for feature lookup).
    # Using the same origination-month features at inference aligns train/inference.
    origination_date     = pd.Timestamp(snapshot_date_str) - relativedelta(months=6)
    origination_date_tag = origination_date.strftime("%Y_%m_%d")
    feature_file         = os.path.join(
        gold_feature_store_dir, f"gold_feature_store_{origination_date_tag}.parquet"
    )
    if not os.path.exists(feature_file):
        print(f"[inference] origination feature file missing for {snapshot_date_str} "
              f"(expected {origination_date_tag}) — skipping")
        return

    df = pd.read_parquet(feature_file)
    print(
        f"[inference] {snapshot_date_str} — scoring all {len(df)} visitors "
        f"from origination month {origination_date_tag}"
    )

    df = _engineer_features(df)   # same ratio features applied at training time

    # --- Champion scoring (production score) ---
    df["score"]           = _score_bundle(champion_bundle, df)
    df["predicted_label"] = (df["score"] >= 0.5).astype(int)
    df["snapshot_date"]   = snapshot_date_str
    df["model_version"]   = model_version

    # --- Challenger scoring (shadow mode — recorded but not used for decisions) ---
    challenger_path = os.path.join(model_store_dir, "challenger_model.pkl")
    challenger_meta = os.path.join(model_store_dir, "challenger_metadata.json")

    if os.path.exists(challenger_path):
        with open(challenger_path, "rb") as f:
            challenger_bundle = pickle.load(f)

        df["challenger_score"]           = _score_bundle(challenger_bundle, df)
        df["challenger_predicted_label"] = (df["challenger_score"] >= 0.5).astype(int)

        ch_version = "unknown"
        if os.path.exists(challenger_meta):
            with open(challenger_meta) as f:
                ch_meta = json.load(f)
            ch_version = ch_meta.get("model_version", "unknown")
        df["challenger_model_version"] = ch_version

        print(
            f"[inference] {snapshot_date_str} — shadow mode active  "
            f"champion={model_version}  challenger={ch_version}"
        )
    else:
        print(f"[inference] {snapshot_date_str} — champion-only (no challenger in shadow)")

    # Save predictions
    os.makedirs(predictions_dir, exist_ok=True)
    save_cols = ["Customer_ID", "snapshot_date", "score", "predicted_label", "model_version"]
    if "challenger_score" in df.columns:
        save_cols += ["challenger_score", "challenger_predicted_label", "challenger_model_version"]

    out_path = os.path.join(predictions_dir, f"predictions_{date_tag}.parquet")
    df[save_cols].to_parquet(out_path, index=False)
    print(f"[inference] {snapshot_date_str} — {len(df)} rows → {out_path}")
