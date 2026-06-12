"""
Model inference: loads champion model bundle, applies the same imputer +
scaler used during training, scores the feature snapshot for a given month,
and writes predictions to datamart/gold/predictions/.
"""

import json
import os
import pickle

import pandas as pd


def run_inference(
    snapshot_date_str: str,
    gold_feature_store_dir: str,
    predictions_dir: str,
    model_store_dir: str,
):
    champion_path = os.path.join(model_store_dir, "champion_model.pkl")
    meta_path     = os.path.join(model_store_dir, "model_metadata.json")

    if not os.path.exists(champion_path):
        print(f"[inference] no champion model found — skipping {snapshot_date_str}")
        return

    with open(champion_path, "rb") as f:
        bundle = pickle.load(f)
    with open(meta_path) as f:
        metadata = json.load(f)

    clf           = bundle["pipeline"]
    imputer       = bundle["imputer"]
    scaler        = bundle["scaler"]
    feature_cols  = bundle["features"]
    model_version = metadata.get("model_version", "unknown")

    # Load feature store partition for this snapshot month
    date_tag     = snapshot_date_str.replace("-", "_")
    feature_file = os.path.join(
        gold_feature_store_dir, f"gold_feature_store_{date_tag}.parquet"
    )
    if not os.path.exists(feature_file):
        print(f"[inference] feature file missing for {snapshot_date_str} — skipping")
        return

    df = pd.read_parquet(feature_file)
    available = [c for c in feature_cols if c in df.columns]

    # Apply same preprocessing as training
    X           = df[available]
    X_imputed   = pd.DataFrame(imputer.transform(X),   columns=available)
    X_scaled    = scaler.transform(X_imputed)

    scores                = clf.predict_proba(X_scaled)[:, 1]
    df["score"]           = scores
    df["predicted_label"] = (scores >= 0.5).astype(int)
    df["snapshot_date"]   = snapshot_date_str
    df["model_version"]   = model_version

    os.makedirs(predictions_dir, exist_ok=True)
    out_path = os.path.join(predictions_dir, f"predictions_{date_tag}.parquet")
    df[["Customer_ID", "snapshot_date", "score", "predicted_label", "model_version"]].to_parquet(
        out_path, index=False
    )
    print(f"[inference] {snapshot_date_str} — {len(df)} rows → {out_path}")
