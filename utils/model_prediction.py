"""
Model inference: loads champion model (and challenger if in shadow mode),
applies the same preprocessing used at training time, scores the feature
snapshot for a given month, and writes predictions to datamart/gold/predictions/.

Shadow mode: if challenger_model.pkl exists alongside champion_model.pkl, both
models score the same customers. The champion score is used for business
decisions (predicted_label). The challenger score is recorded for comparison
in model_monitoring so it can be evaluated against the champion over time.
"""

import glob
import json
import os
import pickle

import pandas as pd

from utils.model_training import _engineer_features


def _loan_customer_ids(gold_label_store_dir: str) -> set:
    """Return the set of Customer_IDs that appear in any label store partition.
    These are definitively loan customers — the only population the model was
    trained on and the only population that makes business sense to score."""
    ids = set()
    for f in glob.glob(os.path.join(gold_label_store_dir, "*.parquet")):
        ids.update(pd.read_parquet(f, columns=["Customer_ID"])["Customer_ID"].tolist())
    return ids


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

    # Load feature store partition for this snapshot month
    date_tag     = snapshot_date_str.replace("-", "_")
    feature_file = os.path.join(
        gold_feature_store_dir, f"gold_feature_store_{date_tag}.parquet"
    )
    if not os.path.exists(feature_file):
        print(f"[inference] feature file missing for {snapshot_date_str} — skipping")
        return

    df = pd.read_parquet(feature_file)

    # Filter to loan customers only. Annual_Income grows non-null over time as the
    # financials join picks up more historical records, so notna() alone is not a
    # reliable filter. Instead use the label store — every Customer_ID that ever
    # appears there is definitively a loan customer.
    n_total = len(df)
    if gold_label_store_dir and os.path.isdir(gold_label_store_dir):
        loan_ids = _loan_customer_ids(gold_label_store_dir)
        df = df[df["Customer_ID"].isin(loan_ids)].copy()
    else:
        df = df[df["Annual_Income"].notna()].copy()
    print(
        f"[inference] {snapshot_date_str} — filtered to loan customers: "
        f"{len(df)}/{n_total} rows"
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

        df["challenger_score"] = _score_bundle(challenger_bundle, df)

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
        save_cols += ["challenger_score", "challenger_model_version"]

    out_path = os.path.join(predictions_dir, f"predictions_{date_tag}.parquet")
    df[save_cols].to_parquet(out_path, index=False)
    print(f"[inference] {snapshot_date_str} — {len(df)} rows → {out_path}")
