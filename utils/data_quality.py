"""
Data quality gate: validates gold feature store and label store partitions
for a given snapshot month before allowing model training and inference.

Raises ValueError on critical failures — this fails the Airflow task and
marks all downstream tasks as upstream_failed, preventing bad data from
reaching the model.
"""

import os
import pandas as pd

# Minimum expected columns in the gold feature store
REQUIRED_FEATURE_COLS = (
    ["Customer_ID", "snapshot_date"]
    + [f"fe_{i}" for i in range(1, 21)]
    + ["Age", "Annual_Income", "Monthly_Inhand_Salary", "Outstanding_Debt"]
)

MIN_ROWS          = 10      # below this is suspicious
NULL_RATE_WARN    = 0.30    # warn if any feature > 30 % null
NULL_RATE_FAIL    = 0.80    # fail if any feature > 80 % null
DUPLICATE_ID_WARN = 0.05    # warn if >5 % duplicate Customer_IDs


def run_data_quality(
    snapshot_date_str: str,
    gold_feature_store_dir: str,
    gold_label_store_dir: str,
):
    """
    Run data quality checks on the gold partitions for snapshot_date_str.
    Prints a full QC report and raises ValueError if any critical check fails.
    """
    print(f"\n{'='*60}")
    print(f"[data_quality] QC report for {snapshot_date_str}")
    print(f"{'='*60}")

    date_tag      = snapshot_date_str.replace("-", "_")
    feature_file  = os.path.join(gold_feature_store_dir, f"gold_feature_store_{date_tag}.parquet")
    label_file    = os.path.join(gold_label_store_dir,   f"gold_label_store_{date_tag}.parquet")

    issues   = []   # critical — will raise
    warnings = []   # non-critical — logged only

    # ------------------------------------------------------------------
    # 1. Feature store checks
    # ------------------------------------------------------------------
    print("\n[data_quality] --- Feature Store ---")

    if not os.path.exists(feature_file):
        issues.append(f"Feature store partition missing: {feature_file}")
    else:
        df = pd.read_parquet(feature_file)
        n_rows, n_cols = df.shape

        print(f"  Rows       : {n_rows}")
        print(f"  Columns    : {n_cols}")

        # Row count
        if n_rows == 0:
            issues.append("Feature store is empty (0 rows)")
        elif n_rows < MIN_ROWS:
            warnings.append(f"Feature store has only {n_rows} rows — possibly incomplete ingestion")

        # Required columns
        missing = [c for c in REQUIRED_FEATURE_COLS if c not in df.columns]
        if missing:
            issues.append(f"Missing required columns: {missing}")

        # Null rates
        if n_rows > 0:
            null_rates = df.isnull().mean().sort_values(ascending=False)
            critical_nulls = null_rates[null_rates > NULL_RATE_FAIL].index.tolist()
            high_nulls     = null_rates[(null_rates > NULL_RATE_WARN) &
                                        (null_rates <= NULL_RATE_FAIL)].index.tolist()

            if critical_nulls:
                issues.append(f"Critical null rate (>{NULL_RATE_FAIL:.0%}) in: {critical_nulls}")
            if high_nulls:
                warnings.append(f"High null rate (>{NULL_RATE_WARN:.0%}) in: {high_nulls}")

            top_null = null_rates.head(5)
            print("  Top-5 null rates:")
            for col, rate in top_null.items():
                flag = "❌" if rate > NULL_RATE_FAIL else ("⚠️" if rate > NULL_RATE_WARN else "✓")
                print(f"    {flag}  {col:35s} {rate:.1%}")

        # Customer_ID uniqueness
        if "Customer_ID" in df.columns and n_rows > 0:
            dup_rate = df["Customer_ID"].duplicated().mean()
            print(f"  Duplicate Customer_IDs: {dup_rate:.1%}")
            if dup_rate > DUPLICATE_ID_WARN:
                warnings.append(f"High duplicate Customer_ID rate: {dup_rate:.1%}")

        # Numeric range sanity (clickstream features should be non-negative)
        fe_cols = [c for c in df.columns if c.startswith("fe_")]
        if fe_cols:
            neg_fe = [c for c in fe_cols if df[c].dropna().lt(0).any()]
            if neg_fe:
                warnings.append(f"Negative values in clickstream features: {neg_fe}")

    # ------------------------------------------------------------------
    # 2. Label store checks
    # ------------------------------------------------------------------
    print("\n[data_quality] --- Label Store ---")

    if not os.path.exists(label_file):
        # Label store for this month only exists 6 months after origination — not an error
        print(f"  Label store not yet available for {snapshot_date_str} (expected after 6 months)")
    else:
        ldf = pd.read_parquet(label_file)
        n_label_rows = len(ldf)
        print(f"  Rows       : {n_label_rows}")

        if n_label_rows == 0:
            warnings.append("Label store is empty — no labeled customers for this month")
        elif "label" in ldf.columns:
            default_rate = ldf["label"].mean()
            print(f"  Default rate : {default_rate:.2%}")
            if default_rate == 0.0:
                warnings.append("Default rate is 0.0 — all labels are 0, check label definition")
            elif default_rate > 0.8:
                warnings.append(f"Unusually high default rate: {default_rate:.2%}")

    # ------------------------------------------------------------------
    # 3. Report
    # ------------------------------------------------------------------
    print(f"\n[data_quality] Warnings ({len(warnings)}):")
    for w in warnings:
        print(f"  ⚠️  {w}")

    print(f"\n[data_quality] Critical issues ({len(issues)}):")
    for i in issues:
        print(f"  ❌  {i}")

    if issues:
        raise ValueError(
            f"[data_quality] FAILED for {snapshot_date_str} — "
            f"{len(issues)} critical issue(s): {'; '.join(issues)}"
        )

    print(f"\n[data_quality] ✓ PASSED for {snapshot_date_str}")
    print(f"{'='*60}\n")
    return True
