import os
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import col


def process_bronze_clickstream(snapshot_date_str, bronze_dir, spark=None):
    """Filter clickstream raw CSV by snapshot_date and save as monthly bronze partition."""
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_dir + partition_name
    if os.path.exists(filepath):
        print(f"[bronze clickstream] {snapshot_date_str} already exists, skipping")
        return None
    df = pd.read_csv("data/feature_clickstream.csv", dtype=str)
    df = df[df["snapshot_date"] == snapshot_date_str]
    count = len(df)
    if count == 0:
        print(f"[bronze clickstream] no data for {snapshot_date_str}, skipping")
        return None
    df.to_csv(filepath, index=False)
    print(f"[bronze clickstream] {snapshot_date_str} row count: {count}, saved to: {filepath}")
    return df


def process_bronze_attributes(bronze_dir, spark=None):
    """Ingest full attributes raw CSV as a single bronze table (one record per customer)."""
    filepath = bronze_dir + "bronze_attributes.csv"
    if os.path.exists(filepath):
        print(f"[bronze attributes] already exists, skipping")
        return
    df = pd.read_csv("data/features_attributes.csv", dtype=str)
    df.to_csv(filepath, index=False)
    print(f"[bronze attributes] saved {len(df)} rows to {filepath}")


def process_bronze_financials(bronze_dir, spark=None):
    """Ingest full financials raw CSV as a single bronze table (one record per customer)."""
    filepath = bronze_dir + "bronze_financials.csv"
    if os.path.exists(filepath):
        print(f"[bronze financials] already exists, skipping")
        return
    df = pd.read_csv("data/features_financials.csv", dtype=str)
    df.to_csv(filepath, index=False)
    print(f"[bronze financials] saved {len(df)} rows to {filepath}")
