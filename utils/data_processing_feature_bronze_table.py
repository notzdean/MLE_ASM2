import os
import pyspark.sql.functions as F
from pyspark.sql.functions import col


def process_bronze_clickstream(snapshot_date_str, bronze_dir, spark):
    """Filter clickstream raw CSV by snapshot_date and save as monthly bronze partition."""
    df = spark.read.csv("data/feature_clickstream.csv", header=True, inferSchema=False)
    df = df.filter(col("snapshot_date") == snapshot_date_str)
    count = df.count()
    if count == 0:
        print(f"[bronze clickstream] no data for {snapshot_date_str}, skipping")
        return None
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_dir + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print(f"[bronze clickstream] {snapshot_date_str} row count: {count}, saved to: {filepath}")
    return df


def process_bronze_attributes(bronze_dir, spark):
    """Ingest full attributes raw CSV as a single bronze table (one record per customer)."""
    filepath = bronze_dir + "bronze_attributes.csv"
    if os.path.exists(filepath):
        print(f"[bronze attributes] already exists, skipping")
        return
    df = spark.read.csv("data/features_attributes.csv", header=True, inferSchema=False)
    count = df.count()
    df.toPandas().to_csv(filepath, index=False)
    print(f"[bronze attributes] saved {count} rows to {filepath}")


def process_bronze_financials(bronze_dir, spark):
    """Ingest full financials raw CSV as a single bronze table (one record per customer)."""
    filepath = bronze_dir + "bronze_financials.csv"
    if os.path.exists(filepath):
        print(f"[bronze financials] already exists, skipping")
        return
    df = spark.read.csv("data/features_financials.csv", header=True, inferSchema=False)
    count = df.count()
    df.toPandas().to_csv(filepath, index=False)
    print(f"[bronze financials] saved {count} rows to {filepath}")
