import os
from itertools import chain
import pyspark.sql.functions as F
from pyspark.sql.functions import col, create_map, lit
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window


# Label-encoding maps for categorical columns
_OCCUPATION_MAP = {
    "Accountant": 0, "Architect": 1, "Developer": 2, "Doctor": 3,
    "Engineer": 4, "Entrepreneur": 5, "Journalist": 6, "Lawyer": 7,
    "Manager": 8, "Media_Manager": 9, "Mechanic": 10, "Musician": 11,
    "Scientist": 12, "Teacher": 13, "Writer": 14,
}

_CREDIT_MIX_MAP = {"Bad": 0, "Standard": 1, "Good": 2}

_PAYMENT_MIN_MAP = {"No": 0, "Yes": 1}

# Ordered from lowest to highest spend/value combination
_BEHAVIOUR_MAP = {
    "Low_spent_Small_value_payments": 0,
    "Low_spent_Medium_value_payments": 1,
    "Low_spent_Large_value_payments": 2,
    "High_spent_Small_value_payments": 3,
    "High_spent_Medium_value_payments": 4,
    "High_spent_Large_value_payments": 5,
}


def _make_mapping_expr(mapping: dict, col_name: str):
    """Build a create_map expression from a Python dict for label encoding."""
    kv_literals = [lit(x) for x in chain(*mapping.items())]
    return create_map(kv_literals)[col(col_name)].cast(IntegerType())


def process_features_gold_table(
    snapshot_date_str,
    silver_clickstream_dir,
    silver_attributes_dir,
    silver_financials_dir,
    gold_dir,
    spark,
):
    """
    Join monthly clickstream features with static attributes and financials,
    apply categorical encoding, and save as gold feature store partition.

    Attributes and financials are treated as customer-level reference tables
    (one record per customer regardless of snapshot_date) and joined only on
    Customer_ID to avoid dropping customers due to date misalignment.
    """
    clickstream_path = (
        silver_clickstream_dir
        + "silver_clickstream_"
        + snapshot_date_str.replace("-", "_")
        + ".parquet"
    )
    if not os.path.exists(clickstream_path):
        print(f"[gold feature store] missing clickstream silver for {snapshot_date_str}, skipping")
        return None

    df = spark.read.parquet(clickstream_path)

    # Join attributes: use the most recent record per customer whose snapshot_date
    # is <= the feature partition date, preventing future data leaking into earlier partitions.
    attributes_path = silver_attributes_dir + "silver_attributes.parquet"
    if os.path.exists(attributes_path):
        attrs = spark.read.parquet(attributes_path)
        attrs = attrs.filter(col("snapshot_date") <= snapshot_date_str)
        w = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
        attrs = attrs.withColumn("_rn", F.row_number().over(w)) \
                     .filter(col("_rn") == 1).drop("_rn", "snapshot_date")
        df = df.join(attrs, on="Customer_ID", how="left")

    # Join financials: same temporal-safe logic as attributes above.
    financials_path = silver_financials_dir + "silver_financials.parquet"
    if os.path.exists(financials_path):
        fins = spark.read.parquet(financials_path)
        fins = fins.filter(col("snapshot_date") <= snapshot_date_str)
        w = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
        fins = fins.withColumn("_rn", F.row_number().over(w)) \
                   .filter(col("_rn") == 1).drop("_rn", "snapshot_date")
        df = df.join(fins, on="Customer_ID", how="left")

    # --- Categorical encoding (string → integer) for ML compatibility ---

    df = df.withColumn("Occupation", _make_mapping_expr(_OCCUPATION_MAP, "Occupation"))
    df = df.withColumn("Credit_Mix", _make_mapping_expr(_CREDIT_MIX_MAP, "Credit_Mix"))
    df = df.withColumn("Payment_of_Min_Amount",
                       _make_mapping_expr(_PAYMENT_MIN_MAP, "Payment_of_Min_Amount"))
    df = df.withColumn("Payment_Behaviour",
                       _make_mapping_expr(_BEHAVIOUR_MAP, "Payment_Behaviour"))

    count = df.count()
    partition_name = (
        "gold_feature_store_" + snapshot_date_str.replace("-", "_") + ".parquet"
    )
    filepath_out = gold_dir + partition_name
    df.write.mode("overwrite").parquet(filepath_out)
    print(f"[gold feature store] {snapshot_date_str} row count: {count}, saved to: {filepath_out}")
    return df
