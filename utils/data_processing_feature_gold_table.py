import os

import pandas as pd

# Label-encoding maps for categorical columns (identical to original PySpark version)
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


def process_features_gold_table(
    snapshot_date_str,
    silver_clickstream_dir,
    silver_attributes_dir,
    silver_financials_dir,
    gold_dir,
    spark=None,   # unused — kept for API compatibility with the DAG call
):
    """
    Join monthly clickstream features with static attributes and financials,
    apply categorical encoding, and save as gold feature store partition.

    Rewritten from PySpark to pandas to avoid TaskResultLost OOM in Docker.
    Output schema and file names are identical to the original implementation.
    """
    partition_name = "gold_feature_store_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath_out   = gold_dir + partition_name

    if os.path.exists(filepath_out):
        print(f"[gold feature store] {snapshot_date_str} already exists, skipping")
        return None

    clickstream_path = (
        silver_clickstream_dir
        + "silver_clickstream_"
        + snapshot_date_str.replace("-", "_")
        + ".parquet"
    )
    if not os.path.exists(clickstream_path):
        print(f"[gold feature store] missing clickstream silver for {snapshot_date_str}, skipping")
        return None

    df = pd.read_parquet(clickstream_path)

    cutoff = pd.Timestamp(snapshot_date_str)

    # Join attributes: most recent record per customer where snapshot_date <= partition date
    attributes_path = silver_attributes_dir + "silver_attributes.parquet"
    if os.path.exists(attributes_path):
        attrs = pd.read_parquet(attributes_path)
        attrs["snapshot_date"] = pd.to_datetime(attrs["snapshot_date"])
        attrs = attrs[attrs["snapshot_date"] <= cutoff]
        attrs = (
            attrs.sort_values("snapshot_date", ascending=False)
                 .drop_duplicates(subset="Customer_ID", keep="first")
                 .drop(columns=["snapshot_date"])
        )
        df = df.merge(attrs, on="Customer_ID", how="left")

    # Join financials: same temporal-safe logic as attributes
    financials_path = silver_financials_dir + "silver_financials.parquet"
    if os.path.exists(financials_path):
        fins = pd.read_parquet(financials_path)
        fins["snapshot_date"] = pd.to_datetime(fins["snapshot_date"])
        fins = fins[fins["snapshot_date"] <= cutoff]
        fins = (
            fins.sort_values("snapshot_date", ascending=False)
                .drop_duplicates(subset="Customer_ID", keep="first")
                .drop(columns=["snapshot_date"])
        )
        df = df.merge(fins, on="Customer_ID", how="left")

    # Categorical encoding (string → integer) for ML compatibility
    df["Occupation"]            = df["Occupation"].map(_OCCUPATION_MAP)
    df["Credit_Mix"]            = df["Credit_Mix"].map(_CREDIT_MIX_MAP)
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].map(_PAYMENT_MIN_MAP)
    df["Payment_Behaviour"]     = df["Payment_Behaviour"].map(_BEHAVIOUR_MAP)

    os.makedirs(gold_dir, exist_ok=True)
    df.to_parquet(filepath_out, index=False)
    print(f"[gold feature store] {snapshot_date_str} row count: {len(df)}, saved to: {filepath_out}")
    return df
