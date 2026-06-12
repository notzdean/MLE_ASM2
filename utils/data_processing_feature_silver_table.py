import os
import pyspark.sql.functions as F
from pyspark.sql.functions import col, nullif, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_clickstream(snapshot_date_str, bronze_dir, silver_dir, spark):
    """Enforce schema on clickstream bronze partition and save as silver parquet."""
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_dir + partition_name
    if not os.path.exists(filepath):
        print(f"[silver clickstream] bronze missing for {snapshot_date_str}, skipping")
        return None

    df = spark.read.csv(filepath, header=True, inferSchema=False)

    for i in range(1, 21):
        df = df.withColumn(f"fe_{i}", col(f"fe_{i}").cast(IntegerType()))

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    count = df.count()
    out_name = "silver_clickstream_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath_out = silver_dir + out_name
    df.write.mode("overwrite").parquet(filepath_out)
    print(f"[silver clickstream] {snapshot_date_str} row count: {count}, saved to: {filepath_out}")
    return df


def process_silver_attributes(bronze_dir, silver_dir, spark):
    """Clean and enforce schema on attributes bronze table; drop PII columns."""
    filepath_out = silver_dir + "silver_attributes.parquet"
    if os.path.exists(filepath_out):
        print(f"[silver attributes] already exists, skipping")
        return

    df = spark.read.csv(bronze_dir + "bronze_attributes.csv", header=True, inferSchema=False)

    # Drop PII — Name and SSN are never used as ML features
    df = df.drop("Name", "SSN")

    # Age: strip trailing underscores and any non-digit chars, cast to int,
    # then null out biologically impossible values (under 18 or over 100)
    df = df.withColumn("Age", F.regexp_replace(col("Age"), "[^0-9]", ""))
    df = df.withColumn("Age", col("Age").cast(IntegerType()))
    df = df.withColumn("Age",
        F.when((col("Age") >= 18) & (col("Age") <= 100), col("Age")).otherwise(None))

    # Occupation: replace the '______' unknown sentinel with null
    df = df.withColumn("Occupation",
        F.when(col("Occupation") == "_______", None)
         .otherwise(col("Occupation")).cast(StringType()))

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    count = df.count()
    df.write.mode("overwrite").parquet(filepath_out)
    print(f"[silver attributes] saved {count} rows to {filepath_out}")
    return df


def process_silver_financials(bronze_dir, silver_dir, spark):
    """Clean all dirty financial columns and enforce schema."""
    filepath_out = silver_dir + "silver_financials.parquet"
    if os.path.exists(filepath_out):
        print(f"[silver financials] already exists, skipping")
        return

    df = spark.read.csv(bronze_dir + "bronze_financials.csv", header=True, inferSchema=False)

    # Annual_Income: trailing underscore corruption (e.g. "52312.68_") → strip, cast, null if ≤ 0
    df = df.withColumn("Annual_Income",
        F.regexp_replace(col("Annual_Income"), "_", ""))
    df = df.withColumn("Annual_Income", col("Annual_Income").cast(FloatType()))
    df = df.withColumn("Annual_Income",
        F.when(col("Annual_Income") > 0, col("Annual_Income")).otherwise(None))

    # Monthly_Inhand_Salary: already numeric, just cast
    df = df.withColumn("Monthly_Inhand_Salary", col("Monthly_Inhand_Salary").cast(FloatType()))

    # Num_Bank_Accounts: null if negative or implausibly large (> 20)
    df = df.withColumn("Num_Bank_Accounts", col("Num_Bank_Accounts").cast(IntegerType()))
    df = df.withColumn("Num_Bank_Accounts",
        F.when((col("Num_Bank_Accounts") >= 0) & (col("Num_Bank_Accounts") <= 20),
               col("Num_Bank_Accounts")).otherwise(None))

    # Num_Credit_Card: null if negative or implausibly large (> 50)
    df = df.withColumn("Num_Credit_Card", col("Num_Credit_Card").cast(IntegerType()))
    df = df.withColumn("Num_Credit_Card",
        F.when((col("Num_Credit_Card") >= 0) & (col("Num_Credit_Card") <= 50),
               col("Num_Credit_Card")).otherwise(None))

    # Interest_Rate: null if > 100 (percentage must be ≤ 100)
    df = df.withColumn("Interest_Rate", col("Interest_Rate").cast(IntegerType()))
    df = df.withColumn("Interest_Rate",
        F.when((col("Interest_Rate") >= 0) & (col("Interest_Rate") <= 100),
               col("Interest_Rate")).otherwise(None))

    # Num_of_Loan: trailing underscore corruption (e.g. "1_") → strip, cast, null if out of [0, 50]
    df = df.withColumn("Num_of_Loan",
        F.regexp_replace(col("Num_of_Loan"), "_", ""))
    df = df.withColumn("Num_of_Loan", col("Num_of_Loan").cast(IntegerType()))
    df = df.withColumn("Num_of_Loan",
        F.when((col("Num_of_Loan") >= 0) & (col("Num_of_Loan") <= 50),
               col("Num_of_Loan")).otherwise(None))

    # Type_of_Loan: convert multi-value string to a count of loan types; null → 0
    df = df.withColumn("num_loan_types",
        F.when(col("Type_of_Loan").isNull(), F.lit(0))
         .otherwise(F.size(F.split(col("Type_of_Loan"), ","))))
    df = df.drop("Type_of_Loan")

    # Delay_from_due_date: null if negative (cannot have negative delay)
    df = df.withColumn("Delay_from_due_date", col("Delay_from_due_date").cast(IntegerType()))
    df = df.withColumn("Delay_from_due_date",
        F.when(col("Delay_from_due_date") >= 0, col("Delay_from_due_date")).otherwise(None))

    # Num_of_Delayed_Payment: trailing underscore corruption → strip, cast, null if out of [0, 100]
    df = df.withColumn("Num_of_Delayed_Payment",
        F.regexp_replace(col("Num_of_Delayed_Payment"), "_", ""))
    df = df.withColumn("Num_of_Delayed_Payment", col("Num_of_Delayed_Payment").cast(IntegerType()))
    df = df.withColumn("Num_of_Delayed_Payment",
        F.when((col("Num_of_Delayed_Payment") >= 0) & (col("Num_of_Delayed_Payment") <= 100),
               col("Num_of_Delayed_Payment")).otherwise(None))

    # Changed_Credit_Limit: pure "_" sentinel → strip to empty string which casts to null
    df = df.withColumn("Changed_Credit_Limit",
        F.regexp_replace(col("Changed_Credit_Limit"), "_", ""))
    df = df.withColumn("Changed_Credit_Limit", F.when(col("Changed_Credit_Limit") == "", None).otherwise(col("Changed_Credit_Limit")).cast(FloatType()))

    # Num_Credit_Inquiries: stored as "11.0" in CSV (float64 in source), so cast via float first —
    # Spark returns null when casting "11.0" directly to IntegerType
    df = df.withColumn("Num_Credit_Inquiries",
        col("Num_Credit_Inquiries").cast(FloatType()).cast(IntegerType()))
    df = df.withColumn("Num_Credit_Inquiries",
        F.when((col("Num_Credit_Inquiries") >= 0) & (col("Num_Credit_Inquiries") <= 100),
               col("Num_Credit_Inquiries")).otherwise(None))

    # Credit_Mix: "_" is an unknown sentinel → replace with null
    df = df.withColumn("Credit_Mix",
        F.when(col("Credit_Mix") == "_", None)
         .otherwise(col("Credit_Mix")).cast(StringType()))

    # Outstanding_Debt: trailing underscore corruption → strip, cast
    df = df.withColumn("Outstanding_Debt",
        F.regexp_replace(col("Outstanding_Debt"), "_", ""))
    df = df.withColumn("Outstanding_Debt", col("Outstanding_Debt").cast(FloatType()))

    # Credit_Utilization_Ratio: already float, just cast
    df = df.withColumn("Credit_Utilization_Ratio",
        col("Credit_Utilization_Ratio").cast(FloatType()))

    # Credit_History_Age: "X Years and Y Months" → total months as integer
    df = df.withColumn("credit_history_months",
        F.regexp_extract(col("Credit_History_Age"), r"(\d+) Years", 1).cast(IntegerType()) * 12
        + F.regexp_extract(col("Credit_History_Age"), r"(\d+) Months", 1).cast(IntegerType()))
    df = df.drop("Credit_History_Age")

    # Payment_of_Min_Amount: "NM" (not mentioned) → null
    df = df.withColumn("Payment_of_Min_Amount",
        F.when(col("Payment_of_Min_Amount") == "NM", None)
         .otherwise(col("Payment_of_Min_Amount")).cast(StringType()))

    # Total_EMI_per_month: already float, just cast
    df = df.withColumn("Total_EMI_per_month", col("Total_EMI_per_month").cast(FloatType()))

    # Amount_invested_monthly: double-underscore corruption (e.g. "__10000__") → strip underscores
    df = df.withColumn("Amount_invested_monthly",
        F.regexp_replace(col("Amount_invested_monthly"), "_", ""))
    df = df.withColumn("Amount_invested_monthly",
        col("Amount_invested_monthly").cast(FloatType()))

    # Payment_Behaviour: "!@9#%8" is a garbage sentinel → keep only known valid categories
    valid_behaviours = [
        "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments",
    ]
    df = df.withColumn("Payment_Behaviour",
        F.when(col("Payment_Behaviour").isin(valid_behaviours), col("Payment_Behaviour"))
         .otherwise(None).cast(StringType()))

    # Monthly_Balance: double-underscore corruption (e.g. "__-333...333__") → strip underscores,
    # cast, then null out astronomically extreme values (clearly data entry errors)
    df = df.withColumn("Monthly_Balance",
        F.regexp_replace(col("Monthly_Balance"), "_", ""))
    df = df.withColumn("Monthly_Balance", col("Monthly_Balance").cast(FloatType()))
    df = df.withColumn("Monthly_Balance",
        F.when(col("Monthly_Balance") > -1_000_000, col("Monthly_Balance")).otherwise(None))

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    count = df.count()
    df.write.mode("overwrite").parquet(filepath_out)
    print(f"[silver financials] saved {count} rows to {filepath_out}")
    return df
