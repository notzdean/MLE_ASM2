"""
CS611 Assignment 2 — End-to-End ML Pipeline DAG

Schedule: monthly (1st of each month), backfillable from 2023-01-01.
Task chain: data_pipeline >> model_training >> model_inference >> model_monitoring

Logical date (ds) = snapshot month being processed (e.g. 2023-01-01).
"""

import os
import sys

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

# Make utils importable from /opt/airflow/utils
sys.path.insert(0, os.environ.get("AIRFLOW_HOME", "/opt/airflow"))

# ---------------------------------------------------------------------------
# Paths — all relative to AIRFLOW_HOME so they work inside Docker
# ---------------------------------------------------------------------------
BASE = os.environ.get("AIRFLOW_HOME", "/opt/airflow")
DATA_DIR                    = os.path.join(BASE, "data")
BRONZE_LMS_DIR              = os.path.join(BASE, "datamart", "bronze", "lms")
SILVER_LOAN_DIR             = os.path.join(BASE, "datamart", "silver", "loan_daily")
GOLD_LABEL_DIR              = os.path.join(BASE, "datamart", "gold", "label_store")
BRONZE_CLICKSTREAM_DIR      = os.path.join(BASE, "datamart", "bronze", "clickstream")
BRONZE_ATTRIBUTES_DIR       = os.path.join(BASE, "datamart", "bronze", "attributes")
BRONZE_FINANCIALS_DIR       = os.path.join(BASE, "datamart", "bronze", "financials")
SILVER_CLICKSTREAM_DIR      = os.path.join(BASE, "datamart", "silver", "clickstream")
SILVER_ATTRIBUTES_DIR       = os.path.join(BASE, "datamart", "silver", "attributes")
SILVER_FINANCIALS_DIR       = os.path.join(BASE, "datamart", "silver", "financials")
GOLD_FEATURE_DIR            = os.path.join(BASE, "datamart", "gold", "feature_store")
PREDICTIONS_DIR             = os.path.join(BASE, "datamart", "gold", "predictions")
MONITORING_DIR              = os.path.join(BASE, "datamart", "gold", "monitoring")
MODEL_STORE_DIR             = os.path.join(BASE, "model_store")


def _make_dirs():
    for d in [
        BRONZE_LMS_DIR, SILVER_LOAN_DIR, GOLD_LABEL_DIR,
        BRONZE_CLICKSTREAM_DIR, BRONZE_ATTRIBUTES_DIR, BRONZE_FINANCIALS_DIR,
        SILVER_CLICKSTREAM_DIR, SILVER_ATTRIBUTES_DIR, SILVER_FINANCIALS_DIR,
        GOLD_FEATURE_DIR, PREDICTIONS_DIR, MONITORING_DIR, MODEL_STORE_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


def _create_spark():
    """Create a local SparkSession suitable for Docker."""
    import pyspark
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("cs611_ml_pipeline")
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ---------------------------------------------------------------------------
# Task 1: Data pipeline (bronze → silver → gold for labels + features)
# ---------------------------------------------------------------------------
def run_data_pipeline(ds: str, **kwargs):
    _make_dirs()
    spark = _create_spark()

    import utils.data_processing_bronze_table           as bronze_lms
    import utils.data_processing_silver_table           as silver_lms
    import utils.data_processing_gold_table             as gold_label
    import utils.data_processing_feature_bronze_table   as bronze_feat
    import utils.data_processing_feature_silver_table   as silver_feat
    import utils.data_processing_feature_gold_table     as gold_feat

    # --- Label store pipeline ---
    bronze_lms.process_bronze_table(ds, BRONZE_LMS_DIR, spark)
    silver_lms.process_silver_table(ds, BRONZE_LMS_DIR, SILVER_LOAN_DIR, spark)
    gold_label.process_labels_gold_table(
        ds, SILVER_LOAN_DIR, GOLD_LABEL_DIR, spark, dpd=30, mob=6
    )

    # --- Feature store pipeline ---
    # Attributes and financials are static reference tables — ingest once
    bronze_feat.process_bronze_attributes(BRONZE_ATTRIBUTES_DIR, spark)
    bronze_feat.process_bronze_financials(BRONZE_FINANCIALS_DIR, spark)
    # Clickstream is monthly
    bronze_feat.process_bronze_clickstream(ds, BRONZE_CLICKSTREAM_DIR, spark)

    silver_feat.process_silver_attributes(BRONZE_ATTRIBUTES_DIR, SILVER_ATTRIBUTES_DIR, spark)
    silver_feat.process_silver_financials(BRONZE_FINANCIALS_DIR, SILVER_FINANCIALS_DIR, spark)
    silver_feat.process_silver_clickstream(ds, BRONZE_CLICKSTREAM_DIR, SILVER_CLICKSTREAM_DIR, spark)

    gold_feat.process_features_gold_table(
        ds,
        SILVER_CLICKSTREAM_DIR,
        SILVER_ATTRIBUTES_DIR,
        SILVER_FINANCIALS_DIR,
        GOLD_FEATURE_DIR,
        spark,
    )

    spark.stop()
    print(f"[data_pipeline] completed for {ds}")


# ---------------------------------------------------------------------------
# Task 2: Model training
# ---------------------------------------------------------------------------
def run_model_training(ds: str, **kwargs):
    _make_dirs()
    from utils.model_training import train_models
    train_models(
        gold_feature_store_dir=GOLD_FEATURE_DIR,
        gold_label_store_dir=GOLD_LABEL_DIR,
        model_store_dir=MODEL_STORE_DIR,
        train_end_date="2024-03-01",   # temporal cut: train ≤ Mar 2024, test Apr–Jun 2024
    )


# ---------------------------------------------------------------------------
# Task 3: Model inference
# ---------------------------------------------------------------------------
def run_model_inference(ds: str, **kwargs):
    _make_dirs()
    from utils.model_prediction import run_inference
    run_inference(
        snapshot_date_str=ds,
        gold_feature_store_dir=GOLD_FEATURE_DIR,
        predictions_dir=PREDICTIONS_DIR,
        model_store_dir=MODEL_STORE_DIR,
    )


# ---------------------------------------------------------------------------
# Task 4: Model monitoring + visualisation
# ---------------------------------------------------------------------------
def run_model_monitoring(ds: str, **kwargs):
    _make_dirs()
    from utils.model_monitoring import run_monitoring
    run_monitoring(
        snapshot_date_str=ds,
        predictions_dir=PREDICTIONS_DIR,
        gold_label_store_dir=GOLD_LABEL_DIR,
        gold_feature_store_dir=GOLD_FEATURE_DIR,
        monitoring_dir=MONITORING_DIR,
        model_store_dir=MODEL_STORE_DIR,
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="ml_pipeline_dag",
    description="CS611 A2 — loan default ML pipeline: data → train → infer → monitor",
    start_date=datetime(2023, 1, 1),
    schedule_interval="@monthly",
    catchup=True,          # enables backfill across Jan 2023 – Dec 2024
    max_active_runs=1,     # run one month at a time to avoid Spark conflicts
    tags=["cs611", "ml", "loan-default"],
) as dag:

    data_pipeline_task = PythonOperator(
        task_id="data_pipeline",
        python_callable=run_data_pipeline,
        op_kwargs={"ds": "{{ ds }}"},
    )

    model_training_task = PythonOperator(
        task_id="model_training",
        python_callable=run_model_training,
        op_kwargs={"ds": "{{ ds }}"},
    )

    model_inference_task = PythonOperator(
        task_id="model_inference",
        python_callable=run_model_inference,
        op_kwargs={"ds": "{{ ds }}"},
    )

    model_monitoring_task = PythonOperator(
        task_id="model_monitoring",
        python_callable=run_model_monitoring,
        op_kwargs={"ds": "{{ ds }}"},
    )

    data_pipeline_task >> model_training_task >> model_inference_task >> model_monitoring_task
