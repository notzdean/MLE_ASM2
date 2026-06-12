"""
CS611 Assignment 2 — End-to-End ML Pipeline DAG

Schedule: monthly (1st of each month), backfillable from 2023-01-01.

Task chain:
  data_pipeline
    >> data_quality
    >> check_retrain_needed          ← BranchPythonOperator
         ↓ retrain_triggered              ↓ stable
    train_challenger           skip_training (EmptyOperator)
         ↓                               ↓
    model_inference  ←──────────────────┘   (NONE_FAILED_MIN_ONE_SUCCESS)
    >> model_monitoring          ← also handles shadow-mode auto-promotion

Retrain is triggered when ANY of:
  1. No champion model exists yet (first ever run — bootstrap)
  2. Score PSI > 0.25 in latest monitoring (score distribution drifted)
  3. Champion AUC dropped > 5pp from its OOT baseline (performance decay)
  4. Challenger is still in shadow mode (needs a fresh challenger each cycle)

Promotion: monitoring compares challenger vs champion on labeled cohorts.
After CONSECUTIVE_WINS_REQUIRED months of challenger > champion, it auto-promotes.

Logical date (ds) = snapshot month being processed (e.g. 2023-01-01).
"""

import glob
import json
import os
import sys

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

sys.path.insert(0, os.environ.get("AIRFLOW_HOME", "/opt/airflow"))

# ---------------------------------------------------------------------------
# Paths — all relative to AIRFLOW_HOME so they work inside Docker
# ---------------------------------------------------------------------------
BASE = os.environ.get("AIRFLOW_HOME", "/opt/airflow")
# All dir paths end with "/" so both A1-style (dir + filename) and
# os.path.join(dir, filename) produce the same result.
DATA_DIR               = os.path.join(BASE, "data") + "/"
BRONZE_LMS_DIR         = os.path.join(BASE, "datamart", "bronze", "lms") + "/"
SILVER_LOAN_DIR        = os.path.join(BASE, "datamart", "silver", "loan_daily") + "/"
GOLD_LABEL_DIR         = os.path.join(BASE, "datamart", "gold", "label_store") + "/"
BRONZE_CLICKSTREAM_DIR = os.path.join(BASE, "datamart", "bronze", "clickstream") + "/"
BRONZE_ATTRIBUTES_DIR  = os.path.join(BASE, "datamart", "bronze", "attributes") + "/"
BRONZE_FINANCIALS_DIR  = os.path.join(BASE, "datamart", "bronze", "financials") + "/"
SILVER_CLICKSTREAM_DIR = os.path.join(BASE, "datamart", "silver", "clickstream") + "/"
SILVER_ATTRIBUTES_DIR  = os.path.join(BASE, "datamart", "silver", "attributes") + "/"
SILVER_FINANCIALS_DIR  = os.path.join(BASE, "datamart", "silver", "financials") + "/"
GOLD_FEATURE_DIR       = os.path.join(BASE, "datamart", "gold", "feature_store") + "/"
PREDICTIONS_DIR        = os.path.join(BASE, "datamart", "gold", "predictions") + "/"
MONITORING_DIR         = os.path.join(BASE, "datamart", "gold", "monitoring") + "/"
MODEL_STORE_DIR        = os.path.join(BASE, "model_store") + "/"

PSI_RETRAIN_THRESHOLD = 0.25   # matches model_monitoring.py
AUC_DROP_THRESHOLD    = 0.05   # retrain if champion AUC drops more than this


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
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.network.timeout", "600s")
        .config("spark.executor.heartbeatInterval", "60s")
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

    import utils.data_processing_bronze_table         as bronze_lms
    import utils.data_processing_silver_table         as silver_lms
    import utils.data_processing_gold_table           as gold_label
    import utils.data_processing_feature_bronze_table as bronze_feat
    import utils.data_processing_feature_silver_table as silver_feat
    import utils.data_processing_feature_gold_table   as gold_feat

    bronze_lms.process_bronze_table(ds, BRONZE_LMS_DIR, spark)
    silver_lms.process_silver_table(ds, BRONZE_LMS_DIR, SILVER_LOAN_DIR, spark)
    gold_label.process_labels_gold_table(
        ds, SILVER_LOAN_DIR, GOLD_LABEL_DIR, spark, dpd=30, mob=6
    )

    bronze_feat.process_bronze_attributes(BRONZE_ATTRIBUTES_DIR, spark)
    bronze_feat.process_bronze_financials(BRONZE_FINANCIALS_DIR, spark)
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
# Task 2: Data quality gate
# ---------------------------------------------------------------------------
def run_data_quality(ds: str, **kwargs):
    _make_dirs()
    from utils.data_quality import run_data_quality as _dq
    _dq(
        snapshot_date_str=ds,
        gold_feature_store_dir=GOLD_FEATURE_DIR,
        gold_label_store_dir=GOLD_LABEL_DIR,
    )


# ---------------------------------------------------------------------------
# Task 3: Branch — should we train a challenger this month?
# ---------------------------------------------------------------------------
def check_retrain_needed(ds: str, **kwargs) -> str:
    """
    Returns the task_id to execute next:
      "train_challenger"  — a new challenger should be trained
      "skip_training"     — champion is stable, run inference as-is

    Triggers retrain when:
      1. No champion exists (first run)
      2. Challenger already in shadow (keep it fresh each cycle)
      3. Score PSI > 0.25 in latest monitoring
      4. Champion AUC dropped > 5pp from baseline
    """
    import pandas as pd

    champion_path   = os.path.join(MODEL_STORE_DIR, "champion_model.pkl")
    challenger_path = os.path.join(MODEL_STORE_DIR, "challenger_model.pkl")

    # Case 1: bootstrap — no champion yet
    if not os.path.exists(champion_path):
        print("[branch] no champion — triggering initial training")
        return "train_challenger"

    # Case 2: challenger already in shadow — retrain to keep it current
    if os.path.exists(challenger_path):
        print("[branch] challenger in shadow mode — refreshing challenger")
        return "train_challenger"

    # Case 3 + 4: check latest monitoring metrics
    mon_files = sorted(glob.glob(os.path.join(MONITORING_DIR, "monitoring_*.parquet")))
    if not mon_files:
        print("[branch] no monitoring data yet — skipping training")
        return "skip_training"

    latest = pd.read_parquet(mon_files[-1])

    # PSI breach
    if "psi_score" in latest.columns:
        max_psi = latest["psi_score"].max()
        if max_psi > PSI_RETRAIN_THRESHOLD:
            print(f"[branch] PSI breach ({max_psi:.4f} > {PSI_RETRAIN_THRESHOLD}) — triggering retrain")
            return "train_challenger"

    # AUC decay vs baseline
    meta_path = os.path.join(MODEL_STORE_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        model_name   = meta.get("model_name", "")
        model_metrics = meta.get("metrics", {}).get(model_name, {})
        baseline_auc = (
            model_metrics.get("oot", {}).get("auc")
            or model_metrics.get("test", {}).get("auc")
        )
        if baseline_auc:
            current_aucs = latest["auc"].dropna()
            if len(current_aucs) > 0:
                current_auc = float(current_aucs.iloc[-1])
                drop = baseline_auc - current_auc
                if drop > AUC_DROP_THRESHOLD:
                    print(
                        f"[branch] AUC decay ({drop:.4f} > {AUC_DROP_THRESHOLD}) — "
                        f"baseline={baseline_auc:.4f} current={current_auc:.4f}"
                    )
                    return "train_challenger"

    print("[branch] model is stable — skipping training")
    return "skip_training"


# ---------------------------------------------------------------------------
# Task 4a: Train challenger (runs only when check_retrain_needed fires)
# ---------------------------------------------------------------------------
def run_train_challenger(ds: str, **kwargs):
    _make_dirs()
    from utils.model_training import train_challenger
    train_challenger(
        gold_feature_store_dir=GOLD_FEATURE_DIR,
        gold_label_store_dir=GOLD_LABEL_DIR,
        model_store_dir=MODEL_STORE_DIR,
        monitoring_dir=MONITORING_DIR,
    )


# ---------------------------------------------------------------------------
# Task 5: Model inference (champion always scores; challenger scores in shadow)
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
# Task 6: Monitoring + visualisation + auto-promotion
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
    description="CS611 A2 — loan default ML pipeline with shadow-mode challenger governance",
    start_date=datetime(2023, 1, 1),
    schedule_interval="@monthly",
    catchup=True,
    max_active_runs=1,
    tags=["cs611", "ml", "loan-default"],
) as dag:

    data_pipeline_task = PythonOperator(
        task_id="data_pipeline",
        python_callable=run_data_pipeline,
        op_kwargs={"ds": "{{ ds }}"},
    )

    data_quality_task = PythonOperator(
        task_id="data_quality",
        python_callable=run_data_quality,
        op_kwargs={"ds": "{{ ds }}"},
    )

    check_retrain_task = BranchPythonOperator(
        task_id="check_retrain_needed",
        python_callable=check_retrain_needed,
        op_kwargs={"ds": "{{ ds }}"},
    )

    train_challenger_task = PythonOperator(
        task_id="train_challenger",
        python_callable=run_train_challenger,
        op_kwargs={"ds": "{{ ds }}"},
    )

    skip_training_task = EmptyOperator(task_id="skip_training")

    # NONE_FAILED_MIN_ONE_SUCCESS: inference runs whether the branch went to
    # train_challenger (which succeeded) or skip_training (which was selected)
    model_inference_task = PythonOperator(
        task_id="model_inference",
        python_callable=run_model_inference,
        op_kwargs={"ds": "{{ ds }}"},
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    model_monitoring_task = PythonOperator(
        task_id="model_monitoring",
        python_callable=run_model_monitoring,
        op_kwargs={"ds": "{{ ds }}"},
    )

    # Wire the task graph
    data_pipeline_task >> data_quality_task >> check_retrain_task
    check_retrain_task >> [train_challenger_task, skip_training_task]
    train_challenger_task >> model_inference_task
    skip_training_task    >> model_inference_task
    model_inference_task  >> model_monitoring_task
