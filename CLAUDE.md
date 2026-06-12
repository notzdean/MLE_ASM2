# CS611 Assignment 2 — Machine Learning Pipelines

## Project Goal
Build an end-to-end ML pipeline that trains a **loan default prediction** model, serves batch inference, and monitors model performance + stability across time — all orchestrated via **Apache Airflow** running in Docker.

Due: **25 Jun 2026** | Grader runs `docker-compose build && docker-compose up`, then triggers the Airflow DAG.

---

## Assessment Criteria (10 marks total)

| Marks | What the grader checks |
|-------|------------------------|
| 2 | `docker-compose up` shows Airflow UI at localhost:8080 |
| 3 | DAG runs end-to-end: model artifacts saved, predictions written to gold table, monitoring gold table written |
| 3 | Presentation deck covers technical design + model monitoring **visualisation screenshots** |
| 2 | Deck is polished / corporate-standard |

---

## What Already Exists (from A1 reference)

The `A1_reference/` folder is our own Assignment 1 work. Copy all files into the root before adding new code.

### Data pipeline (bronze → silver → gold) — already working
- `utils/data_processing_bronze_table.py` — ingest raw LMS CSV, partition by snapshot_date
- `utils/data_processing_silver_table.py` — clean loan daily table, compute mob/dpd
- `utils/data_processing_gold_table.py` — build label store (label = 1 if dpd >= 30 at mob 6)
- `utils/data_processing_feature_bronze_table.py` — ingest clickstream (monthly), attributes, financials
- `utils/data_processing_feature_silver_table.py` — clean all three feature sources
- `utils/data_processing_feature_gold_table.py` — join + label-encode → ML-ready feature store

### Gold feature store output columns (ML-ready)
**Clickstream (20 behavioral):** `fe_1` … `fe_20` (integers)

**Attributes (2):** `Age` (int, 18–100), `Occupation` (label-encoded 0–14)

**Financials (18):**
- `Annual_Income`, `Monthly_Inhand_Salary`, `Total_EMI_per_month`, `Amount_invested_monthly`, `Monthly_Balance` (floats)
- `Num_Bank_Accounts` (0–20), `Num_Credit_Card` (0–50), `Interest_Rate` (0–100), `Num_of_Loan` (0–50), `num_loan_types`, `Delay_from_due_date`, `Num_of_Delayed_Payment`, `Num_Credit_Inquiries`, `Changed_Credit_Limit`, `Outstanding_Debt`, `Credit_Utilization_Ratio`, `credit_history_months` (ints/floats)
- `Credit_Mix` (0=Bad, 1=Standard, 2=Good), `Payment_of_Min_Amount` (0=No, 1=Yes), `Payment_Behaviour` (0–5 ordered by spend×value)

**Label:** `label` (0/1) from label store, joined on `Customer_ID` + `snapshot_date`

---

## Design Decisions

### Feature sourcing choice
Use **our own A1 pipeline** (not classmates') because it is the most complete implementation with full silver-layer cleaning (regex corruption fixes, range validation, PII drop). We incorporate ideas from sample assignments but not their raw data:
- **From Lam Nguyen (Sample 3):** Inspiration for engineered ratio features — add `debt_to_income`, `emi_to_salary`, `credit_utilization_ratio` at training time (not in gold table, to keep the feature store generic)
- **From Jennifer Poernomo (Sample 1):** Confirm that `credit_history_months` (already parsed in our silver) and `num_loan_types` are valuable features worth keeping

### Data leakage prevention (critical)
- **Label at MOB 6:** The label is `dpd >= 30` at the customer's 6th month on book. The feature snapshot_date is the loan origination month. Label is measured 6 months later — never use any data from those 6 months as features.
- **Temporal train/test split (no shuffling):** Train on loans originated **Jan 2023 – Jun 2024**, test on **Jul 2024 – Dec 2024**. The join is `feature.snapshot_date == label.snapshot_date`.
- **Attributes/financials join:** Always filter `snapshot_date <= feature_partition_date` then take the most recent record (already implemented with `row_number()` in gold table).
- **Never** include any loan outcome columns (`dpd`, `mob`, `loan_status`) in the feature set.

### Model choices
Train three candidate models on the training split, pick best by **AUC on test set**:
1. Logistic Regression (baseline, interpretable)
2. Random Forest (ensemble, handles non-linearity)
3. XGBoost (gradient boosting, typically strongest)

Evaluation metrics per model: AUC, Gini (= 2×AUC − 1), KS statistic.

Save the best model artifact as `model_store/champion_model.pkl` plus a `model_store/model_metadata.json` with training date, metrics, hyperparameters, features used.

### Monitoring metrics
**Performance monitoring** (requires ground truth — computed for months where 6-month label is available):
- AUC, Gini, KS per monthly prediction cohort

**Stability monitoring** (no ground truth needed — computed every inference run):
- PSI (Population Stability Index) for the prediction score distribution
  - PSI < 0.10 → stable, 0.10–0.25 → slight drift, > 0.25 → retrain trigger
- PSI for top 5 most important features

### Model governance SOP
- Retrain trigger: PSI > 0.25 on score distribution OR AUC drops > 5 percentage points from baseline
- Retraining cadence: Monthly check, retrain quarterly or when trigger fires
- Deployment: Batch scoring (no real-time API needed for this use case — offline loan decisioning)
- Champion/Challenger: Keep the previous champion in `model_store/` with timestamp until new champion validated

---

## File Structure to Build

```
Assignment_2/
├── dags/
│   └── ml_pipeline_dag.py          ← NEW: Airflow DAG (main orchestrator)
├── utils/
│   ├── data_processing_bronze_table.py         ← copied from A1
│   ├── data_processing_silver_table.py         ← copied from A1
│   ├── data_processing_gold_table.py           ← copied from A1
│   ├── data_processing_feature_bronze_table.py ← copied from A1
│   ├── data_processing_feature_silver_table.py ← copied from A1
│   ├── data_processing_feature_gold_table.py   ← copied from A1
│   ├── model_training.py           ← NEW
│   ├── model_prediction.py         ← NEW
│   └── model_monitoring.py         ← NEW
├── data/
│   ├── feature_clickstream.csv
│   ├── features_attributes.csv
│   ├── features_financials.csv
│   └── lms_loan_daily.csv
├── datamart/
│   ├── bronze/
│   ├── silver/
│   └── gold/
│       ├── label_store/            ← monthly parquet partitions
│       ├── feature_store/          ← monthly parquet partitions
│       ├── predictions/            ← NEW: monthly inference output
│       └── monitoring/             ← NEW: performance + stability results + plots
├── model_store/                    ← NEW: champion_model.pkl, model_metadata.json
├── Dockerfile                      ← UPDATE: add Airflow
├── docker-compose.yaml             ← REPLACE: Airflow stack (was Jupyter-only)
├── requirements.txt                ← UPDATE: add airflow, sklearn, xgboost, etc.
└── Readme.txt                      ← 1 line: GitHub repo link
```

---

## Airflow DAG Design

**DAG id:** `ml_pipeline_dag`
**Schedule:** `@monthly` (1st of each month, backfillable)
**Start date:** 2023-01-01

### Task order (linear chain)
```
data_pipeline_task
    >> model_training_task
    >> model_inference_task
    >> model_monitoring_task
    >> model_visualisation_task
```

### data_pipeline_task
Wraps all A1 data processing steps as a single `PythonOperator`. Runs for `{{ ds }}` (the DAG's logical date = snapshot month).

Steps (in order):
1. Bronze LMS → Silver loan daily → Gold label store (for `{{ ds }}`)
2. Bronze clickstream (for `{{ ds }}`), Bronze attributes (once), Bronze financials (once)
3. Silver for all three feature sources
4. Gold feature store (for `{{ ds }}`)

### model_training_task
Triggered only when enough data exists (skip if `{{ ds }}` < 2024-01-01 — need at least 12 months of labeled data).

Steps in `utils/model_training.py`:
1. Load all gold feature store partitions + label store partitions
2. Join on `Customer_ID` + `snapshot_date`
3. Drop leakage columns (`dpd`, `mob`, `loan_status`, `snapshot_date`)
4. Temporal split: train ≤ 2024-06-01, test > 2024-06-01
5. Impute nulls (median for numeric, mode for categorical)
6. Train Logistic Regression, Random Forest, XGBoost
7. Evaluate all three → pick best AUC on test set
8. Save `model_store/champion_model.pkl` + `model_store/model_metadata.json`

### model_inference_task
Steps in `utils/model_prediction.py`:
1. Load `champion_model.pkl` from `model_store/`
2. Load gold feature store for current `{{ ds }}`
3. Predict probability scores
4. Save `datamart/gold/predictions/predictions_{{ ds }}.parquet` with columns: `Customer_ID`, `loan_id`, `snapshot_date`, `score`, `predicted_label` (threshold 0.5)

### model_monitoring_task
Steps in `utils/model_monitoring.py`:
1. Load all predictions gold tables (all months available)
2. **Stability:** Compute PSI on score distribution vs. training distribution (baseline from model_metadata.json)
3. **Performance:** For months where labels are available (snapshot_date + 6 months ≤ today), join predictions with labels, compute AUC/Gini/KS
4. Save `datamart/gold/monitoring/monitoring_{{ ds }}.parquet` with all metrics

### model_visualisation_task
Steps (inline in DAG or in `utils/model_monitoring.py`):
1. Load all monitoring parquet files
2. Plot AUC / Gini / KS over time → save PNG to `datamart/gold/monitoring/`
3. Plot PSI over time → save PNG
4. Plot score distribution per month (overlaid) → save PNG

---

## Docker Setup Notes

The `docker-compose.yaml` must:
- Start Airflow webserver on port 8080 (`localhost:8080`)
- Use LocalExecutor with Postgres as metadata DB
- Mount `./dags` → `/opt/airflow/dags`
- Mount `./utils` → `/opt/airflow/utils` (or copy into image)
- Mount `./data` → `/opt/airflow/data`
- Mount `./datamart` → `/opt/airflow/datamart`
- Mount `./model_store` → `/opt/airflow/model_store`
- Set `AIRFLOW__CORE__LOAD_EXAMPLES=false`

The `Dockerfile` must install: `apache-airflow`, `pyspark`, `scikit-learn`, `xgboost`, `pandas`, `matplotlib`, `scipy`, `pyarrow` (for parquet), and `default-jdk-headless` (for PySpark).

**PySpark is used for the data pipeline tasks** (bronze → silver → gold) exactly as in A1 — the professor designed the course around PySpark and A2 builds on that work. Java is included in the Docker image. **pandas + sklearn** are used for the new ML tasks (model training, inference, monitoring) since those are pure ML operations, not big-data processing.

---

## Implementation Order

1. Set up Docker + Airflow (`docker-compose.yaml`, `Dockerfile`, `requirements.txt`) — get the 2 marks first
2. Create the Airflow DAG skeleton (`dags/ml_pipeline_dag.py`) with all 5 tasks stubbed out
3. Wrap the A1 PySpark data pipeline functions as Airflow PythonOperators (one date per run)
4. Write `utils/model_training.py`
5. Write `utils/model_prediction.py`
6. Write `utils/model_monitoring.py` with visualisations
7. Wire all tasks into the DAG and do a backfill test
8. Capture monitoring screenshots for the presentation deck

---

## Submission Checklist
- [ ] `docker-compose build` succeeds
- [ ] `docker-compose up` shows Airflow at localhost:8080
- [ ] DAG appears in Airflow UI, can be triggered manually
- [ ] Backfill runs: model artifacts appear in `model_store/`, predictions in `datamart/gold/predictions/`, monitoring in `datamart/gold/monitoring/`
- [ ] Monitoring plots saved as PNGs (needed for the deck)
- [ ] `Readme.txt` contains GitHub repo link
- [ ] Zip contains: dags/, utils/, data/, datamart/, model_store/, Dockerfile, docker-compose.yaml, requirements.txt, Readme.txt
- [ ] PDF presentation deck uploaded separately (≤ 10 slides)
