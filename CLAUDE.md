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
- **Temporal train/test/OOT split (no shuffling):** Train ≤ 2023-12-01, Test Jan–Mar 2024, OOT Apr–Jun 2024. The join is `feature.snapshot_date == label.snapshot_date`.
  - Note: label store partition is named by **measurement date** (the month the label is observed), not origination date. `gold_label_store_2023_07_01.parquet` contains loans **originated Jan 2023** at MOB 6 as of Jul 2023. `_load_labeled_dataset` subtracts 6 months from label partition date to find the matching feature origination date.
  - First valid labeled training data = **Jul 2023** run (earliest label available, pointing to Jan 2023 features).
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
- Champion/Challenger shadow mode: `challenger_model.pkl` runs in parallel for several months; auto-promotes to champion when it wins `CONSECUTIVE_WINS_REQUIRED` months in a row on labeled cohorts
- Bootstrap: if no champion exists yet, `check_retrain_needed` always fires → `train_challenger` → auto-promotes immediately (no labeled comparison needed)

---

## File Structure

```
Assignment_2/
├── dags/
│   └── ml_pipeline_dag.py          ← Airflow DAG (main orchestrator)
├── utils/
│   ├── data_processing_bronze_table.py         ← from A1 (PySpark)
│   ├── data_processing_silver_table.py         ← from A1 (PySpark)
│   ├── data_processing_gold_table.py           ← from A1 (PySpark)
│   ├── data_processing_feature_bronze_table.py ← REWRITTEN to use pandas (was PySpark — OOM in Docker)
│   ├── data_processing_feature_silver_table.py ← from A1 (PySpark)
│   ├── data_processing_feature_gold_table.py   ← from A1 (PySpark)
│   ├── data_quality.py             ← NEW: QC gate between data pipeline and model tasks
│   ├── model_training.py           ← NEW
│   ├── model_prediction.py         ← NEW
│   └── model_monitoring.py         ← NEW (includes visualisation)
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
│       ├── predictions/            ← monthly inference output
│       └── monitoring/             ← performance + stability parquets + PNG plots
├── model_store/                    ← champion_model.pkl, challenger_model.pkl, model_metadata.json
├── Dockerfile
├── docker-compose.yaml             ← Airflow stack (LocalExecutor + Postgres)
├── requirements.txt
└── Readme.txt                      ← 1 line: GitHub repo link
```

---

## Airflow DAG Design

**DAG id:** `ml_pipeline_dag`
**Schedule:** `@monthly` (1st of each month, backfillable)
**Start date:** 2023-01-01
**max_active_runs:** 1 (prevents parallel backfill runs; months process sequentially)

### Actual task graph (implemented)
```
data_pipeline
    >> data_quality
    >> check_retrain_needed  (BranchPythonOperator)
         ↓ "train_challenger"        ↓ "skip_training"
    train_challenger          skip_training (EmptyOperator)
         ↓                               ↓
    model_inference  ←─────────────────┘   (TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    >> model_monitoring          (includes visualisation + auto-promotion)
```

### data_pipeline task
Wraps all A1 data processing steps in one `PythonOperator`. All steps are idempotent (check file exists, skip if so) so retries are safe and fast.

Steps (sequential):
1. Bronze LMS → Silver loan daily → Gold label store (for `{{ ds }}`)
2. Bronze clickstream (for `{{ ds }}`), Bronze attributes (once), Bronze financials (once)
3. Silver for all three feature sources
4. Gold feature store (for `{{ ds }}`)

**Important:** Bronze feature tables use **pandas** not PySpark. The original A1 PySpark implementation caused `TaskResultLost` OOM errors inside Docker after processing multiple months. Since bronze feature steps are simple CSV copy/filter operations with no distributed processing need, pure pandas is correct here.

### data_quality task
Validates gold feature store and label store partitions for `{{ ds }}`. Raises `ValueError` on critical failures (missing file, empty partition, missing required columns) which marks all downstream tasks as `upstream_failed`.

Null rate checks are **warnings only** — attributes/financials columns are expected to be 95%+ null in the gold feature store because clickstream covers all 8,974 visitors while only 530 are loan customers. Model training inner-joins with labels so only loan customers reach training.

### check_retrain_needed task (BranchPythonOperator)
Returns `"train_challenger"` or `"skip_training"`. Triggers retraining when ANY of:
1. No champion exists yet (bootstrap — first ever run)
2. `challenger_model.pkl` already exists (challenger in shadow mode — keep it fresh)
3. Score PSI > 0.25 in latest monitoring
4. Champion AUC dropped > 5pp from its baseline

### train_challenger task
Steps in `utils/model_training.py`:
1. Load all gold feature store partitions + label store partitions
2. Join on `Customer_ID` + `snapshot_date` (inner join — only loan customers have labels)
3. Drop leakage columns (`dpd`, `mob`, `loan_status`, `snapshot_date`)
4. Temporal split: Train ≤ 2023-12-01 | Test Jan–Mar 2024 | OOT Apr–Jun 2024
5. Impute nulls (median for numeric, mode for categorical)
6. Train Logistic Regression, Random Forest, XGBoost
7. Evaluate all three → pick best OOT AUC (falls back to test AUC if OOT unavailable)
8. Save as `model_store/challenger_model.pkl` + `model_store/model_metadata.json`
9. On bootstrap (no champion exists), immediately copy challenger → champion

**sklearn 1.6 note:** `make_scorer` no longer accepts `needs_proba=True`; use `response_method="predict_proba"` instead.

### model_inference task
Steps in `utils/model_prediction.py`:
1. Load `champion_model.pkl` (always scores) + `challenger_model.pkl` if it exists (shadow scores)
2. Load gold feature store for current `{{ ds }}`
3. Predict probability scores for both models
4. Save `datamart/gold/predictions/predictions_{{ ds }}.parquet` with columns: `Customer_ID`, `snapshot_date`, `score`, `predicted_label` (threshold 0.5), `challenger_score`, `challenger_predicted_label`

### model_monitoring task
Steps in `utils/model_monitoring.py` (visualisation is inline, no separate task):
1. Load all prediction parquets
2. **Stability:** PSI on score distribution vs. training baseline (from model_metadata.json)
3. **Performance:** For months where labels are available (snapshot_date + 6 months ≤ today), compute AUC/Gini/KS for champion and challenger
4. **Auto-promotion:** If challenger beats champion for `CONSECUTIVE_WINS_REQUIRED` months, copy challenger → champion and delete challenger
5. Save `datamart/gold/monitoring/monitoring_{{ ds }}.parquet`
6. Plot AUC/Gini/KS over time → PNG, PSI over time → PNG, score distributions overlaid → PNG

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

1. ✅ Set up Docker + Airflow (`docker-compose.yaml`, `Dockerfile`, `requirements.txt`)
2. ✅ Create the Airflow DAG (`dags/ml_pipeline_dag.py`) with all tasks
3. ✅ Wrap A1 PySpark data pipeline as Airflow PythonOperator + add `data_quality` gate
4. ✅ Write `utils/model_training.py` (challenger/champion governance, temporal split)
5. ✅ Write `utils/model_prediction.py` (champion + shadow challenger scoring)
6. ✅ Write `utils/model_monitoring.py` (PSI, AUC/Gini/KS, auto-promotion, PNG plots)
7. ✅ Backfill running Jan 2023 → Dec 2024
8. ⬜ Capture monitoring screenshots for the presentation deck (after backfill completes)
9. ⬜ Build presentation deck (≤ 10 slides)

---

## Known Issues and Runtime Fixes

### Spark OOM in bronze feature processing
`toPandas()` on large DataFrames inside Docker causes `TaskResultLost` from the block manager. Even with `spark.driver.memory=4g`, the driver exhausts memory after processing multiple months in a single Airflow task. **Fix:** `data_processing_feature_bronze_table.py` now uses pandas for all three bronze functions — they are simple CSV copy/filter operations with no distributed processing need.

### Path separator convention
A1 code uses `dir + filename` string concatenation (no `os.path.join`), which requires a trailing `/` on directory paths. All `*_DIR` constants in `ml_pipeline_dag.py` are defined with `+ "/"` appended. Both `dir + filename` and `os.path.join(dir, filename)` then produce the same correct path.

### sklearn 1.6 API change
`make_scorer(roc_auc_score, needs_proba=True)` raises `TypeError` in sklearn 1.6+. Use `response_method="predict_proba"` instead.

### f-string crash with None AUC
`f"OOT AUC={m_oot['auc']:.4f if m_oot['auc'] else 'N/A'}"` — Python parses `:.4f if ...` as the format spec. Pre-compute: `oot_auc_str = f"{m_oot['auc']:.4f}" if m_oot['auc'] is not None else "N/A"`.

### High null rates in gold feature store (expected, not an error)
Clickstream has ~8,974 customers/month; attributes and financials only cover the ~530 loan customers. LEFT JOIN from clickstream produces 95%+ nulls in financial columns. `data_quality.py` treats this as a warning, not a critical failure. Training inner-joins with label store so only loan customers reach the model.

### DAG task ordering with max_active_runs=1
`max_active_runs=1` prevents new DAG runs from starting but does not stop already-running ones. If two runs are simultaneously "running", mark the later one Failed to free the slot, then let the earlier month complete first.

### Fresh restart procedure
```powershell
docker-compose down -v   # removes Postgres volume (full Airflow DB reset)
Remove-Item -Recurse -Force datamart\bronze\*, datamart\silver\*, datamart\gold\* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force model_store\* -ErrorAction SilentlyContinue
docker-compose up -d
# wait ~30s, then toggle DAG ON in UI
```

### DAG vs Lab 5 comparison
Lab 5 (`Lec 7/lab_5/dags/dag.py`) uses one `BashOperator` per bronze/silver/gold step with parallel fanout for independent sources. Our single `data_pipeline` PythonOperator is simpler but equally correct because all steps are idempotent — a retry skips already-completed files. The trade-off is less retry granularity vs a simpler DAG graph. Our DAG is more sophisticated on the ML side (branch logic, shadow mode, auto-promotion) which is what the assignment grades.

---

## Submission Checklist
- [x] `docker-compose build` succeeds
- [x] `docker-compose up` shows Airflow at localhost:8080
- [x] DAG appears in Airflow UI, can be triggered manually
- [ ] Backfill completes: model artifacts in `model_store/`, predictions in `datamart/gold/predictions/`, monitoring in `datamart/gold/monitoring/`
- [ ] Monitoring plots saved as PNGs (needed for the deck)
- [ ] `Readme.txt` contains GitHub repo link
- [ ] Zip contains: dags/, utils/, data/, datamart/, model_store/, Dockerfile, docker-compose.yaml, requirements.txt, Readme.txt
- [ ] PDF presentation deck uploaded separately (≤ 10 slides)
