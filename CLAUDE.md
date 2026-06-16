# CS611 Assignment 2 — Machine Learning Pipelines

## Project Goal

Build an end-to-end ML pipeline that trains a **loan default prediction** model, serves batch
inference, and monitors model performance + stability across time — all orchestrated via
**Apache Airflow** running in Docker.

Due: **25 Jun 2026** | Grader runs `docker-compose build && docker-compose up`, then
triggers the Airflow DAG.

---

## Assessment Criteria (10 marks total)

| Marks | What the grader checks |
|-------|------------------------|
| 2 | `docker-compose up` shows Airflow UI at localhost:8080 |
| 3 | DAG runs end-to-end: ML model artefacts saved, predictions written to gold table, monitoring results written to gold table |
| 3 | Presentation deck covers ML pipeline technical design decisions WITH model monitoring visualisation results |
| 2 | Deck is polished / corporate-standard (slideument format) |

---

## Business Context

Predicting whether a user will default on their loan **at the point of application** (DPD >= 30
at Month-On-Book 6). Batch scoring for offline loan decisioning — no real-time API needed.

Data sources:
- `lms_loan_daily.csv` — loan management system (LMS) daily snapshots
- `feature_clickstream.csv` — monthly user browsing behaviour (8,974 visitors/month)
- `features_attributes.csv` — user demographics
- `features_financials.csv` — user financial profile

---

## Data Leakage Prevention (critical)

The PDF explicitly warns about data leakage. Three controls implemented:

1. **Target leakage**: Never include loan outcome columns (`dpd`, `mob`, `loan_status`) as
   features. Label is derived from `dpd >= 30 at mob 6` — measured 6 months after origination.

2. **Temporal leakage**: Features at origination month (MOB-0) are used to predict default at
   MOB-6. Inference also uses origination-month features so train/inference are aligned.
   Label store partition date = measurement date (origination + 6 months).
   `gold_label_store_2023_07_01.parquet` = Jan 2023 origination cohort, measured Jul 2023.

3. **Train-test contamination**: Temporal 3-way split with no shuffling.
   Train <= 2023-12-01 | Test Jan-Mar 2024 | OOT Apr-Jun 2024.
   OOT cohort (feature dates Apr-Jun 2024, labels Oct-Dec 2024) is genuinely held out.

---

## Feature Engineering

**Gold feature store output (ML-ready, 44 columns):**

- Clickstream (20): `fe_1` ... `fe_20` (monthly behavioural integers)
- Attributes (2): `Age`, `Occupation` (label-encoded 0-14)
- Financials (22): `Annual_Income`, `Monthly_Inhand_Salary`, `Total_EMI_per_month`,
  `Amount_invested_monthly`, `Monthly_Balance`, `Num_Bank_Accounts`, `Num_Credit_Card`,
  `Interest_Rate`, `Num_of_Loan`, `num_loan_types`, `Delay_from_due_date`,
  `Num_of_Delayed_Payment`, `Num_Credit_Inquiries`, `Changed_Credit_Limit`,
  `Outstanding_Debt`, `Credit_Utilization_Ratio`, `credit_history_months`,
  `Payment_of_Min_Amount`, `Credit_Mix`, `Payment_Behaviour`

**Engineered ratio features** (added at training + inference time, not in gold store):
- `debt_to_income` = Outstanding_Debt / Annual_Income
- `emi_to_salary` = Total_EMI_per_month / Monthly_Inhand_Salary
- `debt_per_loan` = Outstanding_Debt / Num_of_Loan
- `free_cash_flow_ratio` = (Salary - EMI) / Salary
- `savings_rate` = Amount_invested_monthly / Salary
- `inquiry_rate` = Num_Credit_Inquiries / credit_history_months
- `clickstream_total`, `clickstream_active_channels` (aggregates of fe_1..fe_20)

---

## Model Training (`utils/model_training.py`)

Three candidates trained, best by OOT AUC selected as challenger:

1. **Logistic Regression** — baseline, interpretable
2. **Random Forest** — ensemble, handles non-linearity
3. **XGBoost** with RandomizedSearchCV (10 iterations, 3-fold CV) — gradient boosting

Preprocessing: median imputation → StandardScaler. Calibration: sigmoid (small test set) or
isotonic (large test set) via `CalibratedClassifierCV`.

**Winner: XGBoost** (best OOT AUC across all runs).

Model artefacts saved to `model_store/`:
- `champion_model.pkl` — production model bundle (pipeline + imputer + scaler + feature list)
- `model_metadata.json` — version, metrics, hyperparameters, PSI baseline distribution,
  feature importance, decile bins for CSI
- `model_comparison_YYYY_MM_DD.csv` — all three candidates side-by-side

---

## Model Inference (`utils/model_prediction.py`)

For each monthly DAG run (`snapshot_date`):
1. Load champion model + challenger (if in shadow mode)
2. Load feature store from **origination month** (snapshot_date - 6 months) — aligns with
   how training joined features (MOB-0 features predict MOB-6 outcome)
3. Score all customers in the origination feature store
4. Save `predictions_{date}.parquet`: `Customer_ID`, `snapshot_date`, `score`,
   `predicted_label` (>= 0.5), `model_version`, and if challenger active:
   `challenger_score`, `challenger_predicted_label`, `challenger_model_version`

Inference skips months before Jul 2023 (no origination feature store exists 6 months back
from Jan-Jun 2023 runs).

---

## Model Monitoring (`utils/model_monitoring.py`)

Runs on every DAG cycle, processes all historical prediction files (full snapshot approach).

### Performance monitoring (requires labels)
For each prediction month where `gold_label_store_{date}.parquet` exists:
- Inner join predictions with labels on `Customer_ID` (~480-530 matched rows/month)
- Compute **AUC**, **Gini** (= 2xAUC - 1), **KS statistic**
- Compute same metrics for challenger (shadow comparison)

### Stability monitoring (no labels needed)
- **Rolling PSI**: Compare current month's score distribution vs prior 3 months of inference.
  Uses full scored population each month for a stable PSI reference.
  PSI < 0.10 = stable (green) | 0.10-0.25 = monitor (orange) | > 0.25 = retrain (red)
- **CSI**: PSI per top feature using decile bins from training baseline stored in metadata
- **Label drift**: Default rate per cohort vs mean +/-20% / +/-40% alert bands

### Auto-promotion logic
Challenger promoted to champion after `CONSECUTIVE_WINS_REQUIRED=2` consecutive months
where challenger AUC > champion AUC on labeled cohorts. Promotion archives old champion,
cleans up challenger files, writes `promotion_record.json`.

### Output
- `monitoring_{date}.parquet` — one row per historical month (full snapshot per run)
- 6 PNG charts (see Backfill Results section)

---

## Model Governance SOP

| Trigger | Action |
|---------|--------|
| No champion exists | Bootstrap: train challenger, auto-promote immediately |
| Rolling PSI (3-month mean) > 0.25 | Train new challenger |
| Champion AUC drops > 5pp from baseline | Train new challenger |
| Challenger wins 2 consecutive labeled months | Auto-promote challenger -> champion |
| Challenger in shadow mode | Skip retraining (let win counter accumulate) |

Deployment: monthly batch scoring. No real-time API. Champion serves until governance
triggers replacement.

---

## Airflow DAG Design

**DAG id:** `ml_pipeline_dag`
**Schedule:** `@monthly` | **Start:** 2023-01-01 | **End:** 2024-12-01
**max_active_runs:** 1 (sequential processing — prevents parallel backfill conflicts)

### Task graph

```
data_pipeline
    >> data_quality
    >> check_retrain_needed  (BranchPythonOperator)
         |-> "train_challenger"      |-> "skip_training" (EmptyOperator)
              |                              |
    model_inference  <-------- (TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    >> model_monitoring
```

### Task descriptions

**data_pipeline** — Bronze -> Silver -> Gold for LMS labels + feature store. Fully idempotent
(skips if output files already exist). Data sources: PySpark for LMS/silver, pandas for
feature bronze + gold.

**data_quality** — Validates gold feature store and label store for snapshot_date. Empty label
store (Jan-Jun 2023) is a warning only — labels only available 6 months after origination.
Raises ValueError on critical failures to block downstream tasks.

**check_retrain_needed** — BranchPythonOperator returning `train_challenger` or
`skip_training`. Logic:
1. No champion -> train (bootstrap)
2. Challenger in shadow -> skip (preserve win counter)
3. Recent 3-month mean PSI > 0.25 -> train
4. Champion AUC drop > 5pp from metadata baseline -> train

**train_challenger** — Trains 3 models, picks best, saves as challenger. Auto-promotes on
bootstrap (no existing champion).

**skip_training** — EmptyOperator. Inference + monitoring run regardless of which branch.

**model_inference** — Loads champion (+ challenger if present), scores origination-month
feature store, saves predictions parquet.

**model_monitoring** — Full history monitoring: rolling PSI, AUC/Gini/KS, CSI, label drift,
auto-promotion check, 6 PNG charts.

---

## Backfill Results (Jan 2023 -> Dec 2024)

All 24 months processed successfully. One training event (bootstrap at Jan 2023).

| Metric | Result |
|--------|--------|
| Champion model | XGBoost (best OOT AUC) |
| Retraining events | 1 (bootstrap only — PSI and AUC remained stable throughout) |
| AUC — in-sample cohorts (Jul 2023-Jun 2024) | 0.89 - 0.93 |
| AUC — true OOT cohorts (Jul-Dec 2024) | 0.80 - 0.84 |
| Gini range | 0.60 - 0.85 |
| KS range | 0.49 - 0.73 |
| Rolling PSI | < 0.03 all months (all green — well below 0.10 threshold) |
| Label drift | 26%-33% default rate, stable within +/-20% band |
| Challenger | Never trained (model stable, no drift trigger fired) |
| Prediction files | 18 months x 8,974 rows each |

### Monitoring PNGs (`datamart/gold/monitoring/`)

- `performance_over_time.png` — AUC/Gini/KS across 18 labeled cohorts
- `psi_over_time.png` — rolling PSI all green, no retrain triggered
- `score_distribution.png` — predicted probability distribution by cohort
- `csi_over_time.png` — feature stability per top feature vs training baseline
- `label_drift_over_time.png` — default rate per cohort with alert bands
- `champion_vs_challenger.png` — shadow mode scorecard (no challenger this run)

---

## File Structure

```
Assignment_2/
|-- dags/
|   `-- ml_pipeline_dag.py           <- Airflow DAG (main orchestrator)
|-- utils/
|   |-- data_processing_bronze_table.py
|   |-- data_processing_silver_table.py
|   |-- data_processing_gold_table.py        <- label store, includes origination_date
|   |-- data_processing_feature_bronze_table.py  <- pandas (clickstream/attributes/financials)
|   |-- data_processing_feature_silver_table.py  <- PySpark cleaning
|   |-- data_processing_feature_gold_table.py    <- pandas join -> ML-ready feature store
|   |-- data_quality.py              <- QC gate
|   |-- model_training.py            <- 3-model training, temporal split, CSI baseline
|   |-- model_prediction.py          <- champion + shadow challenger scoring
|   `-- model_monitoring.py          <- PSI/CSI/AUC/Gini/KS, auto-promotion, PNG plots
|-- data/                            <- CSV source files (gitignored)
|-- datamart/
|   |-- bronze/
|   |-- silver/
|   `-- gold/
|       |-- label_store/             <- monthly parquet (measurement date keyed)
|       |-- feature_store/           <- monthly parquet (origination date keyed)
|       |-- predictions/             <- monthly inference output
|       `-- monitoring/              <- parquets + 6 PNG charts
|-- model_store/                     <- champion_model.pkl, model_metadata.json
|-- Dockerfile
|-- docker-compose.yaml
|-- requirements.txt
`-- Readme.txt                       <- GitHub repo link
```

---

## Docker Setup

`docker-compose.yaml` mounts: `./dags`, `./utils`, `./data`, `./datamart`, `./model_store`

`Dockerfile` installs: `apache-airflow`, `pyspark`, `scikit-learn`, `xgboost`, `pandas`,
`matplotlib`, `scipy`, `pyarrow`, `python-dateutil`, `default-jdk-headless`

Fresh restart (wipes Airflow DB + all data):
```powershell
docker-compose down -v
docker-compose up -d
# wait ~30s, toggle DAG ON at localhost:8080
```

---

## Known Issues and Fixes (ML-relevant)

### PSI always red (fixed)
Training-baseline PSI (current inference cohort vs training distribution aggregate) is
structurally inflated — different population sizes, different MOB. **Fix:** Rolling PSI compares
current month's scores vs prior 3 months of inference scores using the same scored
population. Catches genuine drift, ignores structural offset.

### AUC all null in monitoring (fixed)
Monitoring was looking for labels at `pred_date + 6 months`. Label store is keyed by
**measurement date** = same date as prediction snapshot. **Fix:** `label_file =
gold_label_store_{date_tag}.parquet` (same date, not +6 months).

### Challenger shadow never promoting (fixed)
Old logic returned `train_challenger` when challenger existed in shadow -> overwrote challenger
each month -> win counter reset -> no promotion ever. **Fix:** Case 2 returns `skip_training`
when challenger is in shadow, preserving the consecutive-win counter.

### sklearn 1.6 API change (fixed)
`make_scorer(roc_auc_score, needs_proba=True)` raises TypeError in sklearn 1.6.
**Fix:** `response_method="predict_proba"` parameter instead.

### High null rates in gold feature store (expected, not a bug)
8,974 clickstream visitors/month but only ~480-530 are loan customers -> 95%+ nulls in
financial columns. Data quality gate treats as warning only. Training inner-joins with labels,
nulls are median-imputed; only loan customers reach the model.

---

## Presentation Deck Instructions

**Format:** Slideument (functions as both visual aid and standalone document). Max 10 slides.
**Audience:** Manager + engineers + business users. Corporate-standard. PDF export.

### Slide outline

**Slide 1 - Title**
"CS611 Assignment 2: End-to-End ML Pipeline for Loan Default Prediction"
Student name, date, course

**Slide 2 - Problem Statement & Business Context**
- Predict loan default (DPD >= 30 at MOB 6) at point of application
- Batch decisioning: credit risk, capital allocation, regulatory compliance
- 24 months backfill (Jan 2023 - Dec 2024), ~480-530 labeled loan customers/month

**Slide 3 - System Architecture**
- Medallion architecture: Bronze -> Silver -> Gold (data pipeline)
- ML layer: Train -> Inference -> Monitoring
- Orchestrator: Apache Airflow monthly schedule, Docker deployment

**Slide 4 - Airflow DAG Design**
- Task graph diagram (copy from DAG Design section)
- BranchPythonOperator: bootstrap / PSI breach / AUC decay triggers
- TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS for inference convergence
- Shadow mode: challenger evaluated over 2 consecutive months before promotion

**Slide 5 - Model Training & Feature Engineering**
- 3 candidates: Logistic Regression, Random Forest, XGBoost (winner)
- 44 gold features + 7 engineered ratios = 51 total
- Temporal split: Train <= Dec 2023 | Test Jan-Mar 2024 | OOT Apr-Jun 2024
- Data leakage controls: origination-month features only, temporal join, no shuffle

**Slide 6 - Model Performance Over Time**
- Screenshot: `performance_over_time.png`
- AUC 0.89-0.93 (in-sample Jul 2023-Jun 2024), 0.80-0.84 (true OOT Jul-Dec 2024)
- All cohorts well above 0.60 minimum floor
- Natural AUC decline from in-sample to OOT confirms no data leakage

**Slide 7 - Score Stability Monitoring (Rolling PSI)**
- Screenshot: `psi_over_time.png`
- All 17 months green (PSI < 0.03), no retrain triggered
- Rolling PSI vs prior 3 months detects genuine drift, avoids structural MOB offset

**Slide 8 - Feature Stability (CSI) & Label Drift**
- Screenshot: `csi_over_time.png` + `label_drift_over_time.png`
- Default rate 26-33%, stable within +/-20% band throughout
- Feature distributions stable vs training baseline

**Slide 9 - Champion vs Challenger & Model Governance**
- Screenshot: `champion_vs_challenger.png`
- No challenger trained: PSI and AUC both stable across all 18 labeled months
- Governance SOP table: PSI trigger, AUC decay trigger, 2-consecutive-win promotion rule

**Slide 10 - Summary**
- End-to-end automated pipeline: Docker + Airflow + monthly backfill
- Strong OOT AUC (0.80-0.84), stable PSI (all green), no retraining needed
- Pipeline ready for production: governance rules defined, monitoring automated

### Design tips
- Color scheme: navy headers, white background, accent teal/orange
- Font: Calibri or Inter, 24pt+ body, 36pt+ titles
- Max 4 bullets per slide — let the charts carry the story
- Insert PNGs as images, do not recreate them

---

## Submission Checklist

- [x] `docker-compose build` succeeds
- [x] `docker-compose up` shows Airflow at localhost:8080
- [x] DAG runs end-to-end: model artefacts in `model_store/`, predictions in
  `datamart/gold/predictions/`, monitoring in `datamart/gold/monitoring/`
- [x] All 6 monitoring PNGs generated and verified
- [x] `Readme.txt` contains GitHub repo link
- [ ] PDF presentation deck (max 10 slides)
- [ ] Zip contains: `dags/`, `utils/`, `data/`, `datamart/`, `model_store/`,
  `Dockerfile`, `docker-compose.yaml`, `requirements.txt`, `Readme.txt`
