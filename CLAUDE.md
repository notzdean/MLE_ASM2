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

The `A1/` folder is our own Assignment 1 work (excluded from git via `.gitignore`).

### Data pipeline (bronze → silver → gold) — already working

- `utils/data_processing_bronze_table.py` — ingest raw LMS CSV, partition by snapshot_date
- `utils/data_processing_silver_table.py` — clean loan daily table, compute mob/dpd
- `utils/data_processing_gold_table.py` — build label store (label = 1 if dpd >= 30 at mob 6)
- `utils/data_processing_feature_bronze_table.py` — ingest clickstream (monthly), attributes, financials — **rewritten to pandas** (was PySpark, caused OOM in Docker)
- `utils/data_processing_feature_silver_table.py` — clean all three feature sources (PySpark)
- `utils/data_processing_feature_gold_table.py` — join + label-encode → ML-ready feature store — **rewritten to pandas** (was PySpark, caused TaskResultLost OOM for Aug–Dec 2024)

### Gold feature store output columns (ML-ready)

**Clickstream (20 behavioral):** `fe_1` … `fe_20` (integers)

**Attributes (2):** `Age` (int, 18–100), `Occupation` (label-encoded 0–14)

**Financials (18):**
- `Annual_Income`, `Monthly_Inhand_Salary`, `Total_EMI_per_month`, `Amount_invested_monthly`, `Monthly_Balance` (floats)
- `Num_Bank_Accounts`, `Num_Credit_Card`, `Interest_Rate`, `Num_of_Loan`, `num_loan_types`, `Delay_from_due_date`, `Num_of_Delayed_Payment`, `Num_Credit_Inquiries`, `Changed_Credit_Limit`, `Outstanding_Debt`, `Credit_Utilization_Ratio`, `credit_history_months` (ints/floats)
- `Credit_Mix` (0=Bad, 1=Standard, 2=Good), `Payment_of_Min_Amount` (0=No, 1=Yes), `Payment_Behaviour` (0–5 ordered by spend×value)

**Label:** `label` (0/1) from label store, joined on `Customer_ID` + `snapshot_date`

---

## Design Decisions

### Feature sourcing choice

Use **our own A1 pipeline** because it is the most complete implementation with full silver-layer cleaning (regex corruption fixes, range validation, PII drop).
- Engineered ratio features added at training time (not in gold table): `debt_to_income`, `emi_to_salary`, `credit_utilization_ratio`
- `credit_history_months` and `num_loan_types` confirmed valuable, kept as features

### Data leakage prevention (critical)

- **Label at MOB 6:** The label is `dpd >= 30` at the customer's 6th month on book. Feature snapshot_date = loan origination month. Label measured 6 months later — never use data from those 6 months as features.
- **Label store date convention:** Partitions are named by **measurement date** (origination + 6 months). `gold_label_store_2023_07_01.parquet` contains loans originated Jan 2023, measured at MOB 6 in Jul 2023. The feature snapshot date for this cohort is also 2023-07-01 (the clickstream captured that month). So feature join uses **same date** as label store — not +6 months offset.
- **Temporal train/test/OOT split (no shuffling):** Train ≤ 2023-12-01 | Test Jan–Mar 2024 | OOT Apr–Jun 2024
- **Attributes/financials join:** Filter `snapshot_date <= feature_partition_date`, take most recent record per customer (temporal-safe, no leakage)
- **Never** include loan outcome columns (`dpd`, `mob`, `loan_status`) in feature set

### Model choices

Train three candidate models, pick best by OOT AUC (falls back to test AUC if OOT unavailable):
1. Logistic Regression (baseline, interpretable)
2. Random Forest (ensemble, handles non-linearity)
3. XGBoost (gradient boosting, typically strongest)

Evaluation metrics: AUC, Gini (= 2×AUC − 1), KS statistic.

Save best model as `model_store/champion_model.pkl` + `model_store/model_metadata.json` (training date, metrics, hyperparameters, features, PSI baseline).

### Monitoring metrics

**Performance monitoring** (requires ground truth — same date as prediction cohort):
- AUC, Gini, KS per monthly prediction cohort
- Champion vs challenger comparison in shadow mode

**Stability monitoring** (no ground truth needed):
- **Rolling PSI** (primary): Compare current month's score distribution against prior 3 months of inference scores. This measures genuine recent drift rather than structural MOB-6 vs training distribution difference.
  - PSI < 0.10 → stable (green), 0.10–0.25 → monitor (orange), > 0.25 → retrain trigger (red)
  - `psi_vs_training` stored as secondary reference column (not used for triggers)
- **CSI** (Characteristic Stability Index): PSI computed per top feature using decile bins from training baseline

**Label drift monitoring:**
- Default rate per origination cohort, flagged at +20% and +40% bands above mean

### Model governance SOP

- Retrain trigger: Rolling PSI > 0.25 OR AUC drops > 5pp from baseline
- Retraining cadence: Monthly check, retrain when trigger fires
- Deployment: Batch scoring (offline loan decisioning — no real-time API needed)
- Champion/Challenger shadow mode: challenger runs in parallel; auto-promotes after `CONSECUTIVE_WINS_REQUIRED=2` months of challenger > champion AUC on labeled cohorts
- Bootstrap: no champion exists → `check_retrain_needed` fires → `train_challenger` → auto-promotes immediately

---

## File Structure

```
Assignment_2/
├── dags/
│   └── ml_pipeline_dag.py           ← Airflow DAG (main orchestrator)
├── utils/
│   ├── data_processing_bronze_table.py          ← A1 (PySpark)
│   ├── data_processing_silver_table.py          ← A1 (PySpark)
│   ├── data_processing_gold_table.py            ← A1 (PySpark)
│   ├── data_processing_feature_bronze_table.py  ← REWRITTEN pandas (was PySpark OOM)
│   ├── data_processing_feature_silver_table.py  ← A1 (PySpark)
│   ├── data_processing_feature_gold_table.py    ← REWRITTEN pandas (was PySpark OOM)
│   ├── data_quality.py              ← QC gate between data pipeline and model tasks
│   ├── model_training.py            ← challenger training, temporal split, PSI baseline
│   ├── model_prediction.py          ← champion + shadow challenger scoring
│   └── model_monitoring.py          ← rolling PSI, AUC/Gini/KS, auto-promotion, PNG plots
├── data/                            ← CSV source files (gitignored — add manually)
│   ├── feature_clickstream.csv
│   ├── features_attributes.csv
│   ├── features_financials.csv
│   └── lms_loan_daily.csv
├── datamart/
│   ├── bronze/
│   ├── silver/
│   └── gold/
│       ├── label_store/             ← monthly parquet partitions
│       ├── feature_store/           ← monthly parquet partitions
│       ├── predictions/             ← monthly inference output
│       └── monitoring/              ← performance + stability parquets + 6 PNG plots
├── model_store/                     ← champion_model.pkl, challenger_model.pkl, model_metadata.json
├── Dockerfile
├── docker-compose.yaml              ← Airflow stack (LocalExecutor + Postgres)
├── requirements.txt
└── Readme.txt                       ← 1 line: GitHub repo link
```

---

## Airflow DAG Design

**DAG id:** `ml_pipeline_dag`
**Schedule:** `@monthly` (1st of each month, backfillable)
**Start date:** 2023-01-01 | **End date:** 2024-12-01
**max_active_runs:** 1 (months process sequentially — prevents parallel backfill conflicts)

### Task graph

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

Wraps all data processing steps in one `PythonOperator`. All steps are idempotent (skip if file exists) so retries are safe.

Steps (sequential):
1. Bronze LMS → Silver loan daily → Gold label store (for `{{ ds }}`)
2. Bronze clickstream (for `{{ ds }}`), Bronze attributes (once), Bronze financials (once)
3. Silver for all three feature sources
4. Gold feature store (for `{{ ds }}`)

Bronze + gold feature steps use **pandas** (not PySpark) — simple CSV join/filter operations, no distributed processing needed. Silver feature steps still use PySpark as in A1.

### data_quality task

Validates gold feature store and label store partitions for `{{ ds }}`. Raises `ValueError` on critical failures (missing file, empty partition, missing required columns).

Null rate checks are **warnings only** — attributes/financials columns are 95%+ null because clickstream covers ~8,974 visitors but only ~530 are loan customers. Training inner-joins with labels so only loan customers reach the model.

### check_retrain_needed task (BranchPythonOperator)

Returns `"train_challenger"` or `"skip_training"`. Triggers retraining when ANY of:
1. No champion exists yet (bootstrap — first ever run)
2. Rolling PSI > 0.25 in latest monitoring
3. Champion AUC dropped > 5pp from baseline

**Critical:** When a challenger already exists in shadow mode, the branch returns `"skip_training"` regardless of PSI/AUC. Retraining while a challenger is accumulating wins would reset its consecutive-win counter and prevent auto-promotion. A fresh challenger is only trained if the existing one gets promoted (then shadow slot is free) or if PSI/AUC triggers fire after promotion.

### train_challenger task

Steps in `utils/model_training.py`:
1. Load all gold feature store + label store partitions
2. Inner join on `Customer_ID` + `snapshot_date` (only loan customers have labels)
3. Drop leakage columns (`dpd`, `mob`, `loan_status`, `snapshot_date`)
4. Engineer ratio features: `debt_to_income`, `emi_to_salary`, `credit_utilization_ratio`
5. Temporal split: Train ≤ 2023-12-01 | Test Jan–Mar 2024 | OOT Apr–Jun 2024
6. Impute nulls (median for numeric, mode for categorical)
7. Train Logistic Regression, Random Forest, XGBoost
8. Evaluate all three → pick best OOT AUC (falls back to test AUC if OOT unavailable)
9. Save as `challenger_model.pkl` + `model_metadata.json` (includes 99-percentile PSI baseline + decile bins per top feature for CSI)
10. On bootstrap (no champion), immediately copy challenger → champion

### model_inference task

Steps in `utils/model_prediction.py`:
1. Load `champion_model.pkl` (always scores) + `challenger_model.pkl` if exists (shadow)
2. Filter feature store to loan customers only (via label store Customer_ID set)
3. Apply same ratio feature engineering as training
4. Save `predictions_{{ ds }}.parquet`: `Customer_ID`, `snapshot_date`, `score`, `predicted_label` (≥0.5), `model_version`, and if challenger: `challenger_score`, `challenger_predicted_label`, `challenger_model_version`

### model_monitoring task

Steps in `utils/model_monitoring.py`:
1. Load all prediction parquets (full history on every run)
2. **Rolling PSI:** For each month, compare score distribution vs prior 3 months (not training baseline)
3. **CSI:** Per top feature, compute PSI using decile bins stored in model_metadata.json
4. **Performance:** For months where label file exists at same date, compute AUC/Gini/KS for champion and challenger
5. **Label drift:** Default rate per cohort vs mean ±20%/40% bands
6. **Auto-promotion:** If challenger beats champion for `CONSECUTIVE_WINS_REQUIRED=2` consecutive labeled months, promote challenger → champion
7. Save `monitoring_{{ ds }}.parquet` (one row per historical month — full snapshot)
8. Generate 6 PNGs: `performance_over_time.png`, `psi_over_time.png`, `score_distribution.png`, `csi_over_time.png`, `label_drift_over_time.png`, `champion_vs_challenger.png`

---

## Backfill Results (Jan 2023 → Dec 2024)

All 24 months processed successfully. Key outcomes:

| Metric | Result |
|--------|--------|
| Champion model | XGBoost (best OOT AUC) |
| AUC range (18 labeled cohorts) | 0.75 – 0.86 |
| Gini range | 0.45 – 0.82 |
| KS range | 0.44 – 0.66 |
| Rolling PSI | < 0.06 all months (all green) |
| Label drift | 26%–33% default rate, stable within ±20% band |
| CSI | Payment_Behaviour + Num_Credit_Card orange (0.10–0.25), rest green |
| Auto-promotion | Challenger never reached 2 consecutive wins (champion consistently strong) |

Monitoring PNGs are saved at `datamart/gold/monitoring/*.png`.

---

## Docker Setup Notes

`docker-compose.yaml` mounts:
- `./dags` → `/opt/airflow/dags`
- `./utils` → `/opt/airflow/utils`
- `./data` → `/opt/airflow/data`
- `./datamart` → `/opt/airflow/datamart`
- `./model_store` → `/opt/airflow/model_store`
- `AIRFLOW__CORE__LOAD_EXAMPLES=false`

`Dockerfile` installs: `apache-airflow`, `pyspark`, `scikit-learn`, `xgboost`, `pandas`, `matplotlib`, `scipy`, `pyarrow`, `python-dateutil`, `default-jdk-headless`.

---

## Implementation Order

1. ✅ Set up Docker + Airflow (`docker-compose.yaml`, `Dockerfile`, `requirements.txt`)
2. ✅ Create the Airflow DAG (`dags/ml_pipeline_dag.py`) with all tasks
3. ✅ Wrap A1 PySpark data pipeline as Airflow PythonOperator + add `data_quality` gate
4. ✅ Write `utils/model_training.py` (challenger/champion governance, temporal split)
5. ✅ Write `utils/model_prediction.py` (champion + shadow challenger scoring)
6. ✅ Write `utils/model_monitoring.py` (rolling PSI, AUC/Gini/KS, auto-promotion, PNG plots)
7. ✅ Backfill Jan 2023 → Dec 2024 — all 24 months complete
8. ✅ Monitoring PNGs generated and verified (6 charts)
9. ⬜ Build presentation deck (≤ 10 slides) — see section below
10. ⬜ Write `Readme.txt` with GitHub repo link: `https://github.com/notzdean/MLE_ASM2.git`
11. ⬜ Create submission zip

---

## Known Issues and Runtime Fixes

### Spark OOM in bronze + gold feature processing

`toPandas()` / broadcast join OOM inside Docker causes `TaskResultLost` from Spark block manager. **Fix:** Both `data_processing_feature_bronze_table.py` and `data_processing_feature_gold_table.py` rewritten to pure pandas. Output schema and file names identical to original PySpark versions.

### Path separator convention

A1 code uses `dir + filename` string concatenation requiring trailing `/` on directory paths. All `*_DIR` constants in `ml_pipeline_dag.py` are defined with `+ "/"` appended.

### sklearn 1.6 API change

`make_scorer(roc_auc_score, needs_proba=True)` raises `TypeError`. Use `response_method="predict_proba"` instead.

### PSI always red (structural, now fixed)

Training baseline scores come from the aggregate of all training cohorts. A single monthly inference cohort compared to this aggregate produces high PSI structurally. **Fix:** Rolling PSI compares current month vs prior 3 months of inference — catches genuine drift, ignores structural offset.

### AUC all null in monitoring (fixed)

Monitoring was looking for labels at `pred_date + 6 months`. But label store is dated by **measurement date** = same as prediction date. **Fix:** `label_file = gold_label_store_{date_tag}.parquet` (same date).

### Bar chart invisible bars (fixed)

`ax.bar(..., width=20)` with datetime x-axis = 20 nanoseconds → invisible bars. **Fix:** `width=pd.Timedelta(days=20)` in PSI, CSI, and label drift plots.

### Challenger shadow never promoting (fixed)

Old Case 2 logic: challenger exists → always return `"train_challenger"` → overwrites challenger each month → consecutive-win counter never accumulates → no promotion. **Fix:** When challenger in shadow, return `"skip_training"` so win counter can accumulate.

### High null rates in gold feature store (expected)

~8,974 clickstream visitors/month vs ~530 loan customers → 95%+ nulls in financial columns. `data_quality.py` treats as warning only. Training inner-joins with labels, nulls are imputed, only loan customers reach the model.

### DAG task ordering with max_active_runs=1

Prevents new runs from starting but does not stop already-running ones. If two runs are simultaneously active, mark the later one Failed to free the slot.

### Fresh restart procedure

```powershell
docker-compose down -v
Remove-Item -Recurse -Force datamart\bronze\*, datamart\silver\*, datamart\gold\* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force model_store\* -ErrorAction SilentlyContinue
docker-compose up -d
# wait ~30s, then toggle DAG ON in Airflow UI at localhost:8080
```

### Re-run monitoring only (without rerunning full pipeline)

```powershell
Remove-Item datamart\gold\monitoring\monitoring_*.parquet -Force
Remove-Item datamart\gold\monitoring\*.png -Force
# In Airflow UI: clear the model_monitoring task on the latest (Dec 2024) run
```

---

## Presentation Deck Instructions

**Audience:** Professor / grader. Corporate-standard, ≤ 10 slides. PDF export.

**Goal for the slide builder:** Create a professional PowerPoint/Google Slides deck that tells the story of this ML pipeline — what was built, why the design decisions were made, and what the monitoring results show. Use the PNG charts from `datamart/gold/monitoring/` as screenshots.

### Slide outline (10 slides)

**Slide 1 — Title**
- Title: "CS611 Assignment 2: End-to-End ML Pipeline for Loan Default Prediction"
- Subtitle: Student name, date
- Clean corporate cover, logo or course name

**Slide 2 — Problem Statement & Business Context**
- Predict loan default (DPD ≥ 30 at Month-On-Book 6) for offline batch decisioning
- Why it matters: credit risk, capital allocation, regulatory compliance
- Scope: 24 months of data (Jan 2023 – Dec 2024), ~530 loan customers/month

**Slide 3 — System Architecture**
- Diagram: Bronze → Silver → Gold medallion architecture (data pipeline)
- Right side: ML layer — Train → Inference → Monitoring
- Orchestrator: Apache Airflow (monthly schedule, Docker deployment)
- Label: PySpark for data pipeline, pandas + sklearn for ML tasks

**Slide 4 — Airflow DAG Design**
- Show the task graph (copy from DAG Design section above)
- Explain BranchPythonOperator logic: bootstrap / PSI breach / AUC decay
- Highlight: TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS for inference convergence
- Shadow mode: challenger runs in parallel, auto-promotes after 2 consecutive wins

**Slide 5 — Model Training & Feature Engineering**
- 3 candidates: Logistic Regression, Random Forest, XGBoost
- Feature set: 20 clickstream + 2 attributes + 18 financials + 3 engineered ratios
- Temporal split: Train ≤ Dec 2023 | Test Jan–Mar 2024 | OOT Apr–Jun 2024
- Data leakage controls: no outcome columns, temporal join, no shuffle
- Winner: XGBoost (best OOT AUC)

**Slide 6 — Model Performance Over Time**
- Screenshot: `performance_over_time.png`
- Caption: AUC 0.75–0.86 across 18 cohorts, all above 0.60 floor; Gini 0.45–0.82; KS 0.44–0.66
- Key insight: Model maintains strong discrimination across all origination cohorts with no degradation trend

**Slide 7 — Score Stability Monitoring (Rolling PSI)**
- Screenshot: `psi_over_time.png`
- Caption: Rolling PSI (vs prior 3 months) — all green (< 0.06), no retrain triggered
- Explain: Rolling PSI compares consecutive inference cohorts to detect genuine distribution shift, not structural MOB-offset difference
- Secondary: `psi_vs_training` stored for reference but not used as trigger

**Slide 8 — Feature Stability (CSI) & Label Drift**
- Left/top: screenshot `csi_over_time.png` — Payment_Behaviour + Num_Credit_Card in orange (monitor zone); 5 other features green
- Right/bottom: screenshot `label_drift_over_time.png` — default rate stable at ~26–33%, within ±20% band
- Key insight: No data drift requiring action; two features in watch zone are expected (categorical encoding variation)

**Slide 9 — Champion vs Challenger (Shadow Mode)**
- Screenshot: `champion_vs_challenger.png`
- Caption: Both models track closely (0.75–0.92 AUC). Challenger won isolated months but never 2 consecutive → no promotion. Champion remains deployed.
- Explain governance: CONSECUTIVE_WINS_REQUIRED=2 prevents premature promotion on noise

**Slide 10 — Summary & Governance SOP**
- What was built: end-to-end pipeline, fully automated, Docker + Airflow
- Results: model stable, performance strong, no retraining triggered in 18 labeled months
- Governance rules (table):
  - PSI > 0.25 (rolling) → train new challenger
  - AUC drop > 5pp → train new challenger
  - Challenger wins 2 consecutive months → auto-promote
  - Monthly batch scoring, no real-time API needed

### Design tips for the slide builder

- **Color scheme:** Blue/navy for headers, white background, accent orange or teal for highlights — mimics corporate banking deck style
- **Font:** Calibri or Inter, 24pt+ for body, 36pt+ for slide titles
- **Charts:** Insert the PNG files as images — do not recreate them. Crop whitespace around each PNG before inserting.
- **Avoid bullet-point overload:** Max 4 bullets per slide. Let the charts speak.
- **Export:** Save as PDF for submission.

PNG files location: `datamart/gold/monitoring/`
- `performance_over_time.png`
- `psi_over_time.png`
- `score_distribution.png`
- `csi_over_time.png`
- `label_drift_over_time.png`
- `champion_vs_challenger.png`

---

## Submission Checklist

- [x] `docker-compose build` succeeds
- [x] `docker-compose up` shows Airflow at localhost:8080
- [x] DAG appears in Airflow UI, can be triggered manually
- [x] Backfill complete: model artifacts in `model_store/`, predictions in `datamart/gold/predictions/`, monitoring in `datamart/gold/monitoring/`
- [x] All 6 monitoring PNGs generated and verified
- [ ] `Readme.txt` contains GitHub repo link: `https://github.com/notzdean/MLE_ASM2.git`
- [ ] PDF presentation deck uploaded separately (≤ 10 slides)
- [ ] Zip contains: `dags/`, `utils/`, `data/`, `datamart/`, `model_store/`, `Dockerfile`, `docker-compose.yaml`, `requirements.txt`, `Readme.txt`
