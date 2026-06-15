"""
Model monitoring: computes performance (AUC, Gini, KS), score stability (PSI),
covariate stability (CSI per top feature), label drift (default rate), and
shadow-mode challenger vs champion comparison across monthly prediction cohorts.

Shadow mode promotion logic:
  - Every month, challenger_auc is computed alongside champion_auc (if labels exist)
  - If challenger_auc > champion_auc for 2 consecutive labeled cohorts
    → auto-promote challenger to champion via promote_challenger()

Stability thresholds (PSI / CSI):
  < 0.10  → stable (green)
  0.10–0.25 → moderate drift, monitor (orange)
  > 0.25  → significant drift, consider retraining (red)
"""

import glob
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

CONSECUTIVE_WINS_REQUIRED = 2   # challenger must beat champion this many months to promote


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _gini(y_true, y_score):
    return 2 * roc_auc_score(y_true, y_score) - 1


def _ks(y_true, y_score):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(ks_2samp(pos, neg)[0])


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index using equal-width bins [0, 1]."""
    bins    = np.linspace(0, 1, n_bins + 1)
    exp_pct = np.histogram(np.clip(expected, 0, 1), bins=bins)[0] / max(len(expected), 1)
    act_pct = np.histogram(np.clip(actual,   0, 1), bins=bins)[0] / max(len(actual),   1)
    exp_pct = np.clip(exp_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _feature_psi(baseline_values: np.ndarray, current_values: np.ndarray,
                 n_bins: int = 10) -> float:
    """PSI for a continuous feature using quantile-based bins from baseline."""
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(baseline_values, quantiles))
    if len(bin_edges) < 3:
        return 0.0
    exp_pct = np.histogram(baseline_values, bins=bin_edges)[0] / max(len(baseline_values), 1)
    act_pct = np.histogram(current_values,  bins=bin_edges)[0] / max(len(current_values),  1)
    exp_pct = np.clip(exp_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _psi_flag(val: float) -> str:
    if np.isnan(val):
        return "unknown"
    return "stable" if val < 0.10 else ("drift" if val < 0.25 else "retrain")


# ---------------------------------------------------------------------------
# Alert helpers
# ---------------------------------------------------------------------------

def _write_alert(monitoring_dir: str, snapshot_date_str: str, alerts: list):
    """Append PSI/CSI breach alerts to alerts.log and overwrite latest_alert.json."""
    if not alerts:
        return

    log_path  = os.path.join(monitoring_dir, "alerts.log")
    json_path = os.path.join(monitoring_dir, "latest_alert.json")
    ts        = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    with open(log_path, "a") as f:
        for alert in alerts:
            line = (
                f"{ts} | {snapshot_date_str} | {alert['type']} | "
                f"value={alert['value']:.4f} | flag={alert['flag']} | "
                f"feature={alert.get('feature', 'score')}\n"
            )
            f.write(line)
            print(f"[monitoring] ALERT: {line.strip()}")

    with open(json_path, "w") as f:
        json.dump({
            "timestamp":     ts,
            "snapshot_date": snapshot_date_str,
            "n_alerts":      len(alerts),
            "alerts":        alerts,
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Shadow mode: auto-promotion check
# ---------------------------------------------------------------------------

def _check_and_promote(monitor_df: pd.DataFrame, model_store_dir: str, monitoring_dir: str):
    """
    Check if challenger has beaten champion for CONSECUTIVE_WINS_REQUIRED labeled
    cohorts. If so, call promote_challenger() and record the event.
    """
    challenger_path = os.path.join(model_store_dir, "challenger_model.pkl")
    if not os.path.exists(challenger_path):
        return   # no challenger in shadow — nothing to promote

    if "challenger_auc" not in monitor_df.columns:
        return

    # Only consider rows where both champion and challenger were evaluated
    compared = monitor_df.dropna(subset=["auc", "challenger_auc"]).copy()
    if len(compared) < CONSECUTIVE_WINS_REQUIRED:
        print(
            f"[monitoring] shadow mode: need {CONSECUTIVE_WINS_REQUIRED} labeled comparisons, "
            f"have {len(compared)} — waiting"
        )
        return

    recent = compared.tail(CONSECUTIVE_WINS_REQUIRED)
    recent_wins = (recent["challenger_auc"] > recent["auc"]).all()

    if recent_wins:
        ch_aucs  = recent["challenger_auc"].tolist()
        ch_aucs_str = ", ".join(f"{a:.4f}" for a in ch_aucs)
        champ_aucs  = recent["auc"].tolist()
        champ_str   = ", ".join(f"{a:.4f}" for a in champ_aucs)
        print(
            f"[monitoring] PROMOTION: challenger won {CONSECUTIVE_WINS_REQUIRED} "
            f"consecutive months  challenger AUC=[{ch_aucs_str}]  "
            f"champion AUC=[{champ_str}]"
        )

        from utils.model_training import promote_challenger
        promote_challenger(model_store_dir)

        # Write promotion log to monitoring dir for visibility
        promo_log = {
            "promoted_at":           datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "consecutive_wins":      CONSECUTIVE_WINS_REQUIRED,
            "challenger_auc_months": ch_aucs,
            "champion_auc_months":   champ_aucs,
            "cohorts":               recent["snapshot_date"].tolist(),
        }
        with open(os.path.join(monitoring_dir, "promotion_log.json"), "w") as f:
            json.dump(promo_log, f, indent=2)
        print(f"[monitoring] promotion_log.json written")
    else:
        last_ch  = compared["challenger_auc"].iloc[-1]
        last_ch_str  = f"{last_ch:.4f}"
        last_champ   = compared["auc"].iloc[-1]
        last_champ_str = f"{last_champ:.4f}"
        print(
            f"[monitoring] shadow mode: challenger has not won "
            f"{CONSECUTIVE_WINS_REQUIRED} consecutive months yet  "
            f"(last: challenger={last_ch_str} vs champion={last_champ_str})"
        )


# ---------------------------------------------------------------------------
# Core monitoring function
# ---------------------------------------------------------------------------

def run_monitoring(
    snapshot_date_str: str,
    predictions_dir: str,
    gold_label_store_dir: str,
    gold_feature_store_dir: str,
    monitoring_dir: str,
    model_store_dir: str,
):
    os.makedirs(monitoring_dir, exist_ok=True)

    meta_path = os.path.join(model_store_dir, "model_metadata.json")
    if not os.path.exists(meta_path):
        print("[monitoring] no model metadata found — skipping")
        return

    with open(meta_path) as f:
        metadata = json.load(f)

    train_dist      = metadata.get("train_score_distribution", {})
    baseline_scores = np.array([float(v) for v in train_dist.get("percentiles", {}).values()])
    feature_baseline = metadata.get("feature_baseline_stats", {})
    top_features     = list(feature_baseline.keys())

    pred_files = sorted(glob.glob(os.path.join(predictions_dir, "*.parquet")))
    if not pred_files:
        print("[monitoring] no prediction files found — skipping")
        return

    ROLLING_WINDOW = 3   # number of prior months used as PSI rolling baseline

    monitoring_rows = []

    for idx, pf in enumerate(pred_files):
        basename      = os.path.basename(pf)
        date_tag      = basename.replace("predictions_", "").replace(".parquet", "")
        pred_date     = pd.Timestamp(date_tag.replace("_", "-"))
        pred_date_str = pred_date.strftime("%Y-%m-%d")

        preds = pd.read_parquet(pf)
        if preds.empty or "score" not in preds.columns:
            continue

        scores = preds["score"].dropna().values

        # Rolling PSI: compare current month vs previous ROLLING_WINDOW months of inference.
        # This measures "is the model drifting recently" rather than "does this month look
        # like training time" — avoids structural red-PSI caused by MOB-6 vs training distribution.
        if idx >= 1:
            prior_files  = pred_files[max(0, idx - ROLLING_WINDOW):idx]
            prior_scores = np.concatenate([
                pd.read_parquet(p)["score"].dropna().values
                for p in prior_files
                if os.path.exists(p)
            ])
            psi_score = _psi(prior_scores, scores) if len(prior_scores) > 0 else np.nan
        else:
            psi_score = np.nan   # first month — no prior reference

        # Also track training-baseline PSI as a secondary reference column
        psi_vs_training = _psi(baseline_scores, scores) if len(baseline_scores) > 0 else np.nan

        # --- Performance + label drift (requires ground truth) ---
        # Label store is dated by MEASUREMENT DATE = same as feature snapshot date.
        # Predictions for Jul 2023 contain Jan 2023 cohort customers (scored at MOB 6),
        # whose labels are in gold_label_store_2023_07_01.parquet — same date, not +6 months.
        label_file = os.path.join(
            gold_label_store_dir, f"gold_label_store_{date_tag}.parquet"
        )
        auc = gini = ks = default_rate = np.nan
        challenger_auc = np.nan

        if os.path.exists(label_file):
            labels = pd.read_parquet(label_file)[["Customer_ID", "label"]]
            joined = preds.merge(labels, on="Customer_ID", how="inner")

            if len(joined) >= 10:
                default_rate = float(joined["label"].mean())

                if joined["label"].nunique() > 1:
                    y_true = joined["label"].values
                    y_prob = joined["score"].values
                    auc    = float(roc_auc_score(y_true, y_prob))
                    gini   = _gini(y_true, pd.Series(y_prob))
                    ks     = _ks(y_true, y_prob)

                    # Challenger comparison (shadow mode)
                    if "challenger_score" in joined.columns:
                        ch_prob = joined["challenger_score"].dropna()
                        if len(ch_prob) >= 10:
                            challenger_auc = float(roc_auc_score(y_true, ch_prob.values))

        # --- CSI: feature PSI for top features ---
        csi_row      = {}
        feature_file = os.path.join(
            gold_feature_store_dir, f"gold_feature_store_{date_tag}.parquet"
        )
        if os.path.exists(feature_file) and top_features:
            feat_df = pd.read_parquet(feature_file)
            for feat in top_features:
                if feat not in feat_df.columns or feat not in feature_baseline:
                    continue
                bstat        = feature_baseline[feat]
                current_vals = feat_df[feat].dropna().values
                if "percentiles" in bstat:
                    # Use stored decile bin edges: 10 equal-quantile training bins → expected = uniform
                    pcts      = bstat["percentiles"]
                    bin_edges = np.unique(np.array([float(pcts[str(p)]) for p in range(0, 101, 10)]))
                    if len(bin_edges) >= 3:
                        n_b     = len(bin_edges) - 1
                        exp_pct = np.clip(np.full(n_b, 1.0 / n_b), 1e-6, None)
                        act_cnt = np.histogram(current_vals, bins=bin_edges)[0]
                        act_pct = np.clip(act_cnt / max(len(current_vals), 1), 1e-6, None)
                        csi_val = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
                    else:
                        csi_val = 0.0
                else:
                    baseline_approx = np.array(
                        [bstat.get("p25", 0), bstat.get("mean", 0), bstat.get("p75", 0)]
                        * max(len(current_vals) // 3, 1)
                    )
                    csi_val = _feature_psi(baseline_approx, current_vals)
                csi_row[f"csi_{feat}"] = round(csi_val, 6)

        row = {
            "snapshot_date":    pred_date_str,
            "n_predictions":    len(preds),
            "mean_score":       float(scores.mean()),
            "std_score":        float(scores.std()),
            "psi_score":        psi_score,           # rolling PSI (vs prior 3 months)
            "psi_flag":         _psi_flag(psi_score),
            "psi_vs_training":  psi_vs_training,     # vs training baseline (reference only)
            "auc":             auc,
            "gini":            gini,
            "ks":              ks,
            "default_rate":    default_rate,
            "challenger_auc":  challenger_auc,
            "challenger_wins": (
                not np.isnan(challenger_auc) and not np.isnan(auc)
                and challenger_auc > auc
            ),
            **csi_row,
        }
        monitoring_rows.append(row)

        shadow_str = (
            f"  challenger_AUC={challenger_auc:.4f} {'✓WIN' if row['challenger_wins'] else '✗LOSS'}"
            if not np.isnan(challenger_auc) else ""
        )
        csi_str = (
            f"  max_CSI={max(csi_row.values()):.4f}" if csi_row else ""
        )
        psi_disp     = f"{psi_score:.4f}" if not np.isnan(psi_score) else "N/A (first month)"
        psi_tr_disp  = f"{psi_vs_training:.4f}" if not np.isnan(psi_vs_training) else "N/A"
        print(
            f"[monitoring] {pred_date_str}  "
            f"PSI(rolling)={psi_disp} ({_psi_flag(psi_score)})  "
            f"PSI(vs_training)={psi_tr_disp}  "
            f"AUC={f'{auc:.4f}' if not np.isnan(auc) else 'N/A'}  "
            f"DefaultRate={f'{default_rate:.3f}' if not np.isnan(default_rate) else 'N/A'}"
            f"{shadow_str}{csi_str}"
        )

        # PSI / CSI breach alerts
        breach_alerts = []
        if not np.isnan(psi_score) and psi_score > 0.25:
            breach_alerts.append({
                "type": "PSI_BREACH", "feature": "score",
                "value": round(float(psi_score), 6), "flag": "retrain",
            })
        for csi_key, csi_val in csi_row.items():
            if csi_val > 0.25:
                breach_alerts.append({
                    "type": "CSI_BREACH", "feature": csi_key.replace("csi_", ""),
                    "value": round(float(csi_val), 6), "flag": "retrain",
                })
        if breach_alerts:
            _write_alert(monitoring_dir, pred_date_str, breach_alerts)

    if not monitoring_rows:
        return

    monitor_df = (
        pd.DataFrame(monitoring_rows)
        .sort_values("snapshot_date")
        .reset_index(drop=True)
    )

    date_tag = snapshot_date_str.replace("-", "_")
    out_path = os.path.join(monitoring_dir, f"monitoring_{date_tag}.parquet")
    monitor_df.to_parquet(out_path, index=False)
    print(f"[monitoring] saved → {out_path}")

    # Shadow mode: check if challenger is ready for promotion
    _check_and_promote(monitor_df, model_store_dir, monitoring_dir)

    # Generate charts
    _plot_performance(monitor_df, monitoring_dir)
    _plot_psi(monitor_df, monitoring_dir)
    _plot_score_distribution(pred_files, monitoring_dir)
    _plot_csi(monitor_df, top_features, monitoring_dir)
    _plot_label_drift(monitor_df, monitoring_dir)
    _plot_champion_vs_challenger(monitor_df, monitoring_dir)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _plot_performance(df: pd.DataFrame, out_dir: str):
    perf = df.dropna(subset=["auc"])
    if perf.empty:
        print("[monitoring] no labeled cohorts yet — skipping performance plot")
        return

    dates = pd.to_datetime(perf["snapshot_date"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Model Performance Over Time (Cohort AUC / Gini / KS)",
                 fontsize=14, fontweight="bold")

    specs = [
        ("auc",  "AUC",  "#2196F3", 0.60, "Min acceptable AUC (0.60)"),
        ("gini", "Gini", "#4CAF50", 0.20, "Min acceptable Gini (0.20)"),
        ("ks",   "KS",   "#FF9800", 0.10, "Min acceptable KS (0.10)"),
    ]
    for ax, (col, ylabel, colour, thresh, thresh_label) in zip(axes, specs):
        ax.plot(dates, perf[col], marker="o", color=colour, linewidth=2, zorder=3)
        ax.fill_between(dates, perf[col], alpha=0.15, color=colour)
        ax.axhline(thresh, color="red", linestyle="--", linewidth=1.2, label=thresh_label)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    axes[-1].set_xlabel("Origination Month (Feature Snapshot Date)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "performance_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[monitoring] plot → {path}")


def _plot_psi(df: pd.DataFrame, out_dir: str):
    if df.empty:
        return

    dates   = pd.to_datetime(df["snapshot_date"])
    psi     = df["psi_score"].fillna(0)
    colours = ["#4CAF50" if v < 0.10 else ("#FF9800" if v < 0.25 else "#F44336") for v in psi]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Score Distribution Stability — Rolling PSI (vs Prior 3 Months)",
                 fontsize=14, fontweight="bold")
    ax.bar(dates, psi, color=colours, width=pd.Timedelta(days=20), alpha=0.85, zorder=3)
    ax.axhline(0.10, color="#FF9800", linestyle="--", linewidth=1.5,
               label="Moderate drift threshold (0.10)")
    ax.axhline(0.25, color="#F44336", linestyle="--", linewidth=1.5,
               label="Retrain trigger (0.25)")
    ax.set_xlabel("Origination Month", fontsize=10)
    ax.set_ylabel("PSI (rolling)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(handles=[
        Patch(color="#4CAF50", label="Stable  (PSI < 0.10)"),
        Patch(color="#FF9800", label="Monitor (0.10 ≤ PSI < 0.25)"),
        Patch(color="#F44336", label="Retrain (PSI ≥ 0.25)"),
    ], fontsize=9, loc="upper left")
    plt.tight_layout()
    path = os.path.join(out_dir, "psi_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[monitoring] plot → {path}")


def _plot_score_distribution(pred_files: list, out_dir: str):
    files_to_plot = pred_files[-12:]
    if not files_to_plot:
        return

    cmap = plt.colormaps.get_cmap("viridis")
    n    = len(files_to_plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Predicted Default Probability by Cohort",
                 fontsize=14, fontweight="bold")

    for i, pf in enumerate(files_to_plot):
        basename = os.path.basename(pf)
        date_tag = basename.replace("predictions_", "").replace(".parquet", "")
        label    = date_tag.replace("_", "-")[:7]
        scores   = pd.read_parquet(pf)["score"].dropna().values
        if len(scores) == 0:
            continue
        ax.hist(
            scores, bins=20, range=(0, 1), alpha=0.4,
            label=label, color=cmap(i / max(n - 1, 1)),
            density=True, histtype="stepfilled", linewidth=0.5,
        )

    ax.set_xlabel("Predicted Default Probability", fontsize=10)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "score_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[monitoring] plot → {path}")


def _plot_csi(df: pd.DataFrame, top_features: list, out_dir: str):
    csi_cols = [c for c in df.columns if c.startswith("csi_")]
    if not csi_cols:
        return

    dates   = pd.to_datetime(df["snapshot_date"])
    n_feats = len(csi_cols)
    fig, axes = plt.subplots(n_feats, 1, figsize=(12, max(3 * n_feats, 6)), sharex=True)
    if n_feats == 1:
        axes = [axes]
    fig.suptitle("Feature Covariate Shift Index (CSI) Over Time",
                 fontsize=14, fontweight="bold")

    for ax, col in zip(axes, csi_cols):
        vals    = df[col].fillna(0)
        colours = ["#4CAF50" if v < 0.10 else ("#FF9800" if v < 0.25 else "#F44336") for v in vals]
        ax.bar(dates, vals, color=colours, width=pd.Timedelta(days=20), alpha=0.85, zorder=3)
        ax.axhline(0.10, color="#FF9800", linestyle="--", linewidth=1, alpha=0.8)
        ax.axhline(0.25, color="#F44336", linestyle="--", linewidth=1, alpha=0.8)
        ax.set_ylabel(col.replace("csi_", ""), fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.tick_params(axis="x", rotation=45)

    axes[-1].set_xlabel("Origination Month", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "csi_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[monitoring] plot → {path}")


def _plot_label_drift(df: pd.DataFrame, out_dir: str):
    labelled = df.dropna(subset=["default_rate"])
    if labelled.empty:
        print("[monitoring] no labeled cohorts yet — skipping label drift plot")
        return

    dates         = pd.to_datetime(labelled["snapshot_date"])
    default_rate  = labelled["default_rate"]
    baseline_rate = float(default_rate.mean())

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Label Drift — Default Rate Per Origination Cohort",
                 fontsize=14, fontweight="bold")
    ax.bar(dates, default_rate, color="#9C27B0", width=pd.Timedelta(days=20), alpha=0.75, zorder=3,
           label="Cohort default rate")
    ax.axhline(baseline_rate, color="#333333", linestyle="--", linewidth=1.5,
               label=f"Mean default rate ({baseline_rate:.2%})")
    ax.axhline(baseline_rate * 1.20, color="#FF9800", linestyle=":", linewidth=1.2,
               label="+20% band (monitor)")
    ax.axhline(baseline_rate * 1.40, color="#F44336", linestyle=":", linewidth=1.2,
               label="+40% band (investigate)")
    ax.set_xlabel("Origination Month (Feature Snapshot Date)", fontsize=10)
    ax.set_ylabel("Default Rate", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    path = os.path.join(out_dir, "label_drift_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[monitoring] plot → {path}")


def _plot_champion_vs_challenger(df: pd.DataFrame, out_dir: str):
    """
    Champion vs challenger AUC over time — the shadow mode scorecard.
    Only plotted when at least one month has a challenger_auc value.
    """
    if "challenger_auc" not in df.columns:
        return

    compared = df.dropna(subset=["auc"])
    if compared.empty:
        return

    has_challenger = compared["challenger_auc"].notna().any()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Champion vs Challenger AUC — Shadow Mode Evaluation",
                 fontsize=14, fontweight="bold")

    dates = pd.to_datetime(compared["snapshot_date"])
    ax.plot(dates, compared["auc"], marker="o", color="#2196F3",
            linewidth=2, label="Champion AUC", zorder=3)
    ax.fill_between(dates, compared["auc"], alpha=0.10, color="#2196F3")

    if has_challenger:
        ch_rows = compared.dropna(subset=["challenger_auc"])
        ch_dates = pd.to_datetime(ch_rows["snapshot_date"])
        ax.plot(ch_dates, ch_rows["challenger_auc"], marker="s", color="#FF9800",
                linewidth=2, linestyle="--", label="Challenger AUC", zorder=3)

        # Shade months where challenger wins
        wins = ch_rows[ch_rows["challenger_wins"] == True]
        for _, win_row in wins.iterrows():
            ax.axvspan(
                pd.Timestamp(win_row["snapshot_date"]) - pd.Timedelta(days=15),
                pd.Timestamp(win_row["snapshot_date"]) + pd.Timedelta(days=15),
                alpha=0.12, color="#FF9800", label="_nolegend_",
            )

    ax.set_xlabel("Origination Month", fontsize=10)
    ax.set_ylabel("AUC", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    if not has_challenger:
        ax.text(
            0.5, 0.5, "No challenger in shadow mode yet",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="grey", alpha=0.6,
        )

    plt.tight_layout()
    path = os.path.join(out_dir, "champion_vs_challenger.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[monitoring] plot → {path}")
