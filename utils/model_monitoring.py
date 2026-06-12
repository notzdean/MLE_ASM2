"""
Model monitoring: computes performance (AUC, Gini, KS), score stability (PSI),
and covariate stability (CSI per top feature) across monthly prediction
cohorts. Writes a monitoring parquet and PNG charts to datamart/gold/monitoring/.

Stability thresholds (PSI / CSI):
  < 0.10  → stable (green)
  0.10–0.25 → moderate drift, monitor (orange)
  > 0.25  → significant drift, consider retraining (red)
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")   # headless — no display needed in Docker
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


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
    """
    Population Stability Index.
    Uses equal-width bins [0, 1] for score PSI, or decile bins for feature PSI.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    exp_pct = np.histogram(np.clip(expected, 0, 1), bins=bins)[0] / max(len(expected), 1)
    act_pct = np.histogram(np.clip(actual,   0, 1), bins=bins)[0] / max(len(actual),   1)
    exp_pct = np.clip(exp_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _feature_psi(baseline_values: np.ndarray, current_values: np.ndarray,
                 n_bins: int = 10) -> float:
    """
    PSI for a continuous feature using quantile-based bins derived from baseline.
    More robust than equal-width bins for skewed distributions.
    """
    quantiles  = np.linspace(0, 100, n_bins + 1)
    bin_edges  = np.percentile(baseline_values, quantiles)
    bin_edges  = np.unique(bin_edges)   # deduplicate in case of ties
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
    if val < 0.10:
        return "stable"
    if val < 0.25:
        return "drift"
    return "retrain"


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

    # PSI baseline: reconstruct approximate score distribution from saved percentiles
    train_dist     = metadata.get("train_score_distribution", {})
    baseline_pcts  = train_dist.get("percentiles", {})
    baseline_scores = np.array([float(v) for v in baseline_pcts.values()])

    # CSI baseline: feature mean/std from training split
    feature_baseline = metadata.get("feature_baseline_stats", {})
    top_features     = list(feature_baseline.keys())

    # Load all prediction files accumulated so far
    pred_files = sorted(glob.glob(os.path.join(predictions_dir, "*.parquet")))
    if not pred_files:
        print("[monitoring] no prediction files found — skipping")
        return

    monitoring_rows = []

    for pf in pred_files:
        basename      = os.path.basename(pf)
        date_tag      = basename.replace("predictions_", "").replace(".parquet", "")
        pred_date     = pd.Timestamp(date_tag.replace("_", "-"))
        pred_date_str = pred_date.strftime("%Y-%m-%d")

        preds = pd.read_parquet(pf)
        if preds.empty or "score" not in preds.columns:
            continue

        scores = preds["score"].dropna().values

        # --- Score PSI ---
        psi_score = _psi(baseline_scores, scores) if len(baseline_scores) > 0 else np.nan

        # --- Performance metrics (need ground truth) ---
        # Loans originated at pred_date → labels in label store at pred_date + 6 months
        label_date     = pred_date + relativedelta(months=6)
        label_date_tag = label_date.strftime("%Y_%m_%d")
        label_file     = os.path.join(
            gold_label_store_dir, f"gold_label_store_{label_date_tag}.parquet"
        )
        auc = gini = ks = np.nan
        if os.path.exists(label_file):
            labels = pd.read_parquet(label_file)[["Customer_ID", "label"]]
            joined = preds.merge(labels, on="Customer_ID", how="inner")
            if len(joined) >= 10 and joined["label"].nunique() > 1:
                y_true = joined["label"].values
                y_prob = joined["score"].values
                auc    = float(roc_auc_score(y_true, y_prob))
                gini   = _gini(y_true, pd.Series(y_prob))
                ks     = _ks(y_true, y_prob)

        # --- CSI: feature PSI for top features ---
        csi_row = {}
        feature_file = os.path.join(
            gold_feature_store_dir,
            f"gold_feature_store_{date_tag}.parquet",
        )
        if os.path.exists(feature_file) and top_features:
            feat_df = pd.read_parquet(feature_file)
            for feat in top_features:
                if feat not in feat_df.columns or feat not in feature_baseline:
                    continue
                baseline_stat  = feature_baseline[feat]
                current_vals   = feat_df[feat].dropna().values
                # Reconstruct baseline distribution from p25/mean/p75 stats
                baseline_approx = np.array([
                    baseline_stat["p25"],
                    baseline_stat["mean"],
                    baseline_stat["p75"],
                ] * max(len(current_vals) // 3, 1))
                csi_val = _feature_psi(baseline_approx, current_vals)
                csi_row[f"csi_{feat}"] = round(csi_val, 6)

        row = {
            "snapshot_date":  pred_date_str,
            "n_predictions":  len(preds),
            "mean_score":     float(scores.mean()),
            "std_score":      float(scores.std()),
            "psi_score":      psi_score,
            "psi_flag":       _psi_flag(psi_score),
            "auc":            auc,
            "gini":           gini,
            "ks":             ks,
            **csi_row,
        }
        monitoring_rows.append(row)

        csi_summary = (
            f"  max_CSI={max(csi_row.values()):.4f}" if csi_row else ""
        )
        print(
            f"[monitoring] {pred_date_str}  "
            f"PSI={psi_score:.4f} ({_psi_flag(psi_score)})  "
            f"AUC={f'{auc:.4f}' if not np.isnan(auc) else 'N/A'}"
            f"{csi_summary}"
        )

    if not monitoring_rows:
        return

    monitor_df = pd.DataFrame(monitoring_rows).sort_values("snapshot_date").reset_index(drop=True)

    date_tag = snapshot_date_str.replace("-", "_")
    out_path = os.path.join(monitoring_dir, f"monitoring_{date_tag}.parquet")
    monitor_df.to_parquet(out_path, index=False)
    print(f"[monitoring] saved → {out_path}")

    # Generate charts
    _plot_performance(monitor_df, monitoring_dir)
    _plot_psi(monitor_df, monitoring_dir)
    _plot_score_distribution(pred_files, monitoring_dir)
    _plot_csi(monitor_df, top_features, monitoring_dir)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _plot_performance(df: pd.DataFrame, out_dir: str):
    """AUC, Gini, KS over time — only for cohorts with ground truth labels."""
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
        ax.axhline(thresh, color="red", linestyle="--", linewidth=1.2,
                   label=thresh_label)
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
    """Score PSI over time — colour-coded by stability zone."""
    if df.empty:
        return

    dates = pd.to_datetime(df["snapshot_date"])
    psi   = df["psi_score"].fillna(0)

    colours = [
        "#4CAF50" if v < 0.10 else ("#FF9800" if v < 0.25 else "#F44336")
        for v in psi
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Score Distribution Stability (PSI) Over Time",
                 fontsize=14, fontweight="bold")

    ax.bar(dates, psi, color=colours, width=20, alpha=0.85, zorder=3)
    ax.axhline(0.10, color="#FF9800", linestyle="--", linewidth=1.5,
               label="Moderate drift threshold (0.10)")
    ax.axhline(0.25, color="#F44336", linestyle="--", linewidth=1.5,
               label="Retrain trigger (0.25)")
    ax.set_xlabel("Origination Month", fontsize=10)
    ax.set_ylabel("PSI", fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.tick_params(axis="x", rotation=45)

    legend_patches = [
        Patch(color="#4CAF50", label="Stable  (PSI < 0.10)"),
        Patch(color="#FF9800", label="Monitor (0.10 ≤ PSI < 0.25)"),
        Patch(color="#F44336", label="Retrain (PSI ≥ 0.25)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper left")

    plt.tight_layout()
    path = os.path.join(out_dir, "psi_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[monitoring] plot → {path}")


def _plot_score_distribution(pred_files: list, out_dir: str):
    """Overlaid predicted score distributions per cohort — shows drift visually."""
    files_to_plot = pred_files[-12:]   # last 12 months for readability
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

        scores = pd.read_parquet(pf)["score"].dropna().values
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
    """
    Covariate Shift Index (CSI) — PSI per top feature over time.
    Flags which features are drifting most, informing root-cause analysis.
    """
    csi_cols = [c for c in df.columns if c.startswith("csi_") and c != "csi_"]
    if not csi_cols:
        return

    dates = pd.to_datetime(df["snapshot_date"])
    n_feats = len(csi_cols)
    fig, axes = plt.subplots(
        n_feats, 1, figsize=(12, max(3 * n_feats, 6)), sharex=True
    )
    if n_feats == 1:
        axes = [axes]

    fig.suptitle("Feature Covariate Shift Index (CSI) Over Time",
                 fontsize=14, fontweight="bold")

    for ax, col in zip(axes, csi_cols):
        feat_name = col.replace("csi_", "")
        vals      = df[col].fillna(0)
        colours   = [
            "#4CAF50" if v < 0.10 else ("#FF9800" if v < 0.25 else "#F44336")
            for v in vals
        ]
        ax.bar(dates, vals, color=colours, width=20, alpha=0.85, zorder=3)
        ax.axhline(0.10, color="#FF9800", linestyle="--", linewidth=1, alpha=0.8)
        ax.axhline(0.25, color="#F44336", linestyle="--", linewidth=1, alpha=0.8)
        ax.set_ylabel(feat_name, fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.tick_params(axis="x", rotation=45)

    axes[-1].set_xlabel("Origination Month", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "csi_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[monitoring] plot → {path}")
