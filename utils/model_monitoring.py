"""
Model monitoring: computes performance (AUC, Gini, KS) and stability (PSI)
across monthly prediction cohorts, writes a monitoring parquet, and saves
PNG charts to datamart/gold/monitoring/.
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")   # headless backend — no display required in Docker
import matplotlib.pyplot as plt
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
    stat, _ = ks_2samp(pos, neg)
    return float(stat)


def _psi(expected_scores: np.ndarray, actual_scores: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index between training (expected) and current (actual) distributions.
    PSI < 0.10  → stable
    PSI 0.10–0.25 → slight drift, monitor
    PSI > 0.25  → significant drift, consider retraining
    """
    bins = np.linspace(0, 1, n_bins + 1)
    expected_pct = np.histogram(expected_scores, bins=bins)[0] / len(expected_scores)
    actual_pct   = np.histogram(actual_scores,   bins=bins)[0] / len(actual_scores)

    # Avoid log(0) by clipping
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct   = np.clip(actual_pct,   1e-6, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


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

    # Load model metadata for PSI baseline
    meta_path = os.path.join(model_store_dir, "model_metadata.json")
    if not os.path.exists(meta_path):
        print("[monitoring] no model metadata found — skipping")
        return

    with open(meta_path) as f:
        metadata = json.load(f)

    train_score_dist = metadata.get("train_score_distribution", {})
    baseline_pcts    = train_score_dist.get("percentiles", {})

    # Reconstruct approximate baseline score distribution from saved percentiles
    baseline_scores = np.array([float(v) for v in baseline_pcts.values()])

    # Load all prediction files accumulated so far
    pred_files = sorted(glob.glob(os.path.join(predictions_dir, "*.parquet")))
    if not pred_files:
        print("[monitoring] no prediction files found — skipping")
        return

    monitoring_rows = []

    for pf in pred_files:
        basename    = os.path.basename(pf)
        date_tag    = basename.replace("predictions_", "").replace(".parquet", "")
        pred_date   = pd.Timestamp(date_tag.replace("_", "-"))
        pred_date_str = pred_date.strftime("%Y-%m-%d")

        preds = pd.read_parquet(pf)
        if preds.empty or "score" not in preds.columns:
            continue

        scores = preds["score"].values
        psi    = _psi(baseline_scores, scores) if len(baseline_scores) > 0 else np.nan

        # Performance metrics require ground truth.
        # Labels for loans originated at pred_date are in label store at pred_date + 6 months.
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
                gini   = _gini(y_true, y_prob)
                ks     = _ks(y_true, y_prob)

        row = {
            "snapshot_date":   pred_date_str,
            "n_predictions":   len(preds),
            "mean_score":      float(scores.mean()),
            "psi":             psi,
            "auc":             auc,
            "gini":            gini,
            "ks":              ks,
            "psi_flag":        "stable" if psi < 0.10 else ("drift" if psi < 0.25 else "retrain"),
        }
        monitoring_rows.append(row)
        print(
            f"[monitoring] {pred_date_str}  PSI={psi:.4f}  "
            f"AUC={auc if not np.isnan(auc) else 'N/A'}  "
            f"Gini={gini if not np.isnan(gini) else 'N/A'}"
        )

    if not monitoring_rows:
        return

    monitor_df = pd.DataFrame(monitoring_rows).sort_values("snapshot_date")

    # Save monitoring table
    date_tag  = snapshot_date_str.replace("-", "_")
    out_path  = os.path.join(monitoring_dir, f"monitoring_{date_tag}.parquet")
    monitor_df.to_parquet(out_path, index=False)
    print(f"[monitoring] saved → {out_path}")

    # Generate visualisation plots
    _plot_performance(monitor_df, monitoring_dir)
    _plot_psi(monitor_df, monitoring_dir)
    _plot_score_distribution(pred_files, monitoring_dir)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _plot_performance(df: pd.DataFrame, out_dir: str):
    """AUC, Gini, KS over time — only for months with ground truth."""
    perf = df.dropna(subset=["auc"])
    if perf.empty:
        return

    dates = pd.to_datetime(perf["snapshot_date"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Model Performance Over Time", fontsize=14, fontweight="bold")

    for ax, col, label, colour, threshold in [
        (axes[0], "auc",  "AUC",  "#2196F3", 0.60),
        (axes[1], "gini", "Gini", "#4CAF50", 0.20),
        (axes[2], "ks",   "KS",   "#FF9800", 0.10),
    ]:
        ax.plot(dates, perf[col], marker="o", color=colour, linewidth=2)
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1, label=f"Min threshold ({threshold})")
        ax.set_ylabel(label, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    axes[2].set_xlabel("Prediction Cohort (Origination Month)")
    plt.tight_layout()
    path = os.path.join(out_dir, "performance_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[monitoring] plot → {path}")


def _plot_psi(df: pd.DataFrame, out_dir: str):
    """PSI over time with stability zones."""
    if df.empty or "psi" not in df.columns:
        return

    dates = pd.to_datetime(df["snapshot_date"])
    psi   = df["psi"]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Score Distribution Stability (PSI) Over Time", fontsize=14, fontweight="bold")

    ax.bar(dates, psi, color=[
        "#4CAF50" if v < 0.10 else ("#FF9800" if v < 0.25 else "#F44336")
        for v in psi
    ], width=20, alpha=0.8)
    ax.axhline(0.10, color="#FF9800", linestyle="--", linewidth=1.5, label="Slight drift (0.10)")
    ax.axhline(0.25, color="#F44336", linestyle="--", linewidth=1.5, label="Retrain trigger (0.25)")
    ax.set_xlabel("Prediction Cohort (Origination Month)")
    ax.set_ylabel("PSI")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    # Colour legend patches
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="#4CAF50", label="Stable  (PSI < 0.10)"),
        Patch(color="#FF9800", label="Monitor (PSI 0.10–0.25)"),
        Patch(color="#F44336", label="Retrain (PSI > 0.25)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper left")

    plt.tight_layout()
    path = os.path.join(out_dir, "psi_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[monitoring] plot → {path}")


def _plot_score_distribution(pred_files: list, out_dir: str):
    """Overlaid score distributions per cohort to visualise drift visually."""
    if len(pred_files) == 0:
        return

    # Plot at most 12 months to keep chart readable
    files_to_plot = pred_files[-12:]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Predicted Score Distribution by Cohort", fontsize=14, fontweight="bold")

    cmap   = plt.colormaps.get_cmap("viridis")
    n      = len(files_to_plot)

    for i, pf in enumerate(files_to_plot):
        basename = os.path.basename(pf)
        date_tag = basename.replace("predictions_", "").replace(".parquet", "")
        label    = date_tag.replace("_", "-")[:7]   # YYYY-MM

        preds  = pd.read_parquet(pf)
        scores = preds["score"].dropna().values
        if len(scores) == 0:
            continue

        ax.hist(
            scores, bins=20, range=(0, 1), alpha=0.4,
            label=label, color=cmap(i / max(n - 1, 1)),
            density=True, histtype="stepfilled", linewidth=0.5
        )

    ax.set_xlabel("Predicted Default Probability")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "score_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[monitoring] plot → {path}")
