"""Evaluation metrics and plotting utilities.

All metrics are computed in dollar space (after exp-back-transforming model
predictions from log1p space), except RMSLE which is naturally defined on
log-revenue.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def compute_metrics(
    y_true_dollars: np.ndarray,
    y_pred_dollars: np.ndarray,
) -> dict[str, float]:
    """Compute the metric suite reported in the paper.

    Notes
    -----
    - RMSLE is the Kaggle competition metric for TMDB and is the most
      meaningful single number for revenue prediction since revenue is
      heavy-tailed and spans many orders of magnitude.
    - MAPE is reported for completeness but is inherently unstable on
      heavy-tailed revenue (a single small-revenue indie film can move it
      by hundreds of percent). We therefore also report MedAPE (median
      absolute percentage error), which is the standard robust alternative
      and represents the typical relative error rather than the average
      contaminated by outliers.
    - We clip the denominator at $1 to keep MAPE finite without changing
      the relative ranking of models.
    """
    y_pred_safe = np.clip(y_pred_dollars, a_min=0.0, a_max=None)
    y_true_pos = np.clip(y_true_dollars, 1.0, None)

    rmse = float(np.sqrt(mean_squared_error(y_true_dollars, y_pred_safe)))
    mae = float(mean_absolute_error(y_true_dollars, y_pred_safe))
    r2 = float(r2_score(y_true_dollars, y_pred_safe))
    rmsle = float(np.sqrt(mean_squared_error(
        np.log1p(np.clip(y_true_dollars, 0, None)),
        np.log1p(y_pred_safe),
    )))
    mape = float(mean_absolute_percentage_error(y_true_pos, y_pred_safe))
    # Median absolute percentage error — robust to heavy tails.
    abs_pct_error = np.abs(y_true_dollars - y_pred_safe) / y_true_pos
    medape = float(np.median(abs_pct_error))
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "RMSLE": rmsle,
        "MAPE": mape,
        "MedAPE": medape,
    }


def format_metrics_row(model_name: str, metrics: dict[str, float]) -> str:
    """Pretty-print one row of the leaderboard for the console log."""
    return (
        f"{model_name:<22}"
        f"  RMSE=${metrics['RMSE']:>14,.0f}"
        f"  R2={metrics['R2']:>6.4f}"
        f"  RMSLE={metrics['RMSLE']:>5.3f}"
        f"  MedAPE={metrics['MedAPE'] * 100:>6.2f}%"
        f"  MAPE={metrics['MAPE'] * 100:>7.2f}%"
    )


def save_actual_vs_predicted_grid(
    predictions_per_model: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Multi-panel scatter of actual vs predicted revenue, one panel per model.

    Both axes use log scale because the data span 9 orders of magnitude.
    """
    sns.set_theme(style="whitegrid", context="paper")
    n_models = len(predictions_per_model)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (model_name, (y_true, y_pred)) in zip(axes, predictions_per_model.items()):
        y_pred_safe = np.clip(y_pred, 1.0, None)
        y_true_safe = np.clip(y_true, 1.0, None)
        ax.scatter(y_true_safe, y_pred_safe, alpha=0.4, s=18, edgecolor="none")
        lo = min(y_true_safe.min(), y_pred_safe.min())
        hi = max(y_true_safe.max(), y_pred_safe.max())
        ax.plot([lo, hi], [lo, hi], "--r", linewidth=1.5, label="Perfect prediction")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(model_name)
        ax.set_xlabel("Actual revenue ($, log scale)")
        ax.set_ylabel("Predicted revenue ($, log scale)")
        ax.legend(loc="upper left", fontsize=8)

    # Hide any spare axes if the grid is not full.
    for ax in axes[n_models:]:
        ax.set_visible(False)

    fig.suptitle("Actual vs. Predicted Revenue — All Models", fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metrics_barplot(
    metrics_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    lower_is_better: bool = True,
) -> None:
    """Single-metric horizontal bar chart sorted by performance."""
    sns.set_theme(style="whitegrid", context="paper")
    ordered = metrics_df.sort_values(
        metric, ascending=lower_is_better,
    )
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(ordered) + 1))
    palette = sns.color_palette("viridis", n_colors=len(ordered))
    ax.barh(ordered["Model"], ordered[metric], color=palette)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(
        f"Model Comparison — {metric}"
        f" ({'lower is better' if lower_is_better else 'higher is better'})"
    )
    for i, value in enumerate(ordered[metric].values):
        ax.text(value, i, f"  {value:,.4f}" if metric in {"R2", "RMSLE"} else f"  {value:,.0f}",
                va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_plot(
    feature_names: list[str],
    importances: np.ndarray,
    output_path: Path,
    top_k: int = 20,
    title: str = "Feature Importance",
) -> None:
    """Top-k feature importance bar chart from the best tree model."""
    sns.set_theme(style="whitegrid", context="paper")
    order = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(8, 0.4 * top_k + 1))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        importances[order][::-1],
        color=sns.color_palette("rocket", n_colors=top_k),
    )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_residuals_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    model_name: str,
) -> None:
    """Residuals in log space — what bias remains after the log transform."""
    sns.set_theme(style="whitegrid", context="paper")
    y_true_log = np.log1p(np.clip(y_true, 0, None))
    y_pred_log = np.log1p(np.clip(y_pred, 0, None))
    residuals = y_true_log - y_pred_log

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(y_pred_log, residuals, alpha=0.4, s=18, edgecolor="none")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted log-revenue")
    axes[0].set_ylabel("Residual (log space)")
    axes[0].set_title(f"Residuals vs. Predicted — {model_name}")

    sns.histplot(residuals, bins=40, ax=axes[1], kde=True, color="#3b6ea5")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual (log space)")
    axes[1].set_title("Residual distribution")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_revenue_distribution_plot(revenue: np.ndarray, output_path: Path) -> None:
    """Show the raw and log-transformed revenue distributions side by side
    to justify the log-target choice in the report.
    """
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.histplot(revenue, bins=60, ax=axes[0], color="#c75450")
    axes[0].set_xlabel("Revenue ($)")
    axes[0].set_title("Raw revenue distribution (heavy right tail)")
    axes[0].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    sns.histplot(np.log1p(revenue), bins=60, ax=axes[1], color="#3b8c5a")
    axes[1].set_xlabel("log(1 + revenue)")
    axes[1].set_title("Log-transformed revenue (approximately Gaussian)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_ablation_plot(
    ablation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot how RMSLE rises when each feature group is removed.

    Larger bars = more important feature group. The dashed line is the
    full-feature baseline; bars show the delta from baseline.
    """
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(ablation_df) + 1))
    ordered = ablation_df.sort_values("RMSLE_increase", ascending=True)
    palette = sns.color_palette("flare", n_colors=len(ordered))
    ax.barh(ordered["Group removed"], ordered["RMSLE_increase"], color=palette)
    ax.set_xlabel("Δ RMSLE vs. full-feature baseline (higher = more important)")
    ax.set_title("Ablation Study — Contribution of each feature group")
    for i, value in enumerate(ordered["RMSLE_increase"].values):
        ax.text(value, i, f"  +{value:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
