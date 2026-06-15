"""Main training and evaluation pipeline.

Trains six regression models on the TMDB Box Office Prediction dataset,
evaluates them with cross-validation and on a held-out test split, runs
an ablation study on feature groups, and saves plots and metric tables.

Usage
-----
    python main.py

Outputs are written to ./results/ and ./plots/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

# Local imports — assumes script is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation import (
    compute_metrics,
    format_metrics_row,
    save_ablation_plot,
    save_actual_vs_predicted_grid,
    save_feature_importance_plot,
    save_metrics_barplot,
    save_residuals_plot,
    save_revenue_distribution_plot,
)
from src.feature_engineering import (
    FEATURE_GROUPS,
    build_features,
)
from src.models import CATEGORICAL_COLUMNS, RANDOM_SEED, get_model_factories

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
TRAIN_CSV = PROJECT_ROOT / "train.csv"


def split_columns(feature_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) by exclusion."""
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in feature_df.columns]
    num_cols = [c for c in feature_df.columns if c not in cat_cols and c != "revenue"]
    return num_cols, cat_cols


def cross_validate_model(
    pipeline_factory,
    numeric_columns: list[str],
    categorical_columns: list[str],
    X: pd.DataFrame,
    y_log: np.ndarray,
    n_splits: int = 5,
) -> dict[str, float]:
    """K-fold CV; returns mean RMSLE over folds (in log-revenue space).

    We use RMSLE because predictions and targets are both in log space
    during training. RMSE of log-revenue equals RMSLE of revenue.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_rmsles: list[float] = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        pipeline = pipeline_factory(numeric_columns, categorical_columns)
        pipeline.fit(X.iloc[train_idx], y_log[train_idx])
        pred_log = pipeline.predict(X.iloc[val_idx])
        rmsle = float(np.sqrt(np.mean((y_log[val_idx] - pred_log) ** 2)))
        fold_rmsles.append(rmsle)
    return {
        "cv_rmsle_mean": float(np.mean(fold_rmsles)),
        "cv_rmsle_std": float(np.std(fold_rmsles)),
    }


def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_log: np.ndarray,
    y_test_dollars: np.ndarray,
    numeric_columns: list[str],
    categorical_columns: list[str],
    full_X: pd.DataFrame,
    full_y_log: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    """Train every model factory and return metrics, predictions, fitted pipelines."""
    factories = get_model_factories()
    metric_rows: list[dict[str, float]] = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fitted: dict[str, object] = {}

    print("=" * 110)
    print(f"{'Model':<22}  Training")
    print("=" * 110)

    # Clip predictions to the training range in log space to prevent linear
    # models from extrapolating into unrealistic values (a multi-billion-dollar
    # blowup on a few outliers can dominate the RMSE metric). This is a
    # standard post-processing step that does not affect tree models (which
    # cannot extrapolate by construction) and only affects very rare linear
    # predictions outside the observed log-revenue range.
    log_lo = float(np.min(y_train_log))
    log_hi = float(np.max(y_train_log))

    for name, factory in factories.items():
        t0 = time.time()
        pipeline = factory(numeric_columns, categorical_columns)
        pipeline.fit(X_train, y_train_log)
        pred_log_test = pipeline.predict(X_test)

        # Note on back-transformation: a model trained on log-revenue
        # predicts the conditional median, so naive expm1 systematically
        # under-predicts the mean. A log-normal correction (mu + sigma^2/2)
        # is the textbook fix, but it relies on residuals being approximately
        # Gaussian. On TMDB the residuals are heavy-tailed for linear
        # models, so the correction overshoots; we therefore use naive
        # back-transformation and report MedAPE alongside MAPE so the
        # heavy tail is visible in the metric suite.
        pred_log_clipped = np.clip(pred_log_test, log_lo, log_hi)
        pred_dollars = np.expm1(pred_log_clipped)
        elapsed = time.time() - t0

        metrics = compute_metrics(y_test_dollars, pred_dollars)
        cv_metrics = cross_validate_model(
            factory, numeric_columns, categorical_columns, full_X, full_y_log,
        )
        row = {
            "Model": name,
            "fit_seconds": elapsed,
            **metrics,
            **cv_metrics,
        }
        metric_rows.append(row)
        predictions[name] = (y_test_dollars, pred_dollars)
        fitted[name] = pipeline

        print(format_metrics_row(name, metrics)
              + f"  CV-RMSLE={cv_metrics['cv_rmsle_mean']:.3f}"
              + f"±{cv_metrics['cv_rmsle_std']:.3f}"
              + f"  ({elapsed:.1f}s)")

    return pd.DataFrame(metric_rows), predictions, fitted


def run_ablation(
    best_model_name: str,
    X: pd.DataFrame,
    y_log: np.ndarray,
) -> pd.DataFrame:
    """Drop one feature group at a time, re-train the best model, record ΔRMSLE.

    The baseline is the full-feature model trained under the same CV regime.
    """
    factory = get_model_factories()[best_model_name]

    def cv_rmsle_for(feature_df: pd.DataFrame) -> float:
        num_cols, cat_cols = split_columns(feature_df)
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        scores: list[float] = []
        for train_idx, val_idx in kf.split(feature_df):
            pipeline = factory(num_cols, cat_cols)
            pipeline.fit(feature_df.iloc[train_idx], y_log[train_idx])
            pred = pipeline.predict(feature_df.iloc[val_idx])
            scores.append(float(np.sqrt(np.mean((y_log[val_idx] - pred) ** 2))))
        return float(np.mean(scores))

    print()
    print("=" * 110)
    print(f"Ablation study with best model = {best_model_name}")
    print("=" * 110)

    baseline = cv_rmsle_for(X)
    print(f"Baseline (all features)        CV-RMSLE = {baseline:.4f}")

    rows: list[dict[str, float | str]] = []
    for group_name, group_cols in FEATURE_GROUPS.items():
        cols_to_drop = [c for c in group_cols if c in X.columns]
        if not cols_to_drop:
            continue
        X_reduced = X.drop(columns=cols_to_drop)
        rmsle = cv_rmsle_for(X_reduced)
        delta = rmsle - baseline
        rows.append({
            "Group removed": group_name,
            "CV_RMSLE": rmsle,
            "RMSLE_increase": delta,
            "n_features_removed": len(cols_to_drop),
        })
        print(f"  -{group_name:<20} CV-RMSLE = {rmsle:.4f}   Δ = {delta:+.4f}")

    df = pd.DataFrame(rows)
    df.attrs["baseline_rmsle"] = baseline
    return df


def get_feature_importance(
    fitted_pipeline,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[list[str], np.ndarray]:
    """Pull feature importances out of a fitted tree-based pipeline.

    Reconstructs the post-OHE feature names so the plot is human-readable.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]

    all_feature_names: list[str] = list(numeric_columns)
    if categorical_columns and "cat" in preprocessor.named_transformers_:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        all_feature_names += list(ohe.get_feature_names_out(categorical_columns))

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        # Linear models expose coefficients instead; we use absolute value.
        importances = np.abs(getattr(model, "coef_", np.zeros(len(all_feature_names))))
    return all_feature_names, np.asarray(importances)


def main() -> None:
    np.random.seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    print("Loading raw TMDB data...")
    raw = pd.read_csv(TRAIN_CSV)
    print(f"  Raw shape: {raw.shape}")

    print("Building features...")
    features = build_features(raw)
    print(f"  Feature shape: {features.shape}")

    # Drop the handful of films with revenue placeholders ($1, $5, etc.) that
    # are Kaggle artifacts and would dominate any percent-error metric.
    features = features[features["revenue"] > 1000].reset_index(drop=True)
    print(f"  After dropping revenue<=1000 placeholders: {features.shape}")

    y = features["revenue"].values
    X = features.drop(columns=["revenue"])
    y_log = np.log1p(y)
    numeric_columns, categorical_columns = split_columns(features)

    # Distribution plot — used in the report to justify log-target.
    save_revenue_distribution_plot(y, PLOTS_DIR / "01_revenue_distribution.png")

    # Hold-out split: train models on 80%, evaluate on 20% for the leaderboard,
    # but also CV-evaluate on the full set for a more stable picture.
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.20, random_state=RANDOM_SEED,
    )
    y_test_dollars = np.expm1(y_test_log)

    metrics_df, predictions, fitted_pipelines = train_all_models(
        X_train=X_train,
        X_test=X_test,
        y_train_log=y_train_log,
        y_test_dollars=y_test_dollars,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        full_X=X,
        full_y_log=y_log,
    )

    metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    print(f"\nMetrics saved to {RESULTS_DIR / 'model_metrics.csv'}")

    # Best model = lowest CV-RMSLE (more robust than single-split RMSE on a
    # heavy-tailed target).
    best_model_name = metrics_df.sort_values("cv_rmsle_mean").iloc[0]["Model"]
    print(f"\nBest model by CV-RMSLE: {best_model_name}")

    # Per-metric leaderboards.
    save_actual_vs_predicted_grid(predictions, PLOTS_DIR / "02_actual_vs_predicted_grid.png")
    save_metrics_barplot(metrics_df, "RMSLE", PLOTS_DIR / "03_rmsle_comparison.png", lower_is_better=True)
    save_metrics_barplot(metrics_df, "R2", PLOTS_DIR / "04_r2_comparison.png", lower_is_better=False)
    save_metrics_barplot(metrics_df, "MAE", PLOTS_DIR / "05_mae_comparison.png", lower_is_better=True)

    # Residuals + feature importance for the best model.
    y_test_best, y_pred_best = predictions[best_model_name]
    save_residuals_plot(y_test_best, y_pred_best, PLOTS_DIR / "06_residuals_best.png", best_model_name)

    feat_names, importances = get_feature_importance(
        fitted_pipelines[best_model_name], numeric_columns, categorical_columns,
    )
    save_feature_importance_plot(
        feat_names, importances,
        PLOTS_DIR / "07_feature_importance_best.png",
        top_k=20,
        title=f"Top 20 Feature Importances — {best_model_name}",
    )

    # Ablation study.
    ablation_df = run_ablation(best_model_name, X, y_log)
    ablation_df.to_csv(RESULTS_DIR / "ablation_study.csv", index=False)
    save_ablation_plot(ablation_df, PLOTS_DIR / "08_ablation_study.png")

    # Run summary as JSON for the report.
    summary = {
        "n_samples": int(len(features)),
        "n_features": int(X.shape[1]),
        "best_model": best_model_name,
        "best_model_metrics": metrics_df[
            metrics_df["Model"] == best_model_name
        ].iloc[0].to_dict(),
        "ablation_baseline_rmsle": ablation_df.attrs["baseline_rmsle"],
        "random_seed": RANDOM_SEED,
    }
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print("\nDone. Outputs:")
    print(f"  {RESULTS_DIR}")
    print(f"  {PLOTS_DIR}")


if __name__ == "__main__":
    main()
