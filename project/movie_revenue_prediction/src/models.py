"""Model factory for the box-office regression study.

Each returned object is a scikit-learn-compatible Pipeline that takes the raw
engineered feature DataFrame and outputs predictions in log1p-revenue space.
The training script is responsible for inverting the log transform when
evaluating in dollar space.
"""

from __future__ import annotations

from typing import Callable

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

RANDOM_SEED = 42

CATEGORICAL_COLUMNS: tuple[str, ...] = ("lang_grouped",)


def _build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    scale: bool,
) -> ColumnTransformer:
    """One-hot encode any categorical columns and (optionally) standardize numerics.

    Tree-based models do not need scaling, but linear models do. The flag keeps
    the same code path for both. Categorical columns are passed in explicitly so
    that ablation studies (which may drop the language column entirely) work.
    """
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    transformers: list[tuple[str, Pipeline, list[str]]] = [
        ("num", numeric_pipeline, numeric_columns),
    ]
    if categorical_columns:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers)


def _make_pipeline(
    numeric_columns: list[str],
    estimator,
    scale: bool,
    categorical_columns: list[str] | None = None,
) -> Pipeline:
    """Wrap an estimator together with the column-wise preprocessor."""
    if categorical_columns is None:
        categorical_columns = list(CATEGORICAL_COLUMNS)
    return Pipeline(steps=[
        ("preprocessor", _build_preprocessor(
            numeric_columns, categorical_columns, scale=scale,
        )),
        ("model", estimator),
    ])


def get_model_factories() -> dict[str, Callable[..., Pipeline]]:
    """Return one builder per model. Each builder takes the numeric column list
    (and an optional categorical_columns list for ablation use) and returns a
    fully-wired pipeline (preprocessing + estimator).

    Hyperparameters were chosen from common defaults known to perform well on
    the TMDB dataset; tuning each via grid search is out of scope but the
    ranges used are conservative and reproducible.
    """
    def linear(num_cols: list[str], cat_cols: list[str] | None = None) -> Pipeline:
        return _make_pipeline(num_cols, LinearRegression(), scale=True,
                              categorical_columns=cat_cols)

    def ridge(num_cols: list[str], cat_cols: list[str] | None = None) -> Pipeline:
        return _make_pipeline(num_cols, Ridge(alpha=1.0, random_state=RANDOM_SEED),
                              scale=True, categorical_columns=cat_cols)

    def random_forest(num_cols: list[str], cat_cols: list[str] | None = None) -> Pipeline:
        return _make_pipeline(
            num_cols,
            RandomForestRegressor(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
            scale=False,
            categorical_columns=cat_cols,
        )

    def xgboost(num_cols: list[str], cat_cols: list[str] | None = None) -> Pipeline:
        return _make_pipeline(
            num_cols,
            XGBRegressor(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=RANDOM_SEED,
                tree_method="hist",
                n_jobs=-1,
            ),
            scale=False,
            categorical_columns=cat_cols,
        )

    def lightgbm(num_cols: list[str], cat_cols: list[str] | None = None) -> Pipeline:
        return _make_pipeline(
            num_cols,
            LGBMRegressor(
                n_estimators=800,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1,
            ),
            scale=False,
            categorical_columns=cat_cols,
        )

    def catboost(num_cols: list[str], cat_cols: list[str] | None = None) -> Pipeline:
        return _make_pipeline(
            num_cols,
            CatBoostRegressor(
                iterations=800,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=RANDOM_SEED,
                verbose=False,
            ),
            scale=False,
            categorical_columns=cat_cols,
        )

    return {
        "Linear Regression": linear,
        "Ridge Regression": ridge,
        "Random Forest": random_forest,
        "XGBoost": xgboost,
        "LightGBM": lightgbm,
        "CatBoost": catboost,
    }
