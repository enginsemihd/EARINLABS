"""Feature engineering for TMDB Box Office prediction.

Extracts structured features from raw TMDB columns: JSON-like list strings
(genres, cast, crew, production_companies, ...), date columns, and text
fields. Returns a flat numeric/categorical DataFrame ready for modeling.
"""

from __future__ import annotations

import ast
from typing import Iterable

import numpy as np
import pandas as pd

# Top genres that appear with high frequency in TMDB; used as binary indicators.
TOP_GENRES: tuple[str, ...] = (
    "Drama",
    "Comedy",
    "Thriller",
    "Action",
    "Romance",
    "Adventure",
    "Crime",
    "Science Fiction",
    "Horror",
    "Family",
    "Fantasy",
    "Mystery",
    "Animation",
    "History",
    "Music",
)

# Top original languages by frequency; rest are mapped to "other".
TOP_LANGUAGES: tuple[str, ...] = (
    "en", "fr", "es", "de", "ru", "ja", "it", "hi", "ko", "zh",
)


def _safe_literal_eval(value: object) -> list:
    """Parse a Python-literal list string into a list, returning [] on failure."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _extract_names(items: list, key: str = "name") -> list[str]:
    """Pull the `key` field out of each dict in a parsed list."""
    return [d[key] for d in items if isinstance(d, dict) and key in d]


def _parse_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse all dict-list-string columns once and cache the parsed lists."""
    list_columns = [
        "genres",
        "production_companies",
        "production_countries",
        "spoken_languages",
        "Keywords",
        "cast",
        "crew",
        "belongs_to_collection",
    ]
    for col in list_columns:
        df[col + "_parsed"] = df[col].apply(_safe_literal_eval)
    return df


def _add_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """Number of items in each list field — a strong signal in TMDB data."""
    df["n_genres"] = df["genres_parsed"].apply(len)
    df["n_production_companies"] = df["production_companies_parsed"].apply(len)
    df["n_production_countries"] = df["production_countries_parsed"].apply(len)
    df["n_spoken_languages"] = df["spoken_languages_parsed"].apply(len)
    df["n_keywords"] = df["Keywords_parsed"].apply(len)
    df["n_cast"] = df["cast_parsed"].apply(len)
    df["n_crew"] = df["crew_parsed"].apply(len)
    return df


def _add_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Presence/absence flags for sparse string columns."""
    df["has_collection"] = df["belongs_to_collection_parsed"].apply(
        lambda x: 1 if len(x) > 0 else 0
    )
    df["has_homepage"] = df["homepage"].notna().astype(int)
    df["has_tagline"] = df["tagline"].notna().astype(int)
    df["has_poster"] = df["poster_path"].notna().astype(int)
    return df


def _add_genre_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary indicator per top genre (multi-label one-hot)."""
    df["_genre_names"] = df["genres_parsed"].apply(_extract_names)
    for genre in TOP_GENRES:
        col = "genre_" + genre.lower().replace(" ", "_")
        df[col] = df["_genre_names"].apply(lambda gs, g=genre: int(g in gs))
    df = df.drop(columns=["_genre_names"])
    return df


def _add_language_features(df: pd.DataFrame) -> pd.DataFrame:
    """English-language flag plus a normalized top-language category."""
    df["is_english"] = (df["original_language"] == "en").astype(int)
    df["lang_grouped"] = df["original_language"].where(
        df["original_language"].isin(TOP_LANGUAGES), other="other"
    )
    return df


def _add_crew_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gender mix in cast — proxy for the casting profile of a film."""
    def gender_counts(cast_list: list) -> tuple[int, int, int]:
        n_male = sum(1 for c in cast_list if isinstance(c, dict) and c.get("gender") == 2)
        n_female = sum(1 for c in cast_list if isinstance(c, dict) and c.get("gender") == 1)
        n_unknown = sum(1 for c in cast_list if isinstance(c, dict) and c.get("gender") == 0)
        return n_male, n_female, n_unknown

    counts = df["cast_parsed"].apply(gender_counts)
    df["cast_n_male"] = counts.apply(lambda t: t[0])
    df["cast_n_female"] = counts.apply(lambda t: t[1])
    df["cast_n_unknown_gender"] = counts.apply(lambda t: t[2])
    return df


def _parse_release_date(value: object) -> pd.Timestamp | pd._libs.tslibs.nattype.NaTType:
    """TMDB stores release_date as M/D/YY; pandas needs format='%m/%d/%y' with 2-digit years."""
    if not isinstance(value, str) or not value.strip():
        return pd.NaT
    try:
        ts = pd.to_datetime(value, format="%m/%d/%y", errors="raise")
    except (ValueError, TypeError):
        return pd.NaT
    # Two-digit-year heuristic: TMDB has no movies after the current year, so
    # any date pandas pushed to 20xx that exceeds the current year belongs to 19xx.
    current_year = pd.Timestamp.now().year
    if ts.year > current_year:
        ts = ts.replace(year=ts.year - 100)
    return ts


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Temporal features: year, month, quarter, day-of-week, decade."""
    df["release_date_parsed"] = df["release_date"].apply(_parse_release_date)
    df["release_year"] = df["release_date_parsed"].dt.year
    df["release_month"] = df["release_date_parsed"].dt.month
    df["release_quarter"] = df["release_date_parsed"].dt.quarter
    df["release_dayofweek"] = df["release_date_parsed"].dt.dayofweek
    df["release_decade"] = (df["release_year"] // 10) * 10
    # Summer (Jun-Aug) and holiday (Nov-Dec) windows are known box-office peaks.
    df["is_summer_release"] = df["release_month"].isin([6, 7, 8]).astype(int)
    df["is_holiday_release"] = df["release_month"].isin([11, 12]).astype(int)
    return df


def _add_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lengths of free-text fields — proxy for marketing effort / metadata richness."""
    df["overview_len"] = df["overview"].fillna("").str.len()
    df["title_len"] = df["title"].fillna("").str.len()
    df["tagline_len"] = df["tagline"].fillna("").str.len()
    df["original_title_len"] = df["original_title"].fillna("").str.len()
    df["title_differs_from_original"] = (
        df["title"].fillna("") != df["original_title"].fillna("")
    ).astype(int)
    return df


def _add_budget_features(df: pd.DataFrame) -> pd.DataFrame:
    """log1p(budget), missing-budget flag, and budget-per-runtime-minute."""
    df["budget_missing"] = (df["budget"] == 0).astype(int)
    # Avoid log(0) by adding 1; missing-budget flag captures the imputation.
    df["budget_log"] = np.log1p(df["budget"])
    df["runtime"] = df["runtime"].fillna(df["runtime"].median())
    df["budget_per_minute"] = df["budget"] / df["runtime"].replace(0, np.nan)
    df["budget_per_minute"] = df["budget_per_minute"].fillna(0)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering pipeline on a raw TMDB dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw TMDB Box Office Prediction dataframe (train.csv or test.csv).

    Returns
    -------
    pd.DataFrame
        Engineered features. Includes `revenue` if present in the input.
    """
    df = df.copy()
    df = _parse_list_columns(df)
    df = _add_count_features(df)
    df = _add_binary_features(df)
    df = _add_genre_flags(df)
    df = _add_language_features(df)
    df = _add_crew_features(df)
    df = _add_date_features(df)
    df = _add_text_length_features(df)
    df = _add_budget_features(df)

    keep_columns: list[str] = [
        # Numeric / ordinal
        "budget",
        "budget_log",
        "budget_missing",
        "budget_per_minute",
        "popularity",
        "runtime",
        "release_year",
        "release_month",
        "release_quarter",
        "release_dayofweek",
        "release_decade",
        "is_summer_release",
        "is_holiday_release",
        # Counts
        "n_genres",
        "n_production_companies",
        "n_production_countries",
        "n_spoken_languages",
        "n_keywords",
        "n_cast",
        "n_crew",
        "cast_n_male",
        "cast_n_female",
        "cast_n_unknown_gender",
        # Presence flags
        "has_collection",
        "has_homepage",
        "has_tagline",
        "has_poster",
        "is_english",
        "title_differs_from_original",
        # Text lengths
        "overview_len",
        "title_len",
        "tagline_len",
        "original_title_len",
        # Categorical
        "lang_grouped",
    ]
    keep_columns += ["genre_" + g.lower().replace(" ", "_") for g in TOP_GENRES]

    if "revenue" in df.columns:
        keep_columns.append("revenue")

    return df[keep_columns]


# Feature groups for ablation analysis — each group can be excluded independently
# to measure its contribution to model performance.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "budget": ("budget", "budget_log", "budget_missing", "budget_per_minute"),
    "popularity_runtime": ("popularity", "runtime"),
    "temporal": (
        "release_year", "release_month", "release_quarter",
        "release_dayofweek", "release_decade",
        "is_summer_release", "is_holiday_release",
    ),
    "counts": (
        "n_genres", "n_production_companies", "n_production_countries",
        "n_spoken_languages", "n_keywords", "n_cast", "n_crew",
        "cast_n_male", "cast_n_female", "cast_n_unknown_gender",
    ),
    "presence_flags": (
        "has_collection", "has_homepage", "has_tagline", "has_poster",
        "is_english", "title_differs_from_original",
    ),
    "text_lengths": (
        "overview_len", "title_len", "tagline_len", "original_title_len",
    ),
    "genre_flags": tuple(
        "genre_" + g.lower().replace(" ", "_") for g in TOP_GENRES
    ),
    "language": ("lang_grouped",),
}


def get_all_feature_columns() -> list[str]:
    """Flat list of every engineered column (matches build_features output)."""
    return [c for cols in FEATURE_GROUPS.values() for c in cols]
