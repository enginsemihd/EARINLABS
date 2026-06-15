# Project 22 — Movie Revenue Prediction

A comparative study of six regression algorithms (Linear, Ridge, Random Forest,
XGBoost, LightGBM, CatBoost) on the TMDB Box Office Prediction dataset.

EARIN, Summer 2026 — Engin S. Demirkan & Franziska Axmann.

---

## Project structure

```
movie_revenue_prediction/
├── main.py                      # Orchestrator: load → engineer → train 6 models → CV → ablation → plots
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py   # 49 features from raw TMDB columns (parses JSON cast/crew/genres/...)
│   ├── models.py                # Pipeline factories for all 6 algorithms (shared preprocessor)
│   └── evaluation.py            # Metrics (RMSE, R², RMSLE, MAPE, MedAPE) and 8 plot functions
├── plots/                       # 8 PNG figures (auto-generated)
├── results/                     # model_metrics.csv, ablation_study.csv, summary.json
├── report.tex                   # Final scientific report (LaTeX source)
├── report.pdf                   # Compiled report (9 pages)
├── requirements.txt             # Pinned Python dependencies
├── train.csv                    # TMDB training data (3,000 films, 23 columns)
└── README.md                    # This file
```

## How to run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure `train.csv` (from
   [the Kaggle competition](https://www.kaggle.com/c/tmdb-box-office-prediction))
   is in the project root.

3. Run the full pipeline:
   ```bash
   python main.py
   ```
   This trains all six models, runs 5-fold cross-validation, performs an
   ablation study, and saves every plot and CSV used in the report.

4. (Optional) Recompile the report:
   ```bash
   pdflatex report.tex && pdflatex report.tex
   ```
   Two passes are needed so that cross-references and citations resolve.

The pipeline takes about 30 seconds on a laptop CPU. All randomness is seeded
with `RANDOM_SEED = 42`, so the numbers in the report are exactly reproducible.

## Headline results

| Model              | Hold-out R² | RMSLE | MedAPE | CV-RMSLE        |
|--------------------|------------:|------:|-------:|----------------:|
| Linear Regression  |       0.199 | 1.669 |  69.8% | 1.777 ± 0.066   |
| Ridge Regression   |       0.182 | 1.670 |  69.4% | 1.776 ± 0.065   |
| Random Forest      |   **0.709** | 1.577 |  66.5% | 1.614 ± 0.032   |
| XGBoost            |       0.709 | 1.569 |  65.5% | 1.614 ± 0.044   |
| LightGBM           |       0.656 | 1.582 |  65.3% | 1.642 ± 0.048   |
| **CatBoost**       |       0.670 | **1.539** | 65.4% | **1.584 ± 0.029** |

- Best by primary metric (CV-RMSLE): **CatBoost**.
- Best by hold-out R²: Random Forest (tied with XGBoost).
- Improvement vs. midterm baseline (single XGBoost, six features): R²
  0.6352 → 0.709, MAPE 2{,}096% → 547–612%.

See `report.pdf` for the full discussion, ablation study, residual analysis,
and feature-importance results.

## Why MAPE is reported alongside MedAPE

MAPE is the metric requested in the project brief but is unstable on
heavy-tailed positive targets (a single $50k film mispredicted at $1M
contributes 1,900% to the mean). We additionally report **MedAPE** (median
absolute percentage error), which is the standard robust alternative and
represents the typical relative error of a film rather than the mean
contaminated by outliers. We treat **RMSLE** — the official Kaggle metric for
this dataset — as the primary metric.
