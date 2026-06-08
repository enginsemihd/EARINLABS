# ==========================================
# 1. IMPORTS & SETUP
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

import warnings
warnings.filterwarnings('ignore')

# Ensuring project reproducibility
SEED = 42
np.random.seed(SEED)

def main():
    # ==========================================
    # 2. DATA LOADING & FEATURE ENGINEERING
    # ==========================================
    print("Loading the full TMDB dataset... 🚀")
    df = pd.read_csv('train.csv')

    # Drop missing revenues and filter out zero values (crucial to prevent infinite MAPE)
    df = df.dropna(subset=['revenue'])
    df = df[df['revenue'] > 10000]

    # Feature Engineering: Extracting temporal data
    # Hata almamak için errors='coerce' ekliyoruz, boş tarihleri NaT (Not a Time) yapar
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce') 
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month

    # Expanded Feature Set (Vote sütunları çıkarıldı, 6 güçlü özellik kullanıyoruz)
    features_to_use = [
        'budget', 'runtime', 'popularity', 
        'release_year', 'release_month', 'original_language'
    ]
    
    X = df[features_to_use]
    y = df['revenue']

    # ==========================================
    # 3. PREPROCESSING PIPELINE
    # ==========================================
    numeric_features = ['budget', 'runtime', 'popularity', 'release_year', 'release_month']
    categorical_features = ['original_language']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Eksik yılları/ayları medyan ile doldurur
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 80/20 Split on the FULL dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # ==========================================
    # 4. MODEL TRAINING (XGBOOST)
    # ==========================================
    print("Training Optimized XGBoost Regressor...")
    
    # XGBoost handles non-linear relationships and outliers much better than baseline models
    model = XGBRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=6, 
        random_state=SEED, 
        n_jobs=-1
    )

    clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    # ==========================================
    # 5. METRICS & EVALUATION
    # ==========================================
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("\n--- 📊 Final Model Performance ---")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R^2 Score: {r2:.4f}")
    print(f"  MAPE: {mape * 100:.2f}%\n")

    # ==========================================
    # 6. VISUALIZATION
    # ==========================================
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='#2ca02c')
    
    # Perfect prediction baseline
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], '--r', linewidth=2, label="Perfect Prediction")
    
    plt.title('Final Model (XGBoost): Actual vs. Predicted Revenue', fontsize=14, pad=10)
    plt.xlabel('Actual Revenue ($)', fontsize=12)
    plt.ylabel('Predicted Revenue ($)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plot_filename = 'final_results_plot.jpg'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot successfully saved as '{plot_filename}'.")

if __name__ == "__main__":
    main()