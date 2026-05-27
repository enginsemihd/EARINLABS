# ==========================================
# 1. IMPORTS & SETUP
# ==========================================
# Let's bring in the tools we need for data wrangling, modeling, and plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Hide annoying warnings to keep our console output looking clean
import warnings
warnings.filterwarnings('ignore')

# Setting the seed ensures we get the exact same results every time we run this.
# This is crucial for the reproducibility requirement in your midterm report!
SEED = 42
np.random.seed(SEED)

# ==========================================
# 2. DATA LOADING & CLEANING
# ==========================================
print("Loading up the Kaggle dataset... 🚀")

# Make sure 'train.csv' is in the same folder as this script
df = pd.read_csv('train.csv')

# Let's grab the core features we promised to analyze in the midterm report
features_to_use = ['budget', 'runtime', 'original_language', 'revenue']
df = df[features_to_use]

# If a movie doesn't have box office revenue data, we can't learn from it. 
# Let's drop those rows to keep our training clean.
df = df.dropna(subset=['revenue'])

# ==========================================
# 3. PREPROCESSING PIPELINE
# ==========================================
X = df.drop('revenue', axis=1)
y = df['revenue']

# Splitting our features by type so we can treat them differently
numeric_features = ['budget', 'runtime']
categorical_features = ['original_language']

# For numbers: Fill missing values with the median, then scale them so they play nicely together
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# For text/categories: Fill gaps with a placeholder, then convert to one-hot vectors
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine the two pipelines into one master preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data: 80% for training the model, 20% for testing it (using our trusty seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# ==========================================
# 4. MODEL TRAINING & EVALUATION
# ==========================================
# Setting up our baseline and our advanced tree model
models = {
    "Ridge Regression (Baseline)": Ridge(alpha=1.0, random_state=SEED),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
}

results = {}

print("\n--- 📊 Training Complete! Here are the Midterm Results ---")

for name, model in models.items():
    # Build a pipeline that automatically preprocesses the data right before feeding it to the model
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
    
    # Train the model!
    clf.fit(X_train, y_train)
    
    # Let's see how it performs on the 20% test data it hasn't seen yet
    y_pred = clf.predict(X_test)
    
    # Calculate our core metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Save the results for plotting later
    results[name] = {'RMSE': rmse, 'R2': r2, 'Predictions': y_pred}
    
    # Print the stats in a nice, readable format
    print(f"[{name}]")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R^2 Score: {r2:.4f}\n")

# ==========================================
# 5. VISUALIZATION (FOR THE REPORT)
# ==========================================
# Let's draw the scatter plots showing Actual vs. Predicted revenue
plt.figure(figsize=(14, 6))

for i, (name, metrics) in enumerate(results.items(), 1):
    plt.subplot(1, 2, i)
    sns.scatterplot(x=y_test, y=metrics['Predictions'], alpha=0.5, color='#1f77b4', edgecolor=None)
    
    # Draw the perfect prediction line (y=x) in red
    max_val = max(y_test.max(), metrics['Predictions'].max())
    plt.plot([0, max_val], [0, max_val], '--r', linewidth=2, label="Perfect Prediction")
    
    # Make it look professional
    plt.title(f'{name}\nActual vs. Predicted Revenue', fontsize=12, pad=10)
    plt.xlabel('Actual Revenue ($)', fontsize=10)
    plt.ylabel('Predicted Revenue ($)', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

# Save it specifically as a .jpg to match the LaTeX code we wrote earlier
plot_filename = 'midterm_results_plot.jpg'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

print(f"All done! 🎉 I've saved the plot as '{plot_filename}' in your folder.")
print("You can pop that straight into your LaTeX report now.")
plt.show()