import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn modülleri
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. TEKRARLANABİLİRLİK (REPRODUCIBILITY)
# Raporda istenen "seed" ayarı
SEED = 42
np.random.seed(SEED)

# ==========================================
# 2. VERİ YÜKLEME (DATA LOADING)
# ==========================================
# İndirdiğiniz dosyanın adını buraya yazın (Örn: train.csv)
df = pd.read_csv('train.csv')

# Projede kullanılacak sütunları seçin (Eğer CSV'deki isimler farklıysa güncelleyin)
# Örneğin TMDB veri setinde hedef değişken genellikle 'revenue' sütunudur.
df = df[['budget', 'runtime', 'original_language', 'revenue']]

# Eksik verileri hızlıca temizlemek isterseniz:
df = df.dropna(subset=['revenue']) # Hedef değişkeni boş olanları at
# ==========================================
# 3. VERİ ÖN İŞLEME VE ANALİZ (PREPROCESSING)
# ==========================================
print("--- Veri Seti Özeti ---")
print(df.info())

# Özellikler (Features) ve Hedef Değişken (Target) Ayrımı
X = df.drop('revenue', axis=1)
y = df['revenue']

# Sayısal ve Kategorik Değişkenleri Belirleme
numeric_features = ['budget', 'runtime']
categorical_features = ['original_language']

# Pipeline Kurulumu (Eksik veri doldurma, Ölçeklendirme ve Encoding)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Eğitim ve Test setlerine ayırma (Seed kullanarak)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# ==========================================
# 4. MODELLEME (MODELING)
# ==========================================
models = {
    "Ridge Regression (Baseline)": Ridge(alpha=1.0, random_state=SEED),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=SEED)
}

results = {}

print("\n--- Model Eğitimi ve Değerlendirmesi ---")
for name, model in models.items():
    # Pipeline'a modeli ekle
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
    
    # Modeli Eğit
    clf.fit(X_train, y_train)
    
    # Tahmin Yap
    y_pred = clf.predict(X_test)
    
    # Metrikleri Hesapla
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'RMSE': rmse, 'R2': r2, 'Predictions': y_pred}
    
    print(f"[{name}]")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R^2 Score: {r2:.4f}\n")

# ==========================================
# 5. GÖRSELLEŞTİRME (VISUALIZATION)
# ==========================================
# Rapor için Gerçek vs. Tahmin (Actual vs. Predicted) Dağılım Grafiği (Scatter Plot)
plt.figure(figsize=(14, 6))

for i, (name, metrics) in enumerate(results.items(), 1):
    plt.subplot(1, 2, i)
    sns.scatterplot(x=y_test, y=metrics['Predictions'], alpha=0.6)
    
    # Mükemmel tahmin çizgisi (y=x)
    max_val = max(y_test.max(), metrics['Predictions'].max())
    plt.plot([0, max_val], [0, max_val], '--r', linewidth=2)
    
    plt.title(f'{name}\nActual vs Predicted Revenue')
    plt.xlabel('Actual Revenue ($)')
    plt.ylabel('Predicted Revenue ($)')
    plt.ticklabel_format(style='plain', axis='both') # Bilimsel gösterimi kapat (e+07 yerine normal sayılar)

plt.tight_layout()
plt.savefig('midterm_results_plot.png', dpi=300) # Grafiği rapor için kaydet
print("Grafik 'midterm_results_plot.png' olarak kaydedildi. Raporunuza ekleyebilirsiniz.")
plt.show()