import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset

data = pd.read_csv('train.csv')

# 1. Data Preparation
# Dropping columns that are useless for the prediction or have too many missing values
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Filling missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Fare'] = data['Fare'].fillna(data['Fare'].median()) # just in case

# Convert categorical variables to numbers using pandas get_dummies
data = pd.get_dummies(data, columns=['Pclass', 'Sex', 'Embarked'], drop_first=True)

y = data['Survived']
X = data.drop('Survived', axis=1)

# 2. Data Split
# 80/20 split as mentioned in the instructions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalization step
# Scaling only the numerical features, not the dummy ones
scaler = StandardScaler()
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("--- Starting Model Training ---")

# 3 & 4. Model Definition and Training (with 4-fold CV)

# --- MODEL 1: Logistic Regression ---
print("\nTraining Logistic Regression...")
log_reg = LogisticRegression(max_iter=500)

# Try different parameters
params_lr = {
    'C': [0.1, 1.0, 10.0], 
    'solver': ['liblinear', 'lbfgs']
}

# 4 folds cross validation
grid_lr = GridSearchCV(log_reg, params_lr, cv=4) 
grid_lr.fit(X_train, y_train)

print("Best LR parameters found:", grid_lr.best_params_)
lr_preds = grid_lr.predict(X_test)
print("Logistic Regression Accuracy:", round(accuracy_score(y_test, lr_preds), 4))


# --- MODEL 2: Random Forest ---
print("\nTraining Random Forest...")
rf = RandomForestClassifier(random_state=42)

params_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(rf, params_rf, cv=4)
grid_rf.fit(X_train, y_train)

print("Best RF parameters found:", grid_rf.best_params_)
rf_preds = grid_rf.predict(X_test)
print("Random Forest Accuracy:", round(accuracy_score(y_test, rf_preds), 4))

print("\nDetailed Report for the best model (Random Forest):")
print(classification_report(y_test, rf_preds))

# --- Graph----
print("\nGenerating Graph for Random Forest Confusion Matrix...")

# Calculate confusion matrix 
cm = confusion_matrix(y_test, rf_preds)

# Draw the confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Deceased (0)', 'Survived (1)'])
disp.plot(cmap='Blues', values_format='d')

plt.title('Random Forest - Confusion Matrix')

# Save the figure with high resolution
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Graph successfully saved as 'confusion_matrix.png'!")