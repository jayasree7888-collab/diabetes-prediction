import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("diabetes.csv")
print("--- First 5 Rows ---")
print(df.head())
print("\n--- Data Info ---")
print(df.info())


# columns where 0 is an impossible value
cols_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_clean] = df[cols_to_clean].replace(0, np.nan)
imputer = SimpleImputer(strategy='median')
df[cols_to_clean] = imputer.fit_transform(df[cols_to_clean])
print("\n--- Info After Handling Zeros ---")
print(df.info())
print("\n--- Data Description After Handling Zeros ---")
print(df.describe())
X = df.drop('Outcome', axis=1) 
y = df['Outcome']              
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("\n--- Data Splitting ---")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

#logistic regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
print("\n--- Training Logistic Regression ---")
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}")
print(f"Logistic Regression ROC-AUC Score: {roc_auc_lr:.4f}")
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))

#random forest
print("\n--- Training Random Forest ---")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print(f"Random Forest ROC-AUC Score: {roc_auc_rf:.4f}")
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

#support vector machine
print("\n--- Training Support Vector Machine (SVM) ---")
svm = SVC(random_state=42, probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1])
print(f"SVM Accuracy: {accuracy_svm:.4f}")
print(f"SVM ROC-AUC Score: {roc_auc_svm:.4f}")
print("\nClassification Report (SVM):")
print(classification_report(y_test, y_pred_svm))

# Hyperparameter tuning for Random Forest using GridSearchCV

print("\n--- Tuning Random Forest with GridSearchCV ---")
param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [None, 10, 20],      
    'min_samples_leaf': [1, 2, 4]      
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, scoring='roc_auc', n_jobs=-1)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"Best Parameters found: {grid_search.best_params_}")
y_pred_best_rf = best_rf.predict(X_test)
roc_auc_best_rf = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])
print(f"\nOriginal Random Forest ROC-AUC: {roc_auc_rf:.4f}")
print(f"TUNED Random Forest ROC-AUC:    {roc_auc_best_rf:.4f}")
print("\nClassification Report (Tuned Random Forest):")
print(classification_report(y_test, y_pred_best_rf))