🩺 Diabetes Prediction using Machine Learning

This project builds a machine learning pipeline to predict the likelihood of an individual having diabetes based on 8 key medical diagnostic features.
It uses the Pima Indians Diabetes Dataset from Kaggle
 / UCI Machine Learning Repository.

📊 Project Overview

This project demonstrates a complete end-to-end machine learning workflow, including:

Data Cleaning

Handling impossible zero-values (e.g., Glucose, BMI) by replacing them with the median.

Data Preprocessing

Scaling features using StandardScaler for optimal model performance.

Model Training

Implementing and comparing three classification models:

Logistic Regression

Support Vector Machine (SVM)

Random Forest

Model Evaluation

Evaluating models using Accuracy and ROC-AUC Score.

Model Tuning

Using GridSearchCV to find the best hyperparameters for the top-performing model (Random Forest).

💻 Technologies Used

Python 3

pandas → Data loading and manipulation

NumPy → Numerical operations and handling NaN values

scikit-learn → For:

SimpleImputer → Data cleaning

StandardScaler → Feature scaling

train_test_split → Data splitting

LogisticRegression, SVC, RandomForestClassifier → Models

accuracy_score, roc_auc_score, classification_report → Evaluation

GridSearchCV → Hyperparameter tuning

🚀 How to Run

Clone the repository or download the files.

Install dependencies:

pip install pandas numpy scikit-learn


Place the dataset (diabetes.csv) in the same directory as the script.

Run the script:

python diabetes_prediction.py


The script will execute all steps — from data loading and cleaning to model training and tuning — and print the evaluation results in the terminal.

📈 Results
Model	Accuracy	ROC-AUC Score
Logistic Regression	0.7532	0.8230
Random Forest	0.7403	0.8334
SVM	0.7468	0.8072

The Random Forest model achieved the best baseline ROC-AUC score.

🔧 After Hyperparameter Tuning

Tuned Random Forest ROC-AUC: 0.8404

Best Parameters Found:

{'max_depth': 10, 'min_samples_leaf': 4, 'n_estimators': 300}

🧠 Key Takeaway

This project highlights the effectiveness of data preprocessing, model comparison, and hyperparameter tuning in improving model performance for medical diagnostic predictions.