import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"D:\DataMining\projectphase2\Loan-prediction\Project\Cleaned_Loan_Train_Data.csv")

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 7, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 6, 10],
    "class_weight": [None, "balanced"]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("Best Parameters:", grid.best_params_)
print("Improved Accuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

TN, FP, FN, TP = cm.ravel()

accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = (FP + FN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
precision = TP / (TP + FP)

print("Accuracy:", accuracy)
print("Error rate:", error_rate)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Precision:", precision)

importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,5))
plt.title("Feature Importance (Decision Tree)")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), X.columns[indices], rotation=90)
plt.show()

new_customer = pd.DataFrame([{
    "Gender": 1,
    "Married": 1,
    "Dependents": 0,
    "Education": 0,
    "Self_Employed": 0,
    "ApplicantIncome": 0.45,
    "CoapplicantIncome": 0.10,
    "LoanAmount": 0.30,
    "Loan_Amount_Term": 0.85,
    "Credit_History": 1,
    "Property_Area_Semiurban": 1,
    "Property_Area_Urban": 0
}])

prediction = best_model.predict(new_customer)[0]
print("Loan Prediction:", "Approved" if prediction == 1 else "Rejected")
plt.figure(figsize=(22,12))
plot_tree(
    best_model,
    feature_names=X.columns,
    class_names=["No Loan", "Loan"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.show()
