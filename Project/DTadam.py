 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

 

df = pd.read_csv("Project/Cleaned_Loan_Train_Data_NoScaling.csv")

 
numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

 

kbins = KBinsDiscretizer(
    n_bins=6,
    encode="ordinal",
    strategy="kmeans"   
)

disc_values = kbins.fit_transform(df[numeric_cols])

 
for i, col in enumerate(numeric_cols):
    df[col + "_bin"] = disc_values[:, i]

 
feature_cols = [
    "ApplicantIncome_bin", "CoapplicantIncome_bin",
    "LoanAmount_bin", "Loan_Amount_Term_bin",
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Credit_History",
    "Property_Area_Semiurban", "Property_Area_Urban"
]

X = df[feature_cols]
y = df["Loan_Status"]

 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

 
param_grid = {
    "criterion": ["entropy"],
    "max_depth": [3, 4, 5, 6, 8, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_tree = grid.best_estimator_

print("\nBEST PARAMETERS:", grid.best_params_)
print("Training Accuracy:", best_tree.score(X_train, y_train))
print("Testing Accuracy:", best_tree.score(X_test, y_test))

 

y_pred = best_tree.predict(X_test)

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
 
 

importances = pd.Series(best_tree.feature_importances_, index=feature_cols)
print("\nFEATURE IMPORTANCE:")
print(importances.sort_values(ascending=False))

 

plt.figure(figsize=(24, 14))
plot_tree(
    best_tree,
    feature_names=feature_cols,
    class_names=["No Loan", "Loan"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.show()
