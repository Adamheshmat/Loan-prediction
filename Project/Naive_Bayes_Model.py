import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv("Project/Cleaned_Loan_Train_Data.csv")

df["Property_Area_Rural"] = (
    (df["Property_Area_Semiurban"] == 0) &
    (df["Property_Area_Urban"] == 0)
).astype(int)

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

feature_influence = pd.DataFrame({
    "Feature": X.columns,
    "Mean_Class_0": nb_model.theta_[0],
    "Mean_Class_1": nb_model.theta_[1],
    "Variance_Class_0": nb_model.var_[0],
    "Variance_Class_1": nb_model.var_[1],
})

feature_influence["Mean_Difference"] = abs(
    feature_influence["Mean_Class_1"] - feature_influence["Mean_Class_0"]
)

feature_influence["Avg_Variance"] = (
    feature_influence["Variance_Class_0"] + feature_influence["Variance_Class_1"]
) / 2

feature_influence["Influence_Score"] = (
    feature_influence["Mean_Difference"] / feature_influence["Avg_Variance"]
)

feature_influence = feature_influence.sort_values(
    by="Influence_Score", ascending=False
)

print(feature_influence)

plt.figure(figsize=(12,6))
plt.bar(feature_influence["Feature"], feature_influence["Influence_Score"])
plt.xticks(rotation=90)
plt.title("Feature Influence Scores (Gaussian Naive Bayes)")
plt.tight_layout()
plt.show()
