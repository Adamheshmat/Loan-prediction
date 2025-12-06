import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

 
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
 
param_grid = {
    "var_smoothing": np.logspace(-12, -6, 20)   # search 20 values
}

nb = GaussianNB()

grid = GridSearchCV(
    estimator=nb,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBEST PARAMETERS FOUND BY GRID SEARCH:")
print(grid.best_params_)

 
nb_model = grid.best_estimator_
 
y_train_pred = nb_model.predict(X_train)
y_pred = nb_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

precision = precision_score(y_test, y_pred)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("\n========= MODEL PERFORMANCE =========")
print("Train Accuracy:", train_accuracy)
print("Test Accuracy :", test_accuracy)
print("Precision     :", precision)
print("Sensitivity   :", sensitivity)
print("Specificity   :", specificity)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
 
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

print("\n======== FEATURE INFLUENCE ========")
print(feature_influence)

plt.figure(figsize=(12,6))
plt.bar(feature_influence["Feature"], feature_influence["Influence_Score"])
plt.xticks(rotation=90)
plt.title("Feature Influence Scores (Gaussian Naive Bayes)")
plt.tight_layout()
plt.show()

 
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title("Confusion Matrix (Gaussian Naive Bayes)")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

plt.text(2.5, 0.2,
         f"Accuracy:      {test_accuracy:.3f}\n"
         f"Precision:     {precision:.3f}\n"
         f"Sensitivity:   {sensitivity:.3f}\n"
         f"Specificity:   {specificity:.3f}\n"
         f"Train Acc:     {train_accuracy:.3f}",
         fontsize=12, va='center')

plt.tight_layout()
plt.show()

 