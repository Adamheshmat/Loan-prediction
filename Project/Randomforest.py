import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score
)
 
df = pd.read_csv("Project/Cleaned_Loan_Train_Data.csv")

if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)  # recall for class 1
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
 
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
ax.set_title("Random Forest Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

metrics_text = (
    f"Accuracy:    {accuracy:.3f}\n"
    f"Precision:   {precision:.3f}\n"
    f"Sensitivity: {sensitivity:.3f}\n"
    f"Specificity: {specificity:.3f}"
)

plt.text(2.3, 0.2, metrics_text, fontsize=12, va='center')
plt.tight_layout()
plt.show()
 
importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(importances)

plt.figure(figsize=(10, 6))
plt.barh(importances["Feature"], importances["Importance"])
plt.title("Random Forest Feature Importance")
plt.xlabel("Gini Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
