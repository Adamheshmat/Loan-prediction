import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.inspection import permutation_importance

 
CSV_PATH = 'Project/Cleaned_Loan_Train_Data.csv'
df = pd.read_csv(CSV_PATH)

 
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

 
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('knn', KNeighborsClassifier())
])

 
param_grid = {
    'pca__n_components': [0.90, 0.95, 0.99],
    'knn__n_neighbors': list(range(1, 40)),
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan'],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\nBest hyperparameters:", grid.best_params_)
print("Best CV Accuracy: {:.4f}".format(grid.best_score_))

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall_sensitivity = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print("\nTest Accuracy: {:.4f}".format(acc))
print("Test F1 (macro): {:.4f}".format(f1))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
 
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix (KNN Best Model)")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

 
textstr = (
    f"Accuracy:     {acc:.3f}\n"
    f"Precision:    {prec:.3f}\n"
    f"Sensitivity:  {recall_sensitivity:.3f}\n"
    f"Specificity:  {specificity:.3f}\n"
)

plt.text(2.3, 0.2, textstr, fontsize=12, va='center')

plt.tight_layout()
plt.show()
 
result = permutation_importance(
    best_model, X_test, y_test,
    scoring="accuracy",
    n_repeats=15,
    random_state=42,
    n_jobs=-1
)

# Build DataFrame
importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": result["importances_mean"]
})

 
importances_abs = importances.copy()
importances_abs["Importance"] = importances_abs["Importance"].abs()

 
importances_abs = importances_abs.sort_values("Importance", ascending=True)

print("\nAbsolute Permutation Feature Importance:\n")
print(importances_abs)

 
plt.figure(figsize=(8, 6))
plt.barh(importances_abs["Feature"], importances_abs["Importance"])
plt.title("Permutation Feature Importance")
plt.xlabel("Absolute Mean Change in Accuracy")
plt.tight_layout()
plt.show()