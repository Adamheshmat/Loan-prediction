import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
)

# --- Load data ---
CSV_PATH = 'Project/Cleaned_Loan_Train_Data.csv'
df = pd.read_csv(CSV_PATH)

# Features/target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train/test split (25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --- Pipeline (scaling → PCA → KNN) ---
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),                      # NEW: PCA improves KNN accuracy
    ('knn', KNeighborsClassifier())
])

# --- Hyperparameter grid ---
param_grid = {
    'pca__n_components': [0.90, 0.95, 0.99],    # keep 90–99% information
    'knn__n_neighbors': list(range(1, 40)),     # tune k from 1..40
    'knn__weights': ['uniform', 'distance'],    # weight types
    'knn__metric': ['euclidean', 'manhattan'],  # best metrics for KNN
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='accuracy',                 # accuracy target
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\nBest hyperparameters:", grid.best_params_)
print("Best CV Accuracy: {:.4f}".format(grid.best_score_))

# --- Evaluate best estimator on test set ---
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1_macro_test = f1_score(y_test, y_pred, average='macro')

print("\nTest Accuracy: {:.4f}".format(acc))
print("Test F1 (macro): {:.4f}".format(f1_macro_test))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.named_steps['knn'].classes_)
fig, ax = plt.subplots(figsize=(5,4))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Confusion Matrix (Test set)')
plt.tight_layout()
plt.show()