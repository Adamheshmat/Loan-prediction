# ======================================================
# 1. IMPORTS
# ======================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# 2. READ YOUR ALREADY CLEANED FILE
# ======================================================
df = pd.read_csv("D:\DataMining\projectphase2\Loan-prediction\Project\Cleaned_Loan_Train_Data.csv")

print("\nDATA LOADED. SHAPE:", df.shape)
print(df.head())

# Target variable
y = df["Loan_Status"]
X = df.drop(columns=["Loan_Status"])

# ======================================================
# 3. IDENTIFY NUMERIC FEATURES FOR DISCRETIZATION
# ======================================================
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nNUMERIC COLUMNS TO DISCRETIZE:")
print(num_cols)

# ======================================================
# 4. TRAIN/TEST SPLIT
# ======================================================
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================================================
# 5. AUTO BIN SELECTION
# ======================================================
bin_options = [1,2,3,4,5,6, 7, 8]
bin_results = {}

for n_bins in bin_options:
    try:
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")

        X_train = X_train_orig.copy()
        X_test = X_test_orig.copy()

        X_train[num_cols] = kb.fit_transform(X_train[num_cols])
        X_test[num_cols] = kb.transform(X_test[num_cols])

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        bin_results[n_bins] = acc
    except:
        bin_results[n_bins] = "FAIL"

print("\n=== BIN ACCURACY RESULTS ===")
print(bin_results)

# SELECT BEST BIN COUNT
valid_results = {b: a for b, a in bin_results.items() if a != "FAIL"}
best_bins = max(valid_results, key=valid_results.get)
print(f"\nBEST NUMBER OF BINS = {best_bins}")

# ======================================================
# 6. APPLY FINAL DISCRETIZATION USING BEST n_bins
# ======================================================
kbins = KBinsDiscretizer(n_bins=best_bins, encode="ordinal", strategy="uniform")

X_train = X_train_orig.copy()
X_test = X_test_orig.copy()

X_train[num_cols] = kbins.fit_transform(X_train[num_cols])
X_test[num_cols] = kbins.transform(X_test[num_cols])

# ======================================================
# 7. DECISION TREE + GRIDSEARCHCV
# ======================================================
params = {
    "criterion": ["entropy"],
    "max_depth": [4, 5, 6],
    "min_samples_split": [2, 4, 6,10],
    "min_samples_leaf": [1, 2, 4]
}

tree = DecisionTreeClassifier(random_state=42)
grid = GridSearchCV(tree, params, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

best_tree = grid.best_estimator_
print("\nBEST TREE PARAMETERS:", grid.best_params_)

# ======================================================
# 8. FINAL MODEL EVALUATION
# ======================================================
y_pred = best_tree.predict(X_test)

print("\nTRAIN ACC:", accuracy_score(y_train, best_tree.predict(X_train)))
print("TEST ACC:", accuracy_score(y_test, y_pred))
print("\nCLASSIFICATION REPORT:\n", classification_report(y_test, y_pred))
print("CONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred))

# ======================================================
# 9. FEATURE IMPORTANCE
# ======================================================
importances = best_tree.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,5))
sns.barplot(x=X_train.columns[indices], y=importances[indices])
plt.xticks(rotation=75)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# ======================================================
# 10. TREE VISUALIZATION
# ======================================================
plt.figure(figsize=(22,10))
plot_tree(
    best_tree,
    feature_names=X_train.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.show()
