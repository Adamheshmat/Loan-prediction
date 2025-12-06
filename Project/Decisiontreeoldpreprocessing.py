 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.tree import export_graphviz
import graphviz

 
df = pd.read_csv("Project/Cleaned_Loan_Train_Data.csv")

print("\nDATA LOADED. SHAPE:", df.shape)
print(df.head())

 
y = df["Loan_Status"]
X = df.drop(columns=["Loan_Status"])

 
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nNUMERIC COLUMNS TO DISCRETIZE:")
print(num_cols)

 
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

 
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

valid_results = {b: a for b, a in bin_results.items() if a != "FAIL"}
valid_results = {b: a for b, a in bin_results.items() if a != "FAIL"}

if len(valid_results) == 0:
    raise ValueError("No valid bin results — KBins failed on all options.")

best_bins = max(valid_results, key=lambda b: valid_results[b])

print(f"\nBEST NUMBER OF BINS = {best_bins}")


 
kbins = KBinsDiscretizer(n_bins=best_bins, encode="ordinal", strategy="uniform")

X_train = X_train_orig.copy()
X_test = X_test_orig.copy()

X_train[num_cols] = kbins.fit_transform(X_train[num_cols])
X_test[num_cols] = kbins.transform(X_test[num_cols])

 
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

 
y_pred = best_tree.predict(X_test)

print("\nTRAIN ACC:", accuracy_score(y_train, best_tree.predict(X_train)))
print("TEST ACC:", accuracy_score(y_test, y_pred))
print("\nCLASSIFICATION REPORT:\n", classification_report(y_test, y_pred))
print("CONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred))
# ===========================================
# CONFUSION MATRIX PLOT WITH METRICS (NO RECALCULATION)
# ===========================================
cm = confusion_matrix(y_test, y_pred)

# Metrics using already computed predictions
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)   # recall for class 1

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)                 # recall for class 0

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Display metrics on the right
metrics_text = (
    f"Accuracy:    {accuracy:.3f}\n"
    f"Precision:   {precision:.3f}\n"
    f"Sensitivity: {sensitivity:.3f}\n"
    f"Specificity: {specificity:.3f}"
)
plt.text(2.3, 0.3, metrics_text, fontsize=12, va="center")

plt.tight_layout()
plt.show()

 

 
importances = best_tree.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,5))
sns.barplot(x=X_train.columns[indices], y=importances[indices])
plt.xticks(rotation=75)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()


dot_data = export_graphviz(
    best_tree,
    out_file=None,
    feature_names=X_train.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    special_characters=True,
    precision=2
)

graph = graphviz.Source(dot_data)
graph.render("decision_tree_full", format="png", cleanup=True)



 
 
new_df_original = pd.read_csv("Project/Cleaned_New_Customer.csv")

 
new_df_model = new_df_original.copy()

 
missing_cols = set(X_train_orig.columns) - set(new_df_model.columns)
for col in missing_cols:
    new_df_model[col] = 0  

new_df_model = new_df_model[X_train_orig.columns]

 
new_df_model[num_cols] = kbins.transform(new_df_model[num_cols])

 
cat_cols = new_df_model.select_dtypes(include=['object', 'bool']).columns
for col in cat_cols:
    new_df_model[col] = new_df_model[col].astype(int)
 
predictions = best_tree.predict(new_df_model)

 
new_df_original["Predicted_Loan_Status"] = predictions

 
output_path = "Project/New_Customer_Predicted.csv"
new_df_original.to_csv(output_path, index=False)

print(f"\nPredictions saved to: {output_path}")
print(new_df_original.head())
print(new_df_original['Predicted_Loan_Status'].value_counts())
 

binned_output = new_df_model.copy()
binned_output["Predicted_Loan_Status"] = predictions

binned_output_path = "Project/New_Customer_Predicted_WITH_BINS.csv"
binned_output.to_csv(binned_output_path, index=False)

print("\n====================================================")
print("Binned prediction file saved!")
print("Location:", binned_output_path)
print("Shape:", binned_output.shape)
print("Preview:")
print(binned_output.head())
print("====================================================")

 

subset = new_df_original[
    (new_df_original["Married"] != 0) &
    (new_df_original["Property_Area_Semiurban"] != 0)
]

total = len(subset)
approved = subset["Predicted_Loan_Status"].sum()
percentage = (approved / total) * 100 if total > 0 else 0

print("\n" + "-"*60)
print("MARRIED + SEMI-URBAN CUSTOMERS ANALYSIS")
print("-"*60)
print(f"Total customers (Married + Semiurban): {total}")
print(f"Predicted Approved                  : {approved}")
print(f"Approval Percentage                : {percentage:.2f}%")
print("-"*60)
 

plt.figure(figsize=(6, 4))
sns.countplot(x="Predicted_Loan_Status", data=new_df_original, palette="Blues")
plt.title("Overall Loan Approval Predictions - New Customers")
plt.xlabel("Predicted Loan Status (0=No, 1=Yes)")
plt.show()

if total > 0:
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Predicted_Loan_Status", data=subset, palette="Greens")
    plt.title("Predictions for Married + Semiurban Customers")
    plt.xlabel("Predicted Loan Status")
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.pie(
        [approved, total - approved],
        labels=["Approved", "Rejected"],
        autopct="%.1f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90
    )
    plt.title(f"Approval Rate for Married + Semiurban Customers\n({percentage:.1f}%)")
    plt.show()
else:
    print("No Married + Semiurban customers → skipping pie chart.")
