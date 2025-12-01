import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
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

y_prob = nb_model.predict_proba(X_test)[:, 1]

 

thresholds = np.arange(0.3, 0.8, 0.01)
best_score = -1
best_t = float(thresholds[0])
best_metrics = (0,0,0,0,0,0)

sens_list = []
spec_list = []
thr_list = []

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    score = 1 - abs(sensitivity - specificity)

    sens_list.append(sensitivity)
    spec_list.append(specificity)
    thr_list.append(float(t))

    if score > best_score:
        best_score = score
        best_t = float(t)
        best_metrics = (tn, fp, fn, tp, sensitivity, specificity)

 

tn, fp, fn, tp, sensitivity, specificity = best_metrics
y_pred_best = (y_prob >= best_t).astype(int)

accuracy  = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best)

print("Best Threshold:", best_t)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

 

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



plt.figure(figsize=(10,6))
plt.plot(thr_list, sens_list, label="Sensitivity", marker='o')
plt.plot(thr_list, spec_list, label="Specificity", marker='o')
plt.axvline(float(best_t), color="red", linestyle="--", label=f"Best Threshold = {best_t:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Sensitivity vs Specificity Across Thresholds")
plt.legend()
plt.grid(True)
plt.show()

 

plt.figure(figsize=(12,6))
plt.bar(feature_influence["Feature"], feature_influence["Influence_Score"])
plt.xticks(rotation=90)
plt.title("Feature Influence Scores (Gaussian Naive Bayes)")
plt.tight_layout()
plt.show()
