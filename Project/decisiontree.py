import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("D:\DataMining\projectphase2\Loan-prediction\Project\Cleaned_Loan_Train_Data.csv")
print("Dataset shape:", df.shape)
df.head()


X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(X_train.shape, X_test.shape)

'''
split data to x  input and y target (loan status) 
train 80% test 20%
'''


dt_model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

dt_model.fit(X_train, y_train)

'''
train tree based on the cleaned dataset 
'''

plt.figure(figsize=(22,12))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["No Loan","Loan"],
    filled=True,
    rounded=True
)
plt.show()
'''
visualization of how model makes decision wich feature is the root 
'''


y_pred = dt_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
"""
test the model using the 20% test data 
"""
TN, FP, FN, TP = cm.ravel()

accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = (FP + FN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN)      # Recall
specificity = TN / (TN + FP)
precision = TP / (TP + FP)

print("Accuracy:", accuracy)
print("Error rate:", error_rate)
print("Sensitivity (Recall):", sensitivity)
print("Specificity:", specificity)
print("Precision:", precision)

"""
extract tn,tp,fp,fn then compute error rate......
"""

importances = dt_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,5))
plt.title("Feature Importance (Decision Tree)")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), X.columns[indices], rotation=90)
plt.show()

for i in indices:
    print(f"{X.columns[i]}: {importances[i]:.4f}")
"""
wich paramter plays vital role in loan approval 
"""
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

prediction = dt_model.predict(new_customer)[0]

print("Loan Prediction:", "Approved" if prediction == 1 else "Rejected")


prediction = dt_model.predict(new_customer)[0]
print("Loan Prediction:", "Approved" if prediction==1 else "Rejected")
semiurban = df[df["Property_Area_Semiurban"] == 1]
married = semiurban[semiurban["Married"] == 1]

percentage = (married["Loan_Status"].sum() / married.shape[0]) * 100
print("Percentage of married semiurban customers who obtained loan: %.2f%%" % percentage)
