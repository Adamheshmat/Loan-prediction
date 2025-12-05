import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


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

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))