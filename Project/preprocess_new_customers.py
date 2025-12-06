import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
 
df = pd.read_csv("Project/New Customer.csv")

# Remove Loan_ID
if "Loan_ID" in df.columns:
    df = df.drop(columns=["Loan_ID"])

print("\nRAW NEW CUSTOMER DATA:")
print(df.head())

 
if "Credit_History" in df.columns:
    df["Credit_History"] = df["Credit_History"].astype("object")
 
 
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

 
num_imp = SimpleImputer(strategy="mean")
df[numeric_cols] = num_imp.fit_transform(df[numeric_cols])

 
cat_imp = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

 
if "Dependents" in df.columns:
    df["Dependents"] = df["Dependents"].replace("3+", 3)
    df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce")

 
label_map = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Education": {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Credit_History": {1.0: 1, 0.0: 0}
}

for col, mapping in label_map.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

 
if "Property_Area" in df.columns:
    df = pd.get_dummies(df, columns=["Property_Area"], drop_first=True)

if "Property_Area_Semiurban" not in df.columns:
    df["Property_Area_Semiurban"] = 0

if "Property_Area_Urban" not in df.columns:
    df["Property_Area_Urban"] = 0

 
df["Property_Area_Semiurban"] = df["Property_Area_Semiurban"].astype(int)
df["Property_Area_Urban"] = df["Property_Area_Urban"].astype(int)

 
normalizer = MinMaxScaler()
df[numeric_cols] = normalizer.fit_transform(df[numeric_cols])

 
df.to_csv("Project/Cleaned_New_CustomerNONORMALIZATIOM.csv", index=False)

print("\n====================================================")
print("CLEANED NEW CUSTOMER FILE SAVED!")
print("Location: Project/Cleaned_New_CustomerNONORMALIZATION.csv")
print("Shape:", df.shape)
print("Preview:")
print(df.head())
print("====================================================")
