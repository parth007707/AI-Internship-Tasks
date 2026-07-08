import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Sample dataset
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", np.nan],
    "Age": [25, np.nan, 30, 22, 28],
    "Gender": ["F", "M", "M", "M", "F"],
    "Salary": [50000, 60000, 70000, 65000, 72000]
})

print("Original Dataset:\n")
print(df)

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Name"] = df["Name"].fillna("Unknown")

# Encode categorical data
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

# Features and target
X = df[["Age", "Gender"]]
y = df["Salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nProcessed Dataset:\n")
print(df)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))
