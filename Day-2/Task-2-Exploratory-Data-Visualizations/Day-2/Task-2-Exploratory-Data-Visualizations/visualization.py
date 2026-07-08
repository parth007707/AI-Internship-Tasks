import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset
df = pd.DataFrame({
    "Age": [22, 25, 28, 30, 35, 40, 45],
    "Salary": [25000, 30000, 45000, 50000, 65000, 70000, 80000],
    "Experience": [1, 2, 3, 5, 7, 10, 12]
})

print(df)

# Histogram
plt.figure(figsize=(6,4))
plt.hist(df["Salary"], bins=5)
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot
plt.figure(figsize=(6,4))
plt.scatter(df["Experience"], df["Salary"])
plt.title("Experience vs Salary")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# Box Plot
plt.figure(figsize=(5,4))
sns.boxplot(y=df["Salary"])
plt.title("Salary Boxplot")
plt.show()

# Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

print("Visualization Completed Successfully!")
