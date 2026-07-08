import numpy as np
import pandas as pd

print("Conda Environment Successfully Configured!")

arr = np.array([10, 20, 30, 40, 50])
print("NumPy Array:", arr)

df = pd.DataFrame({
    "Student": ["Alice", "Bob", "Charlie"],
    "Marks": [85, 90, 88]
})

print("\nDataFrame:")
print(df)
