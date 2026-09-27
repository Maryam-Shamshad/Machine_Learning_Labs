# Maryam Shamshad 
# 23-NTU-CS-1047
# Activity 5
# Data Cleaning & Outlier Detection with Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create Dataset
data = {
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor"],
    "Age": [25, 27, 35, 29, None, 40],
    "Salary": [50000, None, 75000, 2000000, 60000, 90000],
    "Department": ["IT", "Finance", "IT", "HR", "Finance", "IT"],
}
df = pd.DataFrame(data)

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())

# Encode the categorical Features
df5_encoded = pd.get_dummies(df, columns=["Department"], dtype=int)
print("--- Processed Data Frame ---")
print(df5_encoded)

# Outlier Detection
q1, q3 = df["Salary"].quantile([0.25, 0.75])
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
plt.figure(figsize=(7, 3))
plt.boxplot(df["Salary"], vert=False, patch_artist=True)
plt.axvline(upper_fence, color="red", linestyle="--", label="Upper Fence")
plt.text(
    df.loc[df["Name"] == "Zara", "Salary"].iloc[0],
    1.05,
    "Zara",
    color="red",
    fontweight="bold",
)
plt.xlabel("Salary")
plt.title("Salary Boxplot Outlier Detection")
plt.legend()
plt.tight_layout()
plt.show()

# Comment on Zara's Salary 
print("\nOutlier Assessment")
print(f"IQR Upper Fence Limit: {upper_fence:,.2f}")
print(f"Zara's Recorded Salary: {df.loc[df['Name']=='Zara', 'Salary'].iloc[0]:,.2f}")
print(
    "Comment: Yes, Zara's salary is treated as a definitive outlier. "
    "It lies significantly above the upper fence statistical boundary"
    "and drastically distorts baseline population metrics."
)
