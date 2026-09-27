# Maryam Shamshad 
# 23-NTU-CS-1047
# Activity 1
# Pakistani Provinces Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Create Dataset
df = pd.DataFrame({
    "Province": ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan"],
    "Population (millions)": [127.7, 55.7, 40.9, np.nan],
    "Literacy Rate (%)": [66.0, np.nan, 55.0, np.nan],
    "Region": ["East", "South", "North", "West"],
})
print(df)
print()
print("Missing values per column:")
print(df.isna().sum())
df = df.dropna(subset=["Population (millions)"]).copy()
median_value = df["Literacy Rate (%)"].median()
df["Literacy Rate (%)"] = df["Literacy Rate (%)"].fillna(median_value)
df = df.reset_index(drop=True)
print("Median used to fill Sindh:", median_value)
print(df)
print("Missing values left:", df.isna().sum().sum())

# Encode the Region column
le = LabelEncoder()
df["Region_Label"] = le.fit_transform(df["Region"])
print("Label mapping:", {region: int(num) for num, region in enumerate(le.classes_)})
one_hot = pd.get_dummies(df["Region"], prefix="Region", dtype=int)
df = pd.concat([df, one_hot], axis=1)
df

# Scatter plot: population vs literacy rate
plt.figure(figsize=(7, 5))
plt.scatter(df["Population (millions)"], df["Literacy Rate (%)"], s=120, edgecolor="black")
for _, row in df.iterrows():      
    plt.annotate(row["Province"], (row["Population (millions)"], row["Literacy Rate (%)"]),
                 xytext=(6, 6), textcoords="offset points")
plt.margins(0.15)                                 
plt.xlabel("Population (millions)")
plt.ylabel("Literacy Rate (%)")
plt.title("Population vs Literacy Rate")
plt.grid(alpha=0.3)
plt.show()

# Outlier Tracing
lit = df["Literacy Rate (%)"]
q1 = lit.quantile(0.25)
q3 = lit.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr            
upper = q3 + 1.5 * iqr           
print(f"Q1={q1}  Q3={q3}  IQR={iqr}  ->  allowed range: {lower} to {upper}")
outliers = df[(lit < lower) | (lit > upper)]
print("Outlier provinces:", outliers["Province"].tolist() if not outliers.empty else "none")
plt.figure(figsize=(4, 4))
plt.boxplot(lit)
plt.xticks([1], ["Literacy Rate"])
plt.ylabel("Literacy Rate (%)")
plt.title("Boxplot of literacy rate")
plt.show()