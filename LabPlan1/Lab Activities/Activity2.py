# Maryam Shamshad 
# 23-NTU-CS-1047
# Activity 2
# Student Performance Tracker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate Student Data
rng = np.random.default_rng(42)
n = 100
math = rng.normal(65, 15, n).clip(0, 100).round(0)
science = rng.normal(68, 12, n).clip(0, 100).round(0)
english = rng.normal(70, 10, n).clip(0, 100).round(0)

# Assigning Grade on base of Average
avg = (math + science + english) / 3
grade = np.where(avg >= 75, "A", np.where(avg >= 62, "B", "C"))
df = pd.DataFrame(
    {"ID": np.arange(1, n + 1), "Math": math, "Science": science, "English": english, "Grade": grade}
)

# Handle Missing Marks
marks = ["Math", "Science", "English"]
for col in marks:
    df.loc[rng.choice(n, 5, replace=False), col] = np.nan
df[marks] = df[marks].fillna(df[marks].mean())

# Categorical Encoding
df["Total"] = df[marks].sum(axis=1)
df["Grade_Encoded"] = df["Grade"].map({"C": 0, "B": 1, "A": 2})
print("Processed Data Head")
print(df.head())

# Histogram of Math Scores
plt.figure(figsize=(6, 3.5))
plt.hist(df["Math"], bins=10, color="steelblue", edgecolor="white")
plt.xlabel("Math score")
plt.ylabel("Number of students")
plt.title("Distribution of Math Scores")
plt.show()

# Outlier Detection
q1, q3 = df["Total"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df[(df["Total"] < low) | (df["Total"] > high)]
print(f"\nOutliers found outside fences [{low:.1f}, {high:.1f}]: {len(outliers)}")
if not outliers.empty:
    print(outliers[["ID", "Total", "Grade"]])

# Boxplot of Total Scores
plt.figure(figsize=(6, 2.5))
plt.boxplot(df["Total"], vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.xlabel("Total score")
plt.title("Boxplot of Total Scores")
plt.show()

# Overall Summary
summary = df.groupby("Grade").agg(
    students=("ID", "count"),
    Math=("Math", "mean"),
    Science=("Science", "mean"),
    English=("English", "mean"),
    Total=("Total", "mean")
).round(1)
print("\nSummary Performance by Grade")
print(summary)
summary[["Math", "Science", "English"]].plot(kind="bar", figsize=(6, 3.5), rot=0)
plt.ylabel("Mean mark")
plt.title("Average Marks per Subject by Grade")
plt.tight_layout()
plt.show()
