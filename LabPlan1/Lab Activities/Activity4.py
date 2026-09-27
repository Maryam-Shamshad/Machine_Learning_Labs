# Maryam Shamshad 
# 23-NTU-CS-1047
# Activity 4
# Multi-format Data Loading & Visualization Challenge
import json
import os
import matplotlib.pyplot as plt
import pandas as pd

# Create Datasets
os.makedirs("lab_files", exist_ok=True)
names = ["Ali", "Sara", "Ahmed", "Fatima", "Usman", "Ayesha", "Hamza", "Zainab", "Bilal", "Hira"]
marks_l = [78, 85, 62, 91, 55, 88, 47, 73, 66, 95]
attend_l = [92, 88, 65, 96, 58, 90, 60, 81, 68, 98]
bonus_l = [5, 8, 2, 10, 0, 9, 1, 4, 3, 10]
pd.DataFrame({"Name": names, "Marks": marks_l}).to_csv("lab_files/students.csv", index=False)
with open("lab_files/attendance.json", "w") as f:
    json.dump([{"Name": n, "Attendance": a} for n, a in zip(names, attend_l)][::-1], f)
bonus_df = pd.DataFrame({"Name": names, "Bonus": bonus_l})
bonus_df[bonus_df["Name"] != "Usman"].sample(frac=1, random_state=1).to_excel("lab_files/extra.xlsx", index=False)

# Load & Merge Data
df_students = pd.read_csv("lab_files/students.csv")
df_attend = pd.read_json("lab_files/attendance.json")
df_bonus = pd.read_excel("lab_files/extra.xlsx")
merged = df_students.merge(df_attend, on="Name", how="left").merge(df_bonus, on="Name", how="left")
merged["Bonus"] = merged["Bonus"].fillna(0).astype(int)

# Create Scatter Plot
low_mask = merged["Attendance"] < 70
plt.figure(figsize=(7, 4.5))
plt.scatter(merged.loc[~low_mask, "Attendance"], merged.loc[~low_mask, "Marks"], s=80, color="tab:blue", label="Attendance ≥ 70%")
plt.scatter(merged.loc[low_mask, "Attendance"], merged.loc[low_mask, "Marks"], s=100, color="red", label="Attendance < 70%")
plt.axvline(70, color="gray", linestyle="--")
for _, r in merged.iterrows():
    plt.annotate(r["Name"], (r["Attendance"], r["Marks"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
plt.xlabel("Attendance (%)")
plt.ylabel("Marks")
plt.title("Scatter Plot: Marks vs Attendance")
plt.legend()
plt.tight_layout()
plt.show()

# One-Hot Encoding
encoded = pd.get_dummies(merged, columns=["Name"], dtype=int)
print("Final One-Hot Encoded Output Preview")
print(encoded.head())