# Maryam Shamshad 
# 23-NTU-CS-1047
# Data Preprocessing
# Handling Missing Values
# Creating a Sample Dataset with Missing Values and Displaying its Original DataFrames
# Handle all Missing Values using dropna and filna and Display Results
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
data = {
    'A': [1, 2, None, 4, 5],
    'B': [6, None, 8, 9, 10],
    'C': [11, 12, 13, None, 15]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
df_dropna_rows = df.dropna(axis=0)
df_dropna_columns = df.dropna(axis=1)
df_fill_median = df.fillna(df.median())
print("\nDataFrame after dropping rows with missing values:")
print(df_dropna_rows)
print("\nDataFrame after dropping columns with missing values:")
print(df_dropna_columns)
print("\nDataFrame after filling missing values with the median:")
print(df_fill_median)