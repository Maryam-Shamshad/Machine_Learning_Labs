# Maryam Shamshad 
# 23-NTU-CS-1047
# Data Preprocessing
# Feature Scaling using Standardization & Normalization
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
data = pd.DataFrame({'Age':[22, 25, 47, 35, 60], 'Income': [25000, 32000, 95000, 60000, 150000]})
standard_scaler = StandardScaler()
data_standard = standard_scaler.fit_transform(data)
minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data)
print("Original Data")
print(data)
print("Standardized Data (Mean=0, SD=1)")
print(data_standard)
print("Min-Max Scaled Data (0 to 1 Range)")
print(data_minmax)
