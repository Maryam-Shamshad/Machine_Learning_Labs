# Maryam Shamshad 
# 23-NTU-CS-1047
# Loading Dataset
# Manual Data Creation___From Dictionary
import pandas as pd
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['Lahore', 'Faisalabad', 'Karachi', 'Multan'],
    'Salary': [50000, 60000, 70000, 55000]
}
df_manual = pd.DataFrame(data)
print("Manual DataFrame:")
print(df_manual)