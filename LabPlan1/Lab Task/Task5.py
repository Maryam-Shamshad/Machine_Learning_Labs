# Maryam Shamshad 
# 23-NTU-CS-1047
# Loading Dataset
# Loading from Text File
import pandas as pd
df_text = pd.read_csv('data.txt', delimiter='\t')
print("Text File Data:")
print(df_text.head())