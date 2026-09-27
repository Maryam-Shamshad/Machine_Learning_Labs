# Maryam Shamshad 
# 23-NTU-CS-1047
# Loading Dataset
# Loading from CSV File____From Local File & From Online source
import pandas as pd
df_csv = pd.read_csv('cars.csv')
print("CSV Data:")
print(df_csv.head())
online_csv_url = 'https://people.sc.fsu.edu/~jburkardt/data/csv/cities.csv'
df_online_csv = pd.read_csv(online_csv_url)
print("\nOnline CSV Data:")
print(df_online_csv.head())