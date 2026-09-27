# Maryam Shamshad 
# 23-NTU-CS-1047
# Loading Dataset
# Loading from Json File____From Local File & From Online source
import pandas as pd
df_json = pd.read_json('attendance.json')
print("JSON Data:")
print(df_json.head())
json_url = 'https://jsonplaceholder.typicode.com/users'
df_online_json = pd.read_json(json_url)
print("\nOnline JSON Data:")
print(df_online_json.head())