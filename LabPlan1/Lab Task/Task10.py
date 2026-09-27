# Maryam Shamshad 
# 23-NTU-CS-1047
# Data Preprocessing
# Handling Categorical Attributes
# Apply Label Encoding and One Hot Encoding Example
# Assume 'Method' is a categorical column
from sklearn.preprocessing import LabelEncoder
import pandas as pd
features_df = pd.DataFrame({'Regionname': ['North', 'South', 'East', 'West', 'North']})
le = LabelEncoder()
features_df['Region'] = le.fit_transform(features_df['Regionname'])
print("\nLabel Encoded Regions:")
print(features_df.value_counts())
features_df['Method'] = ['Method1', 'Method2', 'Method1', 'Method3', 'Method2']
df_one_hot = pd.get_dummies(features_df['Method'])
print("\nOne Hot Encoded DataFrame:")
print(df_one_hot.value_counts())