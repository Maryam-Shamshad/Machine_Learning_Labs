# Maryam Shamshad 
# 23-NTU-CS-1047
# Data Preprocessing
# Handling Categorical Attributes
# Manual Encoding
import pandas as pd
mapping = {'h': 1, 'u': 2, 't': 3}
features = pd.DataFrame({'Type': ['h', 'u', 't', 'h', 'u']})
features['Type'] = features['Type'].map(mapping)
print("\nEncoded Types:")
print(features.groupby('Type').size())