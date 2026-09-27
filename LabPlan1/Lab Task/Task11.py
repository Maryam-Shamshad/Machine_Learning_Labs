# Maryam Shamshad 
# 23-NTU-CS-1047
# Data Preprocessing
# Handling Categorical Attributes
# Manualy Defining the Order
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
data = pd.DataFrame({'Education': ['High School', 'Bachelors', 'Masters', 'PhD']})
order = [['High School', 'Bachelors', 'Masters', 'PhD']]
oe = OrdinalEncoder(categories=order)
data['Education_Encoded'] = oe.fit_transform(data[['Education']])
print(data)