# Maryam Shamshad 
# 23-NTU-CS-1047
# Data Preprocessing
# Feature Selection, Feature Engineering, Dimentionailty Reduction
import pandas as pd
from sklearn.decomposition import PCA
data = pd.DataFrame({
    'Height_cm':[150, 160, 170, 180, 190],
    'Height_Inch':[59, 63, 67, 71, 75],  
    'Weight':[50, 60, 70, 80, 90],
    'Date': ['2026-01-05', '2026-02-10', '2026-03-15', '2026-04-01', '2026-05-20']
})

# Feature Selection
correlation = data[['Height_cm', 'Height_Inch', 'Weight']].corr()
print("Correlation Matrix:\n", correlation)
data = data.drop('Height_Inch', axis=1)

# Feature Engineering
data['BMI'] = data['Weight'] / ((data['Height_cm'] / 100) ** 2)
data['Date'] = pd.to_datetime(data['Date'])
data['Day_of_Week'] = data['Date'].dt.day_name()
print("\nData after Feature Engineering:\n", data)

# Dimensionality Reduction (PCA)
numeric_features = data[['Height_cm', 'Weight', 'BMI']]
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(numeric_features)
print("\nData after PCA (reduced to 2 components):\n", reduced_data)