import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

ins = pd.read_csv('insurance.csv')

print("Shape:", ins.shape)
print(ins.head(10))
print("Missing values:")
print(ins.isnull().sum())
print(ins['charges'].describe())

ins['sex'] = ins['sex'].map({'female': 0, 'male': 1})
ins['smoker'] = ins['smoker'].map({'no': 0, 'yes': 1})
ins = pd.get_dummies(ins, columns=['region'], dtype=int)

X = ins.drop('charges', axis=1)
y = ins['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_test.shape)

ridge = make_pipeline(StandardScaler(), Ridge())
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gbr = GradientBoostingRegressor(random_state=42)

ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)
gbr.fit(X_train, y_train)



base_models = [('ridge', ridge), ('rf', rf), ('gbr', gbr)]

stack = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression(),
    cv=5
)
stack.fit(X_train, y_train)

all_models = {'Ridge': ridge, 'Random Forest': rf,
              'Gradient Boosting': gbr, 'Stacking': stack}

rows = []
for name, m in all_models.items():
    pred = m.predict(X_test)
    rows.append({'Model': name,
                 'MAE': mean_absolute_error(y_test, pred),
                 'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
                 'R2': r2_score(y_test, pred)})

print(pd.DataFrame(rows).set_index('Model').round(3))
coefs = stack.final_estimator_.coef_
for name, c in zip(['ridge: ', 'rf: ', 'gbr: '], coefs):
    print(name, round(c, 3))
print("Intercept", round(stack.final_estimator_.intercept_, 3))
print("The stack trusts most:", ['ridge', 'rf', 'gbr'][np.argmax(coefs)])
stack_pred = stack.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, stack_pred, alpha=0.6)
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, 'r--')
plt.xlabel('Actual charges')
plt.ylabel('Predicted charges')
plt.title('Stacking: actual vs predicted')
plt.show()