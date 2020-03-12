# ------------------------------RANDOM FOREST REGRESSION---------------------------------------
# Importando as librarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# preparando os datasets
dataset = pd.read_csv('Machine_Learning/MachineStudy/Data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
print(dataset)
# Preparando a regrssao da arvore de decisao dos dados
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 368, random_state= 0)
regressor.fit(X, y)

# Prevendo um novo resultado com mais precisao
y_pred = regressor.predict([[6.5]])
print(y_pred)

# Visualizando o resultado da regressao da arvore de decisao
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color= 'red')
plt.plot(X_grid, regressor.predict(X_grid), color= 'blue')
plt.title('Verdade ou Mentira (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
