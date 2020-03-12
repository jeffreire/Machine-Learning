# --------------Polynomial Regression-----------------------
# Importando Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importando os dados
dataset = pd.read_csv('C:/Users/jefferson.maria/Desktop/MachineLearning/Machine_Learning/MachineStudy/Data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X)"""

# Preparando regression linear to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Preparando regressao Polynimial to the dataset
# codigo que calcula o polinomio de X
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree= 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizacao do resultado regressao linear
"""plt.scatter(X, y, color= 'red')
plt.plot(X, lin_reg.predict(X), color= 'blue')
plt.title('Verdade ou Mentira (Regressao linear)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()"""

# Visualizacao do resultado da regressao polinomial
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color= 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color= 'blue')
plt.title('Verdade ou Mentira (Regressao Polinomial)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# prevendo um novo resultado com regressao linear
lin_reg.predict([[6.5]])

# prevendo um novo resultado com regressao polinomial
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))