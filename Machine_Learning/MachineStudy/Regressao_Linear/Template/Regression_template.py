import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#preparando os arquivo com os dados para serem analisados/Manipulados
dataset = pd.read_csv('../Data.csv')
X = dataset.iloc[:, :-1].value
y = dataset.iloc[:, 3].values

# Preparanndo os dados e semparando para treino e testes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)

# Preparando a escala dos dados
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Treinando o algoritimo de regressao
# Create your regressor here

'''y_pred = regressor.predit(6.5)'''

# Visualizando os resultados da regressao
'''plt.scatter(X, y, color= 'red')
plt.plot(X, regressor.predict(X), colo= 'blue')
plt.title('titulo')
plt.xlabel('titulo barra x')
plt.ylabel('titulo barra x')
plt.show()'''

# Visualizando os resultados da regressao
'''x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.rechape((len(x_grid), 1))
'''
'''plt.scatter(X, y, color= 'red')
plt.plot(X, regressor.predict(X), colo= 'blue')
plt.title('titulo')
plt.xlabel('titulo barra x')
plt.ylabel('titulo barra x')
plt.show()'''