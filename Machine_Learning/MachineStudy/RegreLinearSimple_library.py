#Preparar o pre processamento de dados
#bibliotecas padroes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# importando a biblioteca que ira separar os dados
from sklearn.model_selection import train_test_split

arquivo = pd.read_csv('C:/Users/jefferson.maria/Desktop/MachineLearning/Machine_Learning/MachineStudy/Salary_Data.csv')
# Primeira coluna e todas as linhas
x = arquivo.iloc[:, :-1].values
# Utima coluna e todas as linhas
y = arquivo.iloc[:, 1].values

from sklearn.linear_model import LinearRegression
#Codigo que divide em duas arrays os dados do arquivo
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/3, random_state=0)
regressor = LinearRegression()
# X_train é o conjunto de dados que sao independentes
# y_train sao os dados dependentes de x_train
regressor.fit(x_train, y_train)
# usa a linha lineaar e preve com base nos dados do conjunto teste, calculando e prevendo os possiveis dados
y_pred = regressor.predict(x_test)

# Plotando os dados no grafico de dispersao, todos os dados de x_train e  y_train sao as coordenadas
plt.scatter(x_train,y_train, color='red')
# plotando os de testes
plt.scatter(x_test,y_test, color='black')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs experince')
plt.ylabel('Salary')
plt.xlabel('Experience')
plt.show()
