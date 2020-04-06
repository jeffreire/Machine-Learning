# #tupla = (1, 2, 3)
# # tupla = list(tupla)
# # tupla[0] = 7


# # lista = [1, 2, 3]
# # lista[0] = 7


# #lis = [{'Nome':'Gustavo', 'Idade':16}, {'Nome':'Gabriel'}]

# # dicionario = {'p1': 1, 'p2':2, 'p3':3}
# # print(dicionario)
# #print(tupla)

# # class Bandit:
    
# #     def __init__(self, chance):
# #         self.chance = chance
# #         print(chance)

# #     def pull(self):
# #         prob = random.random() # 0..1
# #         if prob < self.chance:
# #             return True
# #         return False

# #     def calcular(self, a, b):
# #         pass

# # bandit1 = Bandit(0.30)
# # bandit2 = Bandit(0.20)
# # bandit2 = Bandit(0.40)
# # bandit1.calcular(1, 2)

'''nameDoArquivo['nome da colula'] = nameDoArquivo['nome da coluna'].replace('nome da linha, valor que irei atribuir')'''
# Variavel preditoras
'''y = nameDoArquivo['nome da coluna']'''
'''x = nameDoArquivo.drop('nome da coluna', axis = 1)'''

# para ver a contidade no arquivo
'''name do aquivo.shape'''
# criando os modelos
''' modelo = ExtraTreesClassicafter()
    modelo.fit(x_traino, y_treino)'''
# Imprimindo os resultados:
'''Resultado = modelo.score(x_teste, y_teste)
print(resultado)'''


import pandas as pd
import numpy as np
import matplotlib.pylab as plt

arquivo = pd.read_csv('C:/Users/jefferson.maria/Desktop/MachineLearning/Machine_Learning/MachineStudy/Data/wine_dataset.csv')

arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)

y = arquivo['style']
X = arquivo.drop('style', axis= 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_traine, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)

from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier(n_estimators=200, max_features= 1)
modelo.fit(X_train, y_traine)

# prevendo os dados é bom para plotar
previsao = modelo.predict(X_test)

resultado = modelo.score(X_test, y_test)
print('Acuracia: {:.3}%, previsao: {}'.format(resultado, previsao))
