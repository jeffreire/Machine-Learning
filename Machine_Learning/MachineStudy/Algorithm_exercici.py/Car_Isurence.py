import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

# age = 1
# jobStr = 2
# maritalStr = 3
# educationStr = 4
# default = 5
# balance = 6
# carloan = 7
# communicationStr = 8
# lastContactDay = 9
# lastContactMonthStr = 10
# noOfContacts = 11
# prevAttempts = 12
# OutoComeStr = 13
# callStart = 14
# callEnd = 15

# preparing data
dataset = pd.read_csv('Machine_Learning/MachineStudy/Data/Car_Insurance.csv')

# preenchendo linhas vazias de cada coluna
dataset['Job'].fillna('no Job', inplace= True)
dataset['Marital'].fillna('no marital', inplace= True)
dataset['Education'].fillna('no Education', inplace= True)
dataset['Communication'].fillna('no comunication', inplace= True)
dataset['LastContactMonth'].fillna('No LastContactMonth', inplace= True)
dataset['Outcome'].fillna('No OutCome', inplace= True)

'''print(dataset.corr())'''
# Separando por varias dependentes e indepedentes
X = dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]].values
y = dataset.iloc[:, -1].values

# preenchendo as linhas vazias com a media de cada coluna com dados escalar
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
X[:, [0,4,5,6,8,10,11]] = imputer.fit_transform(X[:, [0,4,5,6,8,10,11]])

# convertendo as colunas classificatorias para matrizes binarias
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
columnTransformer =  ColumnTransformer([('enconder', OneHotEncoder(), [1,2,3,7,9,12])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X), dtype= np.str)

# obtendo o tempo entre as duas colunas de horas 
new_column = []
for i in X[:, [-2, -1]]:
    data1 = i[0].split(':')
    data2 = i[1].split(':')
    data1 = dt.datetime(1, 1, 1, int(data1[0]), int(data1[1]), int(data1[2]))
    data2 = dt.datetime(1, 1, 1, int(data2[0]), int(data2[1]), int(data2[2]))
    seconds = (data2 - data1).total_seconds()
    new_column.append(seconds)

# deletando as duas ultimas colunas e add uma nova coluna
X = np.delete( X, np.s_[-1], axis=1)
X = np.delete( X, np.s_[-1], axis=1)
X = np.insert(X, 5, new_column, axis=1)

# separando os dados para teste e treino
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state= 0)

# padronizando os dados para serem mais narturais
from  sklearn.preprocessing import StandardScaler
sc_x = StandardScaler().fit(X_train)
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# algoritimo textra trees classifier
from sklearn.ensemble import ExtraTreesClassifier 
classificationExtraTree = ExtraTreesClassifier(n_estimators= 200, random_state= 0)
classificationExtraTree.fit(X_train, y_train)
# prevendo com base no teste
y_pred = classificationExtraTree.predict(X_test)

# matriz dos erros e acertos
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('ExtraTree: {}%'.format(classificationExtraTree.score(X_test, y_test)))

# algoritimo svc
from sklearn.svm import SVC 
classificationSVC = SVC(random_state= 0)
classificationSVC.fit(X_train, y_train)
# prevendo a base no teste
y_pred = classificationSVC.predict(X_test)

# matriz dos erros e acertos
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('-------------------------')
print(cm)
print('SVC: {}%'.format(classificationSVC.score(X_test, y_test)))

# algoritimo logistica
from sklearn.linear_model import LogisticRegression 
classificationLogistic = LogisticRegression(random_state= 0)
classificationLogistic.fit(X_train, y_train)
# prevemdo a base de teste
y_pred = classificationLogistic.predict(X_test)

# matriz dos erros e acertos
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('-------------------------')
print(cm)
print('Logistic: {}%'.format(classificationLogistic.score(X_test, y_test)))