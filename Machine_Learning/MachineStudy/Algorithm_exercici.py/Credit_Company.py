import pandas as pd
import numpy as np
import math
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# score_3 = 4
# score_4 = 5
# score_5 = 6
# score_6 = 7
# risk_rate = 8
# amount_borrowed = 9
# borrowed_in_months = 10
# credit_limit = 11
# income = 13
# ok_since = 22
# n_bankruptcies = 23
# n_defaulted_loans = 24
# n_accounts = 25
# n_issues = 26

# metodo que preenche as linhas das colunas faltantes com a media, passando
# por parametro dataset e a coluna
def _filling_numeric_columns(dt, colum):
  mean = dt[colum].mean()
  mean = math.floor(mean)
  dt.update(dt[colum].fillna(mean))

# metodo que executa os modelos sugeridos de classificacao, passando por parametro
def _model_classification(X_train, X_test, y_train, modelo, name):
    model = modelo
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(_failure_and_hit_matrix(y_test, y_pred))
    print('{}: {}'.format(name, model.score(X_test, y_test)))

# metodo que retorna a matriz de erros e acertos
def _failure_and_hit_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return cm

# Preparando os dados
dataset = pd.read_csv('Machine_Learning/MachineStudy/Data/new_base.csv')

# Preenchendo as linhas numericas vazias
_filling_numeric_columns(dataset, ['score_3'])
_filling_numeric_columns(dataset, ['risk_rate'])
_filling_numeric_columns(dataset, ['amount_borrowed'])
_filling_numeric_columns(dataset, ['borrowed_in_months'])
_filling_numeric_columns(dataset, ['credit_limit']) 
_filling_numeric_columns(dataset, ['income'])
_filling_numeric_columns(dataset, ['ok_since'])
_filling_numeric_columns(dataset, ['n_bankruptcies'])
_filling_numeric_columns(dataset, ['n_accounts'])
_filling_numeric_columns(dataset, ['n_issues'])
_filling_numeric_columns(dataset, ['n_defaulted_loans'])

# Descartando as linhas vazias coluna default
dataset = dataset.dropna()
# separando as colunas essenciais
X = dataset.iloc[:, [4,5,6,7,8,9,10,11,13,22,23,24,25,26] ].values
y = dataset.iloc[:,1].values

# convertendo os dados da couna default para numeros
y = y.astype('int')

# Separando dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state= 0)

_model_classification(X_train, X_test, y_train, ExtraTreesClassifier(random_state=0, n_estimators= 500), 'ExtraTree' )
_model_classification(X_train, X_test, y_train, RandomForestClassifier(random_state=0, n_estimators= 500), 'RandomForest' )
_model_classification(X_train, X_test, y_train, LogisticRegression(random_state=0), 'Logistica' )
_model_classification(X_train, X_test, y_train, AdaBoostClassifier(random_state=0, n_estimators= 500), 'AdaBoost' )