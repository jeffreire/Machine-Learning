import pandas as pd
import numpy as np
import math
import base64
import struct
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
# score_1 = 2
# score_2 = 3
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

# metodo que preenche as linhas das colunas faltantes com a media, 
# passando por parametro dataset e a coluna
def _filling_numeric_columns(dt, colum):
    mean = dt[colum].mean()
    mean = math.floor(mean)
    dt.update(dt[colum].fillna(mean))

# Metodo que descodifica as colunas codificadas passando por parametro o index da coluna
def _decoder_columns_base64(colum):
    r = []
    binario = colum
    for i in binario:
      data = base64.standard_b64decode(i)
      assert len(data) % 4 == 0
      count = len(data) // 4
      result = struct.unpack('<{0}f'.format(count), data)
      r.append(result)
    return r

# Metodo que separa os dados codificados em quatro novas listas, 
# passando por parametro o array como sendo a coluna descodificada, X sendo a matrix para add as novas colunas
# e numero sendo o index da nova coluna
def _add_news_coluns_in_X(arr, X, numero):
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    for i in arr:
      for a in i:
        if a == i[0]:
          a1.append(a)
        elif a == i[1]:
          a2.append(a)
        elif a == i[2]:
         a3.append(a)
        else:
          a4.append(a)
    return _add_colunas(X, numero, a1, a2, a3, a4)

# Metodo que add de fato as novas colunas
# passando como paramentro, X sendo a matriz que sera add
# numero o numero do index da nova coluna
# colum0, ..., colum3 sendo as colunas  que serao add
def _add_colunas(X, numero, colum0, colum1, colum2, colum3):
      X = np.insert(X, numero, colum0, axis=1)
      c2 = numero + 1
      X = np.insert(X, (c2), colum1, axis=1)
      c3 = c2 + 1
      X = np.insert(X, (c3), colum2, axis=1)
      c4 = c3 + 1
      X = np.insert(X, (c4), colum3, axis=1)
      return X

# metodo que executa os modelos sugeridos de classificacao, passando por parametro
def _model_classification(X_train, X_test, y_train, y_test, modelo, name):
    model = modelo
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(_failure_and_hit_matrix(y_test, y_pred))
    print('{}: {:.2f}'.format(name, model.score(X_test, y_test)))

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
X = dataset.iloc[:, [2,3,4,5,6,7,8,9,10,11,13,22,23,24,25,26] ].values
y = dataset.iloc[:,1].values

# Descodificando as informacoes da coluna
X[:, 0] = _decoder_columns_base64(X[:, 0])
X[:, 1] = _decoder_columns_base64(X[:, 1])


# Incluindo colunas novas
X = _add_news_coluns_in_X(X[:, 0], X, 14)
X = _add_news_coluns_in_X(X[:, 1], X, 18)

# Deletando as duas colunas descodificadas
X = np.delete( X, np.s_[0], axis= 1)
X = np.delete( X, np.s_[0], axis= 1)

# convertendo os dados da coluna default para numeros
y = y.astype('int')

# Separando dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state= 0)

# Resolvendo o problema de erro de distribuicao 
X_train, y_train = RandomUnderSampler().fit_resample(X_train, y_train)

'''from  sklearn.preprocessing import StandardScaler
sc_x = StandardScaler().fit(X_train)
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)'''

_model_classification(X_train, X_test, y_train, y_test, ExtraTreesClassifier(random_state=0, n_estimators= 100), 'ExtraTree' )
_model_classification(X_train, X_test, y_train, y_test, RandomForestClassifier(random_state=0, n_estimators= 100), 'RandomForest' )
_model_classification(X_train, X_test, y_train, y_test, LogisticRegression(random_state=0), 'Logistica' )
_model_classification(X_train, X_test, y_train, y_test, AdaBoostClassifier(random_state=0, n_estimators= 1), 'AdaBoost' )