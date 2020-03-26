import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def _filling_numeric_columns(dt, colum):
    mean = dt[colum].mean()
    mean = math.floor(mean)
    dt.update(dt[colum].fillna(mean))

train = pd.read_csv('Competicao_kaggle/Data/train.csv')
test = pd.read_csv('Competicao_kaggle/Data/test.csv')

train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace= True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace= True)

_filling_numeric_columns(train, ['Age'])
_filling_numeric_columns(test, ['Age'])
_filling_numeric_columns(test, ['Fare'])

train['Embarked'].fillna('no', inplace= True)

X = train.iloc[:, [0,2,3,4,5,6,7,8] ].values
y = train.iloc[:, 1].values

oneHotEncoder = OneHotEncoder(categories='auto')
scalar_features = X[:, :2]
scalar_features1 = X[:, 3:6]
states = oneHotEncoder.fit_transform( X[:, 2].reshape(-1,1) ).toarray()
states1 = oneHotEncoder.fit_transform( X[:, 6].reshape(-1,1) ).toarray()

states = np.delete( states, np.s_[0], axis=1)
X = np.concatenate( (scalar_features, states, scalar_features1, states1), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state= 0)

from sklearn.ensemble import RandomForestClassifier 
classificationExtraTree = RandomForestClassifier(random_state=0, n_estimators= 300)
classificationExtraTree.fit(X_train, y_train)
y_pred = classificationExtraTree.predict(X_test)

print(round(accuracy_score(y_pred, y_test) * 100, 2))