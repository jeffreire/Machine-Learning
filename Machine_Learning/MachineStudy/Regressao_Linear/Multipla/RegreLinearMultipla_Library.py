#----------------------------Multiple Linear Multipla-------------------------------------------------------------#
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the DataSets
file = pd.read_csv('C:/Users/jefferson.maria/Desktop/MachineLearning/Machine_Learning/MachineStudy/Data/50_Startups.csv')
fi_x = file.iloc[:,:-1].values
fi_y = file.iloc[:, 4].values

# Imortando a biblioteca para 'Criptogtrafar as strings da tabela'
# Reshap transforma em  matrix dimensao -1,1
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
oneHotEncoder = OneHotEncoder(categories='auto')
scalar_features = fi_x[:, :3]
states = oneHotEncoder.fit_transform( fi_x[:, 3].reshape(-1,1) ).toarray()

# dummy variable trap avoiding
states = np.delete( states, np.s_[0], axis=1)
fi_x = np.concatenate( (scalar_features, states), axis=1)

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
fi_X_train, fi_x_test, fi_Y_train, fi_y_test = train_test_split(fi_x, fi_y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
# Ajustando a regressão linear múltipla ao conjunto de treinamento
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(fi_X_train, fi_Y_train)

#Predicting the test set Result
# Metodo PREDICT preve as observacoes do conjunto de tests
y_pred= regressor.predict(fi_x_test)
""" for i in range(len(y_pred)):
    print('Pred: ',(round(y_pred[i], 2)),' ', 'file_test: ',(round(fi_y_test[i], 2)))"""

# Codigo que gera uma tabela com dados da tecnica Backward Elimination
# A coluna que obeter um resultado de t > p-valor, oui seja, t > 0,05, teremos que recusar a hipotese. remover a tabela
import statsmodels.api as sm
fi_x  = np.append(arr = np.ones((50,1)).astype(int), values= fi_x, axis=1)
x_opt = fi_x[:, [0,1,2,3,4,5]]  
model = sm.OLS(endog=fi_y, exog=x_opt).fit()
model.sumary()

# Removendo a coluna 2, pois seu valor de hipostese é superor ao valor-p, >0,5
"""x_opt = fi_x[:, [0,1,3,4,5]]  
model = sm.OLS(endog=fi_y, exog=x_opt).fit()
model.sumary()
"""
# -----------------------------------eliminando para tras de forma automatica com valores-p
"""import statsmodels.api as sm
def backwardElimination(x, sl):
    numVars = len(fi_x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(fi_y, fi_x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = fi_x[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)"""

# --------------------Eliminacao para tras com valores-p e r ajustado ao quadrado
"""import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)"""
