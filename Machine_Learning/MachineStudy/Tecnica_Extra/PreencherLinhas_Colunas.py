# -------------------Preeecnher linas e colunas vazias------------------------
'''from sklearn.impute import SimpleImputer'''
#Importo valores ausentes nas linhas em branco NaN
'''imputer = SimpleImputer(missing_values=np.nan, strategy='mean')'''
#pego todas as colunas e todas as linhas do arquivo de dados e calculo a medias dos valores
'''imputer.fit(x[:, 1:3])'''
#pego a media calculada e importo nas linhas vazias
'''x[:, 1:3] = imputer.transform(x[:, 1:3])'''

#----------------Codificar colunas especificas---------------#
#Codificando a array x, coluna country
#importo a biblioteca com os recursos par a codificacao
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder'''
# fit_trasform é o metodo que codifica a coluna na posicao 0
'''labelencoder_x = LabelEncoder() 
x[:,0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categories="Auto")
x = onehotencoder.fit_transform(x).toarray() 
labelencoder_y = LabelEncoder() 
y = labelencoder_x.fit_transform(y)
print(x)
print(y)'''

#OBS = Machine learning é baseado em formulas matematicas entao é de suma importancia 
# #Converter as strings das arrays para numeros, para que assim podemos usar-las nas formulas dos algoritmos