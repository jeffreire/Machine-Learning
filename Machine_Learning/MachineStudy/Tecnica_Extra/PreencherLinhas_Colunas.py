# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

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
# autoCome = 13
# callStart = 14
# callEnd = 15

# dataset = pd.read_csv('Machine_Learning/MachineStudy/Data/Car_Insurance.csv')

# dataset['Job'].fillna('no Job', inplace= True)
# dataset['Marital'].fillna('no marital', inplace= True)
# dataset['Education'].fillna('no Education', inplace= True)
# dataset['Communication'].fillna('no comunication', inplace= True)
# dataset['LastContactMonth'].fillna('No LastContactMonth', inplace= True)
# dataset['Outcome'].fillna('No OutCome', inplace= True)

# print(dataset.corr())
# X = dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]].values
# y = dataset.iloc[:, -1].values

# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer()
# X[:, [0,4,5,6,8,10,11]] = imputer.fit_transform(X[:, [0,4,5,6,8,10,11]])
# # age = 0
# # job = 1 #classificatoria
# # marital = 2 #classi
# # educationStr = 3
# # default = 4
# # balance = 5
# # carloan = 6
# # communicationStr = 7
# # lastContactDay = 8
# # lastContactMonthStr = 9
# # noOfContacts = 10
# # prevAttempts = 11
# # autoCome = 12
# # callStart = 13
# # callEnd = 14

# # from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# # labelEnconder = LabelEncoder()

# # X[:, 1]= labelEnconder.fit_transform(X[:, 1])

# # oneHotEncoder = OneHotEncoder()
# # X = oneHotEncoder.fit_transform(X.reshape(-1,1) ).toarray()

# # X_categorical = np.delete(X, np.s_[0], axis= 1)
# # X = np.concatenate((X), axis=1)

# # states_1 = oneHotEncoder.fit_transform( X[:, 1].reshape(-1,1) ).toarray()
# # states_1 = np.delete( states_1, np.s_[0], axis=1)

# # states_2 = oneHotEncoder.fit_transform( X[:, 2].reshape(-1,1) ).toarray()
# # states_2 = np.delete( states_2, np.s_[0], axis=1)

# # states_3 = oneHotEncoder.fit_transform( X[:, 3].reshape(-1,1) ).toarray()
# # states_3 = np.delete( states_3, np.s_[0], axis=1)

# # states_7 = oneHotEncoder.fit_transform( X[:, 7].reshape(-1,1) ).toarray()
# # states_7 = np.delete( states_7, np.s_[0], axis=1)

# # states_9 = oneHotEncoder.fit_transform( X[:, 9].reshape(-1,1) ).toarray()
# # states_9 = np.delete( states_9, np.s_[0], axis=1)
# # X = np.concatenate(([states_1]), axis= 1)

# # X = np.concatenate( (states_1, states_2, states_3, states_7, states_9), axis=1)

# # from sklearn.preprocessing import OneHotEncoder
# # oneHotEncoder = OneHotEncoder(categories='auto')
# # states_1 = oneHotEncoder.fit_transform( dataset[:, 1].reshape(-1,1) ).toarray()
# # states_1 = np.delete( states_1, np.s_[0], axis=1)
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder

# columnTransformer =  ColumnTransformer([('enconder', OneHotEncoder(), [1,2,3,7,9,12])], remainder='passthrough')
# X = np.array(columnTransformer.fit_transform(X), dtype= np.str)

# new_column = []
# import datetime
# for i in X[:, [-2, -1]]:
#     data1 = i[0].split(':')
#     data2 = i[1].split(':')
#     data1 = datetime.datetime(1, 1, 1, int(data1[0]), int(data1[1]), int(data1[2]))
#     data2 = datetime.datetime(1, 1, 1, int(data2[0]), int(data2[1]), int(data2[2]))
#     seconds = (data2 - data1).total_seconds()
#     new_column.append(seconds)

# X = np.delete( X, np.s_[-1], axis=1)
# X = np.delete( X, np.s_[-1], axis=1)
# X = np.insert(X, 5, new_column, axis=1)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state= 0)

# from  sklearn.preprocessing import StandardScaler
# sc_x = StandardScaler().fit(X_train)
# X_train = sc_x.fit_transform(X_train)
# X_test = sc_x.transform(X_test)

# from sklearn.ensemble import ExtraTreesClassifier 
# regression = ExtraTreesClassifier(n_estimators= 1000, random_state= 0)
# regression.fit(X_train, y_train)

# y_pred = regression.predict(X_test)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# print(regression.score(X_test, y_test))

# # dataset['Job'].fillna('no Job', inplace= True)
# # dataset['Marital'].fillna('no marital', inplace= True)
# # dataset['Education'].fillna('no EDucation', inplace= True)
# # dataset['Communication'].fillna('no comunication', inplace= True)
# # dataset['LastContactMonth'].fillna('No LastContactMonth', inplace= True)

# # from sklearn.preprocessing import OneHotEncoder
# # oneHotEncoder = OneHotEncoder(categories='auto')
# # states_1 = oneHotEncoder.fit_transform( dataset[:, 1].reshape(-1,1) ).toarray()
# # states_1 = np.delete( states_1, np.s_[0], axis=1)

# # states_2 = oneHotEncoder.fit_transform( X[:, 2].reshape(-1,1) ).toarray()
# # states_2 = np.delete( states_2, np.s_[0], axis=1)

# # states_3 = oneHotEncoder.fit_transform( X[:, 3].reshape(-1,1) ).toarray()
# # states_3 = np.delete( states_3, np.s_[0], axis=1)

# # states_5 = oneHotEncoder.fit_transform( X[:, 5].reshape(-1,1) ).toarray()
# # states_5 = np.delete( states_5, np.s_[0], axis=1)

# # states_7 = oneHotEncoder.fit_transform( X[:, 7].reshape(-1,1) ).toarray()
# # states_7 = np.delete( states_7, np.s_[0], axis=1)
# # X = np.concatenate(([states_1]), axis= 1)

# # X = np.concatenate( (states_1, states_2, states_3, states_5, states_7), axis=1)


# Para consultar a quantidade de dados em uma coluna pelo name
'''fraud= dataset[dataset['default']== False]
credit= dataset[dataset['default']== True]'''