# ------------------------Decision tree Classification -------------------------
# Importando as bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# preparando os dados
dataset = pd.read_csv('Machine_Learning/MachineStudy/Data/Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Preparando os dados para os testes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 0)

# Preparando a escala dos dados
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Treinando o algoritimo da arvore de classificacao
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion= 'entropy', random_state= 0)
classifier.fit(X_train, y_train)

# Prevemos os resultados do teste
y_pred = classifier.predict(X_test)

# Fazendo a matrix de confusao para verificar erros e acertos 'acertos[1,:]'
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizando os resultados do treino no grafico
'''from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start= X_set[:, 0].min() - 1, stop= X_set[:, 0].max() + 1, step= 0.01),
                     np.arange(start= X_set[:, 1].min() - 1, stop= X_set[:, 1].max() + 1, step= 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha= 0.75, cmap= ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.xlim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c= ListedColormap(('red', 'green'))(i), label= j)
plt.title('Classification Tree (Tranding training)')
plt.xlabel('Age')
plt.ylabel('Entimated Salary')
plt.legend()
plt.show()'''

"""# Visualizando os resultados dos teste no grafico
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start= X_set[:, 0].min() - 1, stop= X_set[:, 0].max() + 1, step= 0.01),
                     np.arange(start= X_set[:, 1].min() - 1, stop= X_set[:, 1].max() + 1, step= 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha= 0.75, cmap= ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.xlim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c= ListedColormap(('red', 'green'))(i), label= j)
plt.title('Classification tree (Tranding set)')
plt.xlabel('Age')
plt.ylabel('Entimated Salary (Training test)')
plt.legend()
plt.show()"""