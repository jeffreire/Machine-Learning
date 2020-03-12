import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""arquivo = pd.read_csv('C:/Users/jefferson.maria/Desktop/MachineLearning/Machine Learning/MachineStudy/Salary_Data.csv')
x = arquivo.iloc[:,0]
y = arquivo.iloc[:,1]"""

x = [ 1.1, 1.3, 1.5, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1,
5.3, 6.0, 6.8, 7.1, 4.5, 7.9, 8.2, 8.7, 9.0, 9.6, 10.3, 10.5 ]

y = [ 39343.00, 46205.00, 37731.00, 39891.00, 56642.00,  60150.00,
54445.00, 64445.00, 57189.00, 63218.00, 55794.00, 56957.00, 57081.00, 61111.00,
67938.00, 66029.00, 83088.00, 93940.00, 91738.00, 98273.00, 61111.00,
101302.00, 113812.00, 109431.00, 105582.00, 112635.00, 122391.00, 121872.00]

x_t = [2.0, 5.9, 9.5]
y_t = [43525.00, 81363.00, 116969.00]

st = len(x)
xy=0
xsoma = 0
for i in range(len(x)): xy += x[i]*y[i]
for i in x: xsoma += (i)**2

m_n = st*xy - (sum(x)*sum(y))
m_d = st*xsoma-(sum(x)**2)
m = m_n/m_d

b_n = (xsoma*sum(y))-(sum(x)*xy)
b_d = (st*xsoma)-((sum(x)**2))
b = b_n/b_d
x1 = 9.0
s= m*x1+b
print('y:', round(s,2))

y_pred = []
for p in x_t:
    s = m * p + b
    y_pred.append(s)

plt.scatter(x, y)
"""plt.plot(x, y, color='black')"""
plt.plot(x_t, y_t, color='black') 
plt.plot(x_t, y_pred, color='red')
plt.xlabel("Idade")
plt.ylabel("Salary")
plt.show()

#Para obter os dados dentro de um arquivo e nao adicionar los em forma manual
"""t = open('C:/Users/jefferson.maria/Desktop/MachineLearning/Machine Learning/MachineStudy/Salary_Data.csv', 'r')
tl = [i.strip().split(',') for i in t]
x = []
y = []"""

"""for i in tl:
    if i != ['YearsExperience', 'Salary']:
        x.append(i[0])
        y.append(float(i[1]))"""

"""
print(x, '\n', '*'*30, '\n', y)
t.close()"""