import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from perceptron import Perceptron


df = pd.read_csv("./Projeto Pratico/treinamento.txt")
df2 = open('./Projeto Pratico/teste.txt','r')

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    print ('Previsto: ', y_pred)
    print ('Esperado: ', y_true)
    return accuracy*100

X, y = datasets.make_blobs(n_samples=150,n_features=2,centers=2,cluster_std=1.05,random_state=2)
# X_train = df.iloc[:,0:3]
# y_train = df.iloc[:,3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)
# predictions = p.predict(X_test)

# print("Perceptron classification accuracy", accuracy(y_test, predictions))

