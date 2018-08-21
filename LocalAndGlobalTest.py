import numpy as np
import matplotlib.pyplot as plt

import numpy as np

import matplotlib.pyplot as plt
#import sklearn.datasets
import scipy.spatial

def predict(X,Y, W, alpha, iterations):
    np.fill_diagonal(W,0)
    D = np.sum(W, axis=0)
    Dhalfinverse = 1 / np.sqrt(D)
    Dhalfinverse = np.diag(Dhalfinverse)
    S = np.dot(np.dot(Dhalfinverse, W), Dhalfinverse)

    F = np.zeros((X.shape[0], 2))
    for i in range(iterations):
        F = np.dot(alpha * S, F) + (1 - alpha) * Y

    result = np.zeros(X.shape[0])
    #uniform argmax
    for i in range(X.shape[0]):
        result[i] = np.random.choice(np.flatnonzero(F[i] == F[i].max()))

    return result

    #return np.argmax(F, axis=1)
'''
X2 = np.random.randn(100,2)

X1 = np.random.randn(100,2) + (3,3)
plt.scatter(X1[:,0], X1[:,1], c='b')

plt.scatter(X2[:,0], X2[:,1], c='r')

plt.figure()

X = np.concatenate((X1,X2))

X,y = sklearn.datasets.make_moons(n_samples=200, shuffle=True, noise=0.05, random_state=None)



sigma = 0.02

W = np.exp(-(scipy.spatial.distance.cdist(X, X)) ** 2 / (2 * sigma ** 2))
D = np.sum(W,axis=0)
Dhalfinverse = 1/np.sqrt(D)
Dhalfinverse = np.diag(Dhalfinverse)
S = np.dot(np.dot(Dhalfinverse, W), Dhalfinverse)

Y = np.zeros((200,2))
Y[0,0] = 1
Y[100,1] = 1

alpha = 0.5
F = np.zeros((200,2))
for i in range(50):
    labels = np.argmax(F, axis=1)

    predict_pos = X[labels == 0]
    predict_neg = X[labels == 1]

    plt.scatter(predict_pos[:, 0], predict_pos[:, 1], c='b')

    plt.scatter(predict_neg[:, 0], predict_neg[:, 1], c='r')
    F = np.dot(alpha*S,F) + (1 - alpha)*Y

    plt.figure()

labels = np.argmax(F, axis=1)

predict_pos = X[labels==0]
predict_neg = X[labels==1]

plt.scatter(predict_pos[:,0], predict_pos[:,1], c='b')


plt.scatter(predict_neg[:,0], predict_neg[:,1], c='r')


plt.show()'''