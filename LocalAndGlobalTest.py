import numpy as np

import matplotlib.pyplot as plt
#import sklearn.datasets
import scipy.spatial
from sklearn.datasets import make_moons

if __name__ == "__main__":

    X,y = make_moons(n_samples=200, shuffle=True, noise=0.05, random_state=None)



    sigma = 0.03

    W = np.exp(-(scipy.spatial.distance.cdist(X, X)) ** 2 / (2 * sigma ** 2))
    D = np.sum(W,axis=0)
    Dhalfinverse = 1/np.sqrt(D)
    Dhalfinverse = np.diag(Dhalfinverse)
    S = np.dot(np.dot(Dhalfinverse, W), Dhalfinverse)

    Y = np.zeros((200,2))
    Y[0,0] = 1
    Y[100,1] = 1

    alpha = 0.9
    F = np.zeros((200,2))
    oldF = np.ones((200,2))
    oldF[:2,:2] = np.eye(2)
    i = 0
    eps = 0.000001
    while (np.abs(oldF - F) > eps).any():
        labels = np.argmax(F, axis=1)

        predict_pos = X[labels == 0]
        predict_neg = X[labels == 1]

        plt.scatter(predict_pos[:, 0], predict_pos[:, 1], c='b')

        plt.scatter(predict_neg[:, 0], predict_neg[:, 1], c='r')
        oldF = F
        F = np.dot(alpha*S,F) + (1 - alpha)*Y

        if i % 5 == 0:
            plt.figure()
        i += 1
    labels = np.argmax(F, axis=1)

    predict_pos = X[labels==0]
    predict_neg = X[labels==1]

    plt.scatter(predict_pos[:,0], predict_pos[:,1], c='b')


    plt.scatter(predict_neg[:,0], predict_neg[:,1], c='r')


    plt.show()