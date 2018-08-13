import numpy as np
import matplotlib.pyplot as plt
from sympy.core.basic import preorder_traversal

import LocalAndGlobalTest
import scipy.spatial
import networkx as nx

def mincut_strategy(W, L):
    g = nx.from_numpy_matrix(W)
    upper_bound = 2*np.sum(W)
    g.add_node('s')
    g.add_node('t')

    for i in range(W.shape[0]):
        if L[i][0] == 1:
            g.add_edge(i, 's', {'weight': upper_bound})
        elif L[i][1] == 1:
            g.add_edge('t', i, {'weight': upper_bound})

    cut_value, (zero_class, positive_class) = nx.minimum_cut(g, 's', 't', capacity='weight')
    prediction = np.zeros(W.shape[0])
    for i in zero_class:
        prediction[i] = 0
    for i in positive_class:
        prediction[i] = 1

    return prediction

X = np.genfromtxt('res/benchmark/SSL,set=2,X.tab')
y = 0.5*(np.genfromtxt('res/benchmark/SSL,set=2,y.tab')+1)

L = np.zeros((1500,2))

positions = list(np.genfromtxt('res/benchmark/SSL,set=2,splits,labeled=10.tab', dtype=int))[0] #zeroth row of this file, use any line you want

L[positions,0] = 1 - y[positions]
L[positions,1] = y[positions]

dists = scipy.spatial.distance.cdist(X, X)

sigma = np.min(dists[np.nonzero(dists)])*2
W = np.exp(-(dists) ** 2 / (2 * sigma ** 2))
np.fill_diagonal(W, 0)

predictionIterative = LocalAndGlobalTest.predict(X,L,0.5,0.1,200)
prediction_min_cut = mincut_strategy(W, L)

print(np.dot(predictionIterative-y,predictionIterative-y))
print(np.dot(prediction_min_cut-y,prediction_min_cut-y))