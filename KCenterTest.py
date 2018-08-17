import numpy as np
import matplotlib.pyplot as plt
import graph_tool
#from sympy.core.basic import preorder_traversal

import LocalAndGlobalTest
import scipy.spatial
import networkx as nx
import time
import graph_tool as gt
from graph_tool import flow

def compute_k_center(dists, k):
    center = [np.random.randint(0,dists.shape[0])]
    #dists = dists[~np.eye(dists.shape[0],dtype=bool)].reshape(dists.shape[0],-1) #remove zero diagonal
    #np.fill_diagonal(dists, upper_bound)
    for i in range(1,k):
        mask = np.ones(dists.shape[0], dtype=bool)
        mask[center] = False
        if len(center) > 1:
            distsToCenters = dists[:,center]
            tempidx = np.argmax(np.min(distsToCenters, axis=1))
        else:
            tempidx = np.argmax(dists[center[0],mask])
        center.append(tempidx) #+ sum(j <= tempidx + 1 for j in center))

    return center



def mincut_strategy(W, L):
    start = time.time()
    for i in range(W.shape[0]):
        if L[i][0] == 1:
            g.add_edge(i, 's', weight=upper_bound)
        elif L[i][1] == 1:
            g.add_edge('t', i, weight=upper_bound)
        else:
            if g.has_edge(i, 's'):
                g.remove_edge(i, 's')
            if g.has_edge('t', i):
                g.remove_edge('t', i)
    mid = time.time()
    print(mid - start)

    cut_value, (zero_class, positive_class) = nx.minimum_cut(g, 's', 't', capacity='weight')
    print("--> "+ str(cut_value))
    cut_time = time.time()
    print(cut_time - mid)

    prediction = np.zeros(W.shape[0])
    for i in zero_class:
        if (i != 's'):
            prediction[i] = 0
    for i in positive_class:
        if (i != 't'):
            prediction[i] = 1
    print(time.time() - cut_time)
    return prediction

def faster_min_cut_strategy(W, L):
    start = time.time()

    g2.clear_vertex(s)
    g2.clear_vertex(t)

    zeroExampleCount = np.sum(L[:,0])
    positiveExampleCount = np.sum(L[:,1])

    sourceEdges = np.transpose(np.vstack((s*np.ones(zeroExampleCount), L[:,0].nonzero()[0], upper_bound * np.ones(zeroExampleCount))))
    sinkEdges = np.transpose(np.vstack((L[:,1].nonzero()[0], t * np.ones(positiveExampleCount), upper_bound * np.ones(positiveExampleCount))))
    g2.add_edge_list(sourceEdges, eprops=eprops)
    g2.add_edge_list(sinkEdges, eprops=eprops)

    mid = time.time()
    print(mid - start)

    res = flow.push_relabel_max_flow(g2, s, t, weight)
    part = flow.min_st_cut(g2, s, weight, res)

    max_flow = sum((weight[e] - res[e]) for e in g2.vertex(t).in_edges())
    print("-->:" +str(max_flow))

    cut_time = time.time()
    print(cut_time - mid)

    prediction = np.zeros(W.shape[0])
    for i in range(W.shape[0]):
        if part[i]:
            prediction[i] = 0
        else:
            prediction[i] = 1
    print(time.time() - cut_time)
    return prediction

dataset = 4

X = np.genfromtxt('res/benchmark/SSL,set='+str(dataset)+',X.tab')
y = 0.5*(np.genfromtxt('res/benchmark/SSL,set='+str(dataset)+',y.tab')+1) #1 --> 1, -1 --> 0

dists = scipy.spatial.distance.cdist(X, X)

average_knn_dist = np.average(np.sort(X)[:,:5])

sigma = np.min(dists[np.nonzero(dists)])
W = np.exp(-(dists) ** 2 / (2 * sigma ** 2))
np.fill_diagonal(W, 0)
#W2 = np.copy(W) less edges is slower strangely
#W2[W2 <= 0.1] = 0

g = nx.from_numpy_matrix(W)
upper_bound = 200 * np.sum(W)
g.add_node('s')
g.add_node('t')

np.nonzero(W.flatten())
weights = W.flatten()
weights[weights != 0]
a, b = W.nonzero()
edges = np.transpose(np.vstack((a, b, weights[weights != 0])))


g2 = gt.Graph()

# construct actual graph
g2.add_vertex(W.shape[0])
# add source, sink
s = g2.vertex_index[g2.add_vertex()]
t = g2.vertex_index[g2.add_vertex()]
weight = g2.new_edge_property("double")
eprops = [weight]
g2.add_edge_list(edges, eprops=eprops)


for i in range(11):
    if i > 0:
        positions = [int(idx) for idx in list(np.genfromtxt('res/benchmark/SSL,set='+str(dataset)+',splits,labeled=10.tab')[i-1] - 1)]
    else:
        positions = compute_k_center(dists, 100)
    L = np.zeros((X.shape[0], 2))
    L[positions, 0] = 1 - y[positions]
    L[positions, 1] = y[positions]
    L = L.astype(int)

    predictionIterative = LocalAndGlobalTest.predict(X,L,sigma,0.5,200)
    prediction_min_cut = mincut_strategy(W, L)
    prediction_faster_min_cut = faster_min_cut_strategy(W, L)
    prediction_random = np.random.randint(2, size=prediction_min_cut.shape)


    print(np.dot(predictionIterative-y,predictionIterative-y))
    print(np.dot(prediction_min_cut - y, prediction_min_cut - y))
    print(np.dot(prediction_faster_min_cut - y, prediction_faster_min_cut - y))
    print(np.dot(prediction_random-y,prediction_random-y))
    print("==================================================")