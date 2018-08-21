import numpy as np
import matplotlib.pyplot as plt
import graph_tool
#from sympy.core.basic import preorder_traversal

import LocalAndGlobalTest
import MinCut
import scipy.spatial
import networkx as nx
import time
import graph_tool as gt
from graph_tool import flow
from itertools import combinations

def compute_k_center(dists, k, first_vertex):
    center = [first_vertex]
    #dists = dists[~np.eye(dists.shape[0],dtype=bool)].reshape(dists.shape[0],-1) #remove zero diagonal
    #np.fill_diagonal(dists, upper_bound)
    for i in range(1,k):
        mask = np.ones(dists.shape[0], dtype=bool)
        mask[center] = False
        if len(center) > 1:
            distsToCenters = dists[:,center]
            tempidx = np.argmax(np.min(distsToCenters, axis=1))
        else:
            tempidx = np.argmax(dists[center[0]])
        center.append(tempidx) #+ sum(j <= tempidx + 1 for j in center))

    return center

def mincut_strategy(g,upper_bound,W, L):
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


def faster_min_cut_strategy(g2,s,t,upper_bound,weight,W, L, previous_L, visitor):
    start = time.time()

    #instead of deleting edges here and adding some others later again,
    #adding all edges (s,x) (x,t) with infinite weight first and then just adapt the weights, instead of changing the edge vector all the time
    #no reindex, refitting, etc. needed anymore.
    #g2.clear_vertex(s)
    #g2.clear_vertex(t)

    #zeroExampleCount = np.sum(L[:,0])
    #positiveExampleCount = np.sum(L[:,1])

    #g2.reindex_edges()

    #sourceEdges = np.transpose(np.vstack((s*np.ones(zeroExampleCount), L[:,0].nonzero()[0], upper_bound * np.ones(zeroExampleCount))))
    #sinkEdges = np.transpose(np.vstack((L[:,1].nonzero()[0], t * np.ones(positiveExampleCount), upper_bound * np.ones(positiveExampleCount))))
    #g2.add_edge_list(sourceEdges, eprops=[weight])
    #g2.add_edge_list(sinkEdges, eprops=[weight])

    #print((g2.num_edges(), g2.edge_index_range))

    if previous_L is not None:
        for v in previous_L[:,0].nonzero()[0]:
            e = g2.edge(s,v)
            weight[e] = 0
        for v in previous_L[:, 1].nonzero()[0]:
            e = g2.edge(v,t)
            weight[e] = 0

    for v in L[:, 0].nonzero()[0]:
        e = g2.edge(s, v)
        weight[e] = upper_bound
    for v in L[:, 1].nonzero()[0]:
        e = g2.edge(v, t)
        weight[e] = upper_bound

    mid = time.time()
    #print(mid - start)
    start = time.time()

    res = flow.push_relabel_max_flow(g2, s, t, weight)
    #max_flow = sum((weight[e] - res[e]) for e in g2.vertex(t).in_edges())
    #print("-->:" + str(max_flow))

    part = MinCut.own_min_cut(visitor, s, weight, res)

    cut_time = time.time()
    #print(cut_time - mid)

    prediction = np.ones(W.shape[0])
    prediction[part] = 0
    #print(time.time() - cut_time)
    return prediction


'''
dataset = 4

X = np.genfromtxt('res/benchmark/SSL,set='+str(dataset)+',X.tab')
#standardize
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
y = 0.5*(np.genfromtxt('res/benchmark/SSL,set='+str(dataset)+',y.tab')+1) #1 --> 1, -1 --> 0

dists = scipy.spatial.distance.cdist(X, X)

#average_knn_dist = np.average(np.sort(X)[:,:5])

sigma = 1#np.min(dists[np.nonzero(dists)])
W = np.exp(-(dists) ** 2 / (2 * sigma ** 2))
np.fill_diagonal(W, 0)
#W2 = np.copy(W) less edges is slower strangely
#W2[W2 <= 0.1] = 0

g = nx.from_numpy_matrix(W)
upper_bound = 2*np.sum(W)
g.add_node('s')
g.add_node('t')

np.nonzero(W.flatten())
weights = W.flatten()
a, b = W.nonzero()
edges = np.transpose(np.vstack((a, b, weights[weights != 0])))

np.random.seed(0)

g2 = gt.Graph()

# construct actual graph
g2.add_vertex(W.shape[0])
# add source, sink
s = g2.vertex_index[g2.add_vertex()]
t = g2.vertex_index[g2.add_vertex()]
weight = g2.new_edge_property("long double")
eprops = [weight]
g2.add_edge_list(edges, eprops=eprops)

first_vertex = np.random.randint(0, dists.shape[0])

best_iterative = 1000
best_cut = 1000

for positions in combinations((range(X.shape[0])), 2):
    #if i > 0:
     #   positions = [int(idx) for idx in list(np.genfromtxt('res/benchmark/SSL,set='+str(dataset)+',splits,labeled=100.tab')[i-1] - 1)]
    #else:
    if positions[1] % 50 == 0:
        print(positions)
    positions = list(positions)

    #positions = compute_k_center(dists, i*20,first_vertex)
    L = np.zeros((X.shape[0], 2))
    L[positions, 0] = 1 - y[positions]
    L[positions, 1] = y[positions]
    L = L.astype(int)

    predictionIterative = LocalAndGlobalTest.predict(X,L,sigma,0.5,200)
    #prediction_min_cut = mincut_strategy(W, L)
    prediction_faster_min_cut = faster_min_cut_strategy(W, L)
    #prediction_random = np.random.randint(2, size=prediction_faster_min_cut.shape)

    err_iterative = np.dot(predictionIterative-y,predictionIterative-y)
    if err_iterative < best_iterative:
        best_iterative = err_iterative

    err_cut = np.dot(prediction_faster_min_cut - y, prediction_faster_min_cut - y)
    if err_cut < best_cut:
        best_iterative = err_cut

    #print(err_iterative)
    #print(np.dot(prediction_min_cut - y, prediction_min_cut - y))
    #print(err_cut)
    #print(np.dot(prediction_random-y,prediction_random-y))
    #print("==================================================")


print((best_iterative, best_cut))'''
