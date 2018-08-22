import time

import networkx as nx
import numpy as np
from graph_tool import flow

import MinCut


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
    else:
        for v in range(W.shape[0]):
            e = g2.edge(s,v)
            weight[e] = 0
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


def local_global_strategy(X, Y, W, alpha, iterations=200, eps=0.000001):
    np.fill_diagonal(W,0)
    D = np.sum(W, axis=0)
    Dhalfinverse = 1 / np.sqrt(D)
    Dhalfinverse = np.diag(Dhalfinverse)
    S = np.dot(np.dot(Dhalfinverse, W), Dhalfinverse)

    F = np.zeros((X.shape[0], 2))
    oldF = np.ones((X.shape[0], 2))
    oldF[:2, :2] = np.eye(2)
    i = 0
    while (np.abs(oldF - F) > eps).any() or i >= iterations:
        oldF = F
        F = np.dot(alpha * S, F) + (1 - alpha) * Y

    result = np.zeros(X.shape[0])
    #uniform argmax
    for i in range(X.shape[0]):
        result[i] = np.random.choice(np.flatnonzero(F[i] == F[i].max()))

    return result

    #return np.argmax(F, axis=1)