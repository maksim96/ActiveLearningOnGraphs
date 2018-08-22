import numpy as np
#from sympy.core.basic import preorder_traversal

import LocalAndGlobalTest
import scipy.spatial
import networkx as nx
import graph_tool as gt
from itertools import combinations

import PredictionStrategies
import MinCut
import SelectionStrategies

if __name__ == "__main__":
    dataset = 4

    X = np.genfromtxt('res/benchmark/SSL,set='+str(dataset)+',X.tab')
    #standardize
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = 0.5*(np.genfromtxt('res/benchmark/SSL,set='+str(dataset)+',y.tab')+1) #1 --> 1, -1 --> 0

    dists = scipy.spatial.distance.cdist(X, X)

    #average_knn_dist = np.average(np.sort(X)[:,:5])
    dists_without_diagonal = np.reshape(dists[~np.eye(dists.shape[0], dtype=bool)], (dists.shape[0], dists.shape[1] - 1))

    sigma =np.average(np.sort(dists_without_diagonal)[:,:5])/3
    W = np.exp(-(dists) ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)
    #W2 = np.copy(W) less edges is slower strangely
    #W2[W2 <= 0.1] = 0

    upper_bound = 2*np.sum(W)

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
    source_edges = np.transpose(np.vstack((s * (np.ones(W.shape[0])), np.array(range(W.shape[0])), np.zeros(W.shape[0]))))
    sink_edges = np.transpose(np.vstack((np.array(range(W.shape[0])), t * (np.ones(W.shape[0])), np.zeros(W.shape[0]))))

    g2.add_edge_list(source_edges, eprops=eprops)
    g2.add_edge_list(sink_edges, eprops=eprops)

    first_vertex = np.random.randint(0, dists.shape[0])

    best_iterative = 1000
    best_cut = 1000

    visitor = MinCut.VisitorExample(g2)

    for i in range(2,21):
         #   positions = [int(idx) for idx in list(np.genfromtxt('res/benchmark/SSL,set='+str(dataset)+',splits,labeled=100.tab')[i-1] - 1)]
        #else:

        positions = SelectionStrategies.compute_k_center(dists, i,first_vertex)

        while np.sum(y[positions]) == i or np.sum(y[positions]) == 0:
            first_vertex = np.random.randint(0, dists.shape[0])
            positions = SelectionStrategies.compute_k_center(dists, i, first_vertex)
            # print("skipped, since label belong to one class")
            # print("==================================================")
            #continue

        L = np.zeros((X.shape[0], 2))
        L[positions, 0] = 1 - y[positions]
        L[positions, 1] = y[positions]
        L = L.astype(int)

        predictionIterative = PredictionStrategies.local_global_strategy(X, L, W, 0.5, 200)
        #prediction_min_cut = mincut_strategy(W, L)
        prediction_faster_min_cut = PredictionStrategies.faster_min_cut_strategy(g2,s,t,upper_bound,weight,W, L, None, visitor)
        #prediction_random = np.random.randint(2, size=prediction_faster_min_cut.shape)

        err_iterative = np.dot(predictionIterative-y,predictionIterative-y)
        if err_iterative < best_iterative:
            best_iterative = err_iterative

        err_cut = np.dot(prediction_faster_min_cut - y, prediction_faster_min_cut - y)
        if err_cut < best_cut:
            best_cut = err_cut

        print((err_iterative, err_cut))
        #print("==================================================")


    print((best_iterative, best_cut))
