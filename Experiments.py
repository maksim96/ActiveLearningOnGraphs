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

def test_all_strategies(strategies, repitions, dataset, label_budget):
    results = []
    X = np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',X.tab')
    #X = np.genfromtxt('res/pulmon/features.csv', skip_header=1, dtype=object, delimiter=',')
    #y = X[:, 3]
    #y = np.logical_or(y == y[8], y == y[1])  # y holds string labels, binaries them: ethanol=y[1] or not
    #numerical_features_mask = np.ones(X.shape[1], dtype=bool)
    #numerical_features_mask[:7] = False
    #X = X[:, numerical_features_mask]
    #X = X.astype(float)
    # standardize
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = 0.5 * (np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',y.tab') + 1)  # 1 --> 1, -1 --> 0

    dists = scipy.spatial.distance.cdist(X, X)

    # average_knn_dist = np.average(np.sort(X)[:,:5])
    dists_without_diagonal = np.reshape(dists[~np.eye(dists.shape[0], dtype=bool)],
                                        (dists.shape[0], dists.shape[1] - 1))
    sigma = np.average(np.sort(dists_without_diagonal)[5]) / 3

    W = np.exp(-(dists) ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)
    # W2 = np.copy(W) less edges is slower strangely
    # W2[W2 <= 0.1] = 0

    upper_bound = 2 * np.sum(W)

    weights = W.flatten()
    a, b = W.nonzero()
    edges = np.transpose(np.vstack((a, b, weights[weights != 0])))

    np.random.seed(0)

    g = gt.Graph()

    # construct actual graph
    g.add_vertex(W.shape[0])
    # add source, sink
    s = g.vertex_index[g.add_vertex()]
    t = g.vertex_index[g.add_vertex()]
    weight = g.new_edge_property("long double")
    eprops = [weight]
    g.add_edge_list(edges, eprops=eprops)
    source_edges = np.transpose(
        np.vstack((s * (np.ones(W.shape[0])), np.array(range(W.shape[0])), np.zeros(W.shape[0]))))
    sink_edges = np.transpose(np.vstack((np.array(range(W.shape[0])), t * (np.ones(W.shape[0])), np.zeros(W.shape[0]))))

    g.add_edge_list(source_edges, eprops=eprops)
    g.add_edge_list(sink_edges, eprops=eprops)



    best_iterative = 1000000000000000
    best_cut = 1000000000000000

    visitor = MinCut.VisitorExample(g)
    for selection_strategy in strategies:
        current_results = []
        np.random.seed(0)
        for i in range(2, repitions+2):
            #   positions = [int(idx) for idx in list(np.genfromtxt('res/benchmark/SSL,set='+str(dataset)+',splits,labeled=100.tab')[i-1] - 1)]
            # else:

            first_vertex = np.random.randint(dists.shape[0])
            positions = selection_strategy(dists, label_budget, first_vertex)

            while np.sum(y[positions]) == label_budget or np.sum(y[positions]) == 0:
                first_vertex = np.random.randint(dists.shape[0])
                positions = selection_strategy(dists, label_budget, first_vertex)
                # print("skipped, since label belong to one class")
                # print("==================================================")
                # continue

            L = np.zeros((X.shape[0], 2))
            L[positions, 0] = 1 - y[positions]
            L[positions, 1] = y[positions]
            L = L.astype(int)

            predictionIterative = PredictionStrategies.local_global_strategy(X, L, W, 0.5)
            # prediction_min_cut = mincut_strategy(W, L)
            prediction_faster_min_cut = PredictionStrategies.faster_min_cut_strategy(g, s, t, upper_bound, weight, W, L,
                                                                                     None, visitor)
            # prediction_random = np.random.randint(2, size=prediction_faster_min_cut.shape)

            err_iterative = np.dot(predictionIterative - y, predictionIterative - y)
            if err_iterative < best_iterative:
                best_iterative = err_iterative
                print((positions, best_cut))

            err_cut = np.dot(prediction_faster_min_cut - y, prediction_faster_min_cut - y)
            if err_cut < best_cut:
                best_cut = err_cut
                print((positions, best_cut))

            current_results.append([err_iterative, err_cut])
            print(i)

        results.append(current_results)
    return results

if __name__ == "__main__":
    [resA,resB,resC] = test_all_strategies([SelectionStrategies.compute_k_center, SelectionStrategies.compute_k_means_plus_plus_init,   SelectionStrategies.uniform_strategy], 20, 2, 150 )
    resA = np.array(resA)
    resB = np.array(resB)
    print("400 instances, label_budget=6")
    print("[error label propagation, error min cut]")
    print("k-center:")
    print("max: ", np.max(resA, axis=0))
    print("min: ",np.min(resA, axis=0))
    print("avg: ",np.average(resA, axis=0))
    print("med: ",np.median(resA, axis= 0))
    print("std: ", np.std(resA, axis=0))
    print("k-means:")
    print("max: ",np.max(resB, axis=0))
    print("min: ",np.min(resB, axis=0))
    print("avg: ",np.average(resB, axis=0))
    print("med: ",np.median(resB, axis=0))
    print("std: ", np.std(resB, axis=0))
    print("uniform:")
    print("max: ",np.max(resC, axis=0))
    print("min: ",np.min(resC, axis=0))
    print("avg: ",np.average(resC, axis=0))
    print("med: ",np.median(resC, axis=0))
    print("std: ", np.std(resC, axis=0))