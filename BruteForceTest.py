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
import KCenterTest
import pandas as pd

#read the data

X = np.genfromtxt('res/pulmon/features.csv',skip_header=1,dtype=object,delimiter=',')
y = X[:,3]
y = np.logical_or(y == y[8],  y == y[1]) #y holds string labels, binaries them: ethanol=y[1] or not
numerical_features_mask = np.ones(X.shape[1], dtype=bool)
numerical_features_mask[:7] = False
X = X[:,numerical_features_mask]
X = X.astype(float)

#standardize
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

#compute similarity graph
dists = scipy.spatial.distance.cdist(X, X, 'sqeuclidean')
dists_without_diagonal = np.reshape(dists[~np.eye(dists.shape[0],dtype=bool)],(dists.shape[0], dists.shape[1]-1))
sigma = np.average(np.sort(dists_without_diagonal)[:,:5])/3
W = np.exp(-(dists) / (2 * sigma ** 2))
np.fill_diagonal(W, 0)

np.nonzero(W.flatten())
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
upper_bound = 2*np.sum(W)
source_edges = np.transpose(np.vstack((s*(np.ones(W.shape[0])), np.array(range(W.shape[0])), np.zeros(W.shape[0]))))
sink_edges = np.transpose(np.vstack((np.array(range(W.shape[0])), t*(np.ones(W.shape[0])), np.zeros(W.shape[0]))))

weight = g.new_edge_property("long double")
eprops = [weight]
g.add_edge_list(edges, eprops=eprops)
g.add_edge_list(source_edges, eprops=eprops)
g.add_edge_list(sink_edges, eprops=eprops)


#first_vertex = np.random.randint(0, dists.shape[0])

best_iterative = np.inf
best_cut = np.inf

f=open("outputOneGraph.txt", "w+")

label_budget = 3
start_time = time.time()

visitor = MinCut.VisitorExample(g)

previous_L = None

for positions in combinations((range(X.shape[0])), label_budget):
    #if i > 0:
     #   positions = [int(idx) for idx in list(np.genfromtxt('res/benchmark/SSL,set='+str(dataset)+',splits,labeled=100.tab')[i-1] - 1)]
    #else:
    #print(positions)

    positions = list(positions)


    if np.sum(y[positions]) == label_budget or np.sum(y[positions]) == 0:
        #print("skipped, since label belong to one class")
        #print("==================================================")
        continue


    #positions = compute_k_center(dists, i*20,first_vertex)
    L = np.zeros((X.shape[0], 2))
    L[positions, 0] = 1 - y[positions]
    L[positions, 1] = y[positions]
    L = L.astype(int)

    local_start_time = time.time()
    predictionIterative = LocalAndGlobalTest.predict(X,L,W,0.5,200)
    iterative_time = time.time() - local_start_time
    prediction_faster_min_cut = KCenterTest.faster_min_cut_strategy(g,s,t,upper_bound,weight,W, L, previous_L, visitor)
    #g.shrink_to_fit()
    cut_time = time.time() - iterative_time - local_start_time


    err_iterative = np.dot(predictionIterative-y,predictionIterative-y)
    if err_iterative < best_iterative:
        best_iterative = err_iterative

    err_cut = np.dot(prediction_faster_min_cut - y, prediction_faster_min_cut - y)
    if err_cut < best_cut:
        best_cut = err_cut

    f.write(str(positions) + "," + str(err_iterative) + "," + str(err_cut) + "," + str(iterative_time) + "," + str(cut_time) + "\n")
    previous_L = L
    #print(err_iterative)
    #print(err_cut)
    #print("==================================================")

f.close()
print((best_iterative, best_cut))
print(time.time() - start_time)