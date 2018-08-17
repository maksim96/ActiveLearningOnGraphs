# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import graph_tool
# from sympy.core.basic import preorder_traversal

import scipy.spatial
import networkx as nx
import time
import graph_tool as gt
from graph_tool import flow
from graph_tool.all import graph_draw

class VisitorExample(gt.search.BFSVisitor):

    reachable_vertices = []

    first_time = True #we don't want to add the root

    def discover_vertex(self, u):
        if self.first_time:
            self.first_time = False
        else:
            self.reachable_vertices.append(g2.vertex_index[u])


def compute_k_center(dists, k):
    center = [np.random.randint(0, dists.shape[0])]
    # dists = dists[~np.eye(dists.shape[0],dtype=bool)].reshape(dists.shape[0],-1) #remove zero diagonal
    # np.fill_diagonal(dists, upper_bound)
    for i in range(1, k):
        mask = np.ones(dists.shape[0], dtype=bool)
        mask[center] = False
        if len(center) > 1:
            distsToCenters = dists[:, center]
            tempidx = np.argmax(np.min(distsToCenters, axis=1))
        else:
            tempidx = np.argmax(dists[center[0]])
            print((np.argmax(dists[center[0]]), np.max(dists[center[0]])))
        center.append(tempidx)  # + sum(j <= tempidx + 1 for j in center))

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
    print(zero_class)
    print("--> " + str(cut_value))
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

    zeroExampleCount = np.sum(L[:, 0])
    positiveExampleCount = np.sum(L[:, 1])

    sourceEdges = np.transpose(
        np.vstack((s * np.ones(zeroExampleCount), L[:, 0].nonzero()[0], upper_bound * np.ones(zeroExampleCount))))
    sinkEdges = np.transpose(np.vstack(
        (L[:, 1].nonzero()[0], t * np.ones(positiveExampleCount), upper_bound * np.ones(positiveExampleCount))))
    g2.add_edge_list(sourceEdges, eprops=eprops)
    g2.add_edge_list(sinkEdges, eprops=eprops)

    mid = time.time()
    print(mid - start)

    res = flow.push_relabel_max_flow(g2, s, t, weight)
    #eps = np.percentile(res.a[np.nonzero(res.a)],10) #cut off all values smaller than the 2-percentile of all values greater 0
    #roundedres = res.a
    #roundedres[res.a < eps] = 0
    #res.a[res.a < eps] = 0
    #almost_full = np.greater(res.a, weight.a - eps)
    #res.a[almost_full] = weight.a[almost_full]

    pos = g2.new_vertex_property("vector<double>")
    pos.set_2d_array(np.transpose(np.vstack((X, np.array([-3,0]), np.array([8,0])))))

    graph_draw(g2, pos=pos, edge_pen_width=gt.draw.prop_to_size(res), output="output.pdf")

    part = flow.min_st_cut(g2, s, weight, res) #part[i] == true <=> vertex i is in the partition of the source s

    nonzero_res = g2.new_edge_property("bool")
    nonzero_res.a = res.a > 0

    g2.set_edge_filter(nonzero_res)


    visitor = VisitorExample()

    gt.search.bfs_search(g2, s, visitor)

    print(visitor.reachable_vertices)

    max_flow = sum((weight[e] - res[e]) for e in g2.vertex(t).in_edges())
    print("-->:" + str(max_flow))

    cut_time = time.time()
    print(cut_time - mid)

    print(part.a)

    prediction = np.ones(W.shape[0])
    '''for i in range(W.shape[0]):
        if part[i]:
            prediction[i] = 0
        else:
            prediction[i] = 1'''
    prediction[visitor.reachable_vertices] = 0
    print(time.time() - cut_time)
    return prediction


# In[2]:


def local_global_strategy(X, Y, sigma, alpha, iterations):
    W = np.exp(-(scipy.spatial.distance.cdist(X, X)) ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)
    D = np.sum(W, axis=0)
    Dhalfinverse = 1 / np.sqrt(D)
    Dhalfinverse = np.diag(Dhalfinverse)
    S = np.dot(np.dot(Dhalfinverse, W), Dhalfinverse)

    alpha = 0.5
    F = np.zeros((X.shape[0], 2))
    for i in range(iterations):
        F = np.dot(alpha * S, F) + (1 - alpha) * Y

    result = np.zeros(X.shape[0])
    # uniform argmax
    for i in range(X.shape[0]):
        result[i] = np.random.choice(np.flatnonzero(F[i] == F[i].max()))

    return result


# In[3]:


np.random.seed(0)
dataset = 4
X = np.vstack((np.random.randn(100, 2), np.random.randn(100, 2) + np.array([3, 0])))
y = np.zeros(100)
y = np.append(y, np.ones(100))

dists = scipy.spatial.distance.cdist(X, X)

average_knn_dist = np.average(np.sort(X)[:, :5])

sigma = 0.1  # np.min(dists[np.nonzero(dists)])
W = np.exp(-((dists) / (np.sqrt(2) * sigma) ** 2))
np.fill_diagonal(W, 0)
# W2 = np.copy(W) less edges is slower strangely
# W2[W2 <= 0.1] = 0

plt.scatter(X[:100, 0], X[:100, 1])
plt.scatter(X[100:, 0], X[100:, 1], c='r')


(W, -(dists) / (np.sqrt(2) * sigma) ** 2)

# In[4]:


g = nx.from_numpy_matrix(W)
upper_bound = np.sum(W)
g.add_node('s')
g.add_node('t')

np.nonzero(W.flatten())
weights = W.flatten()
weights[weights != 0]
a, b = W.nonzero()
edges = np.transpose(np.vstack((a, b, weights[weights != 0])))

# In[5]:


g2 = gt.Graph()

# construct actual graph
g2.add_vertex(W.shape[0])
# add source, sink
s = g2.vertex_index[g2.add_vertex()]
t = g2.vertex_index[g2.add_vertex()]
weight = g2.new_edge_property("long double")
eprops = [weight]
g2.add_edge_list(edges, eprops=eprops)

# In[6]:


positions = compute_k_center(dists, 20)
print(X[positions, :])
plt.scatter(X[positions,0], X[positions,1], alpha=0.5, s=20, c='g')
plt.show()
L = np.zeros((X.shape[0], 2))
L[positions, 0] = 1 - y[positions]
L[positions, 1] = y[positions]
L = L.astype(int)

# In[7]:


predictionIterative = local_global_strategy(X, L, sigma, 0.5, 200)
prediction_min_cut = mincut_strategy(W, L)
prediction_faster_min_cut = faster_min_cut_strategy(W, L)
prediction_random = np.random.randint(2, size=prediction_min_cut.shape)

print(np.dot(predictionIterative - y, predictionIterative - y))
print(np.dot(prediction_min_cut - y, prediction_min_cut - y))
print(np.dot(prediction_faster_min_cut - y, prediction_faster_min_cut - y))
print(np.dot(prediction_random - y, prediction_random - y))
print("==================================================")

