import numpy as np
import graph_tool.flow
#return the centers as k-means++ would
#instead of just picking the furthest instance as new center as in k-center, this method uses the distances as probabilities.
import MinCut


def compute_k_means_plus_plus_init(dists, k, first_vertex):
    center = [first_vertex]
    # dists = dists[~np.eye(dists.shape[0],dtype=bool)].reshape(dists.shape[0],-1) #remove zero diagonal
    # np.fill_diagonal(dists, upper_bound)
    for i in range(1, k):
        mask = np.ones(dists.shape[0], dtype=bool)
        mask[center] = False
        if len(center) > 1:
            distsToCenters = dists[:, center]
            prob = np.min(distsToCenters, axis=1)**2
        else:
            prob = dists[center[0]] ** 2
        prob /= np.sum(prob)
        tempidx = np.random.choice(range(dists.shape[0]), p=prob)

        center.append(tempidx)  # + sum(j <= tempidx + 1 for j in center))

    return center


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

def uniform_strategy(dists, k, first_vertex):
    center = [first_vertex]
    for i in range(1,k):
        x = np.random.randint(dists.shape[0])
        while x in center:
            x = np.random.randint(dists.shape[0])
        center.append(x)
    return center

#should be called once, before any computations. Will add two helper nodes x,y and zero weight edges.
#The zero edges, will not change the mincut computations, but will be there in place for the psi computation.
#Not to confuse with almost the same construction for s,t in the mincut
def adjust_graph_to_compute_psi(g, n, weight):
    x = g.vertex_index[g.add_vertex()]
    y = g.vertex_index[g.add_vertex()]
    source_edges = np.transpose(np.vstack((x * (np.ones(n)), np.array(range(n), np.zeros(n)))))
    sink_edges = np.transpose(np.vstack((np.array(range(n)), y * (np.ones(n)), np.zeros(n))))

    g.add_edge_list(source_edges, eprops=[weight])
    g.add_edge_list(sink_edges, eprops=[weight])

    return x,y

def outgoing_edge_sum(g, T, weight):
    sum = 0
    for e in g.edges:
        if e.source() in T and e.target not in T:
            sum += weight[e]

#g graph, L vertex bool-propertymap in L or not
#make sure that
def compute_psis_cut(g, L, weight, visitor, n, x, y):
    #adjust graph
    #"contract" L
    old_weights_in_L = {}

    eps = np.min(weight.a[np.nonzero(weight.a)]) / (2*n)

    for v in L:
        for w in L:
            if v == w:
                continue
            old_weights_in_L[v,w] = weight[g.edge(v,w)]
            weight[g.edge(v,w)] = np.inf



    for v in L:
        weight[g.edge(v, y)] = np.inf

    T_prime = set(range(n)) - set(L)

    while True:
        T = T_prime
        weighted_cut = outgoing_edge_sum(g,T,weight)
        p = outgoing_edge_sum(g,T,weight)/len(T)

        for v in range(n):
            if v in L:
                continue
            weight[g.edge(x, v)] = p

        res = graph_tool.flow.push_relabel_max_flow(g, x, y, weight)
        T_prime = MinCut.own_min_cut(visitor, x, weight, res)

        if np.abs(outgoing_edge_sum(g,T_prime,weight) - p*len(T_prime)) < eps:
            break





    #undoing changes
    # "un-contract" L
    for v in L:
        for w in L:
            if v == w:
                continue
            weight[g.edge(v, w)] = old_weights_in_L[v,w]


    for v in range(n):
        weight[g.edge(x, v)] = 0
    return T

def psi_value_to_cut(g,T,weight):
    return outgoing_edge_sum(g,T,weight)/len(T)

def greedy_psi_maximizer(g, n, weight, visitor, k):
    choices = {}
    x,y = adjust_graph_to_compute_psi(g, n, weight)

    current_psi_value = 0

    for i in range(k):
        best_gain = 0

        next_greedy_node = None
        for v in range(n):

            if v in choices:
                continue

            candidate = compute_psis_cut(g,set(choices).add(v), weight, visitor, n, x, y)
            candidate_value = psi_value_to_cut(g,candidate, weight)
            if candidate - current_psi_value > best_gain:
                best_gain = candidate - current_psi_value
                next_greedy_node = v

        choices.add(next_greedy_node)

