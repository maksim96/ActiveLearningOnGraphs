import numpy as np

#return the centers as k-means++ would
#instead of just picking the furthest instance as new center as in k-center, this method uses the distances as probabilities.
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