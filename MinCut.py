import graph_tool as gt
from graph_tool.search import BFSVisitor
import numpy as np

class VisitorExample(BFSVisitor):



    def __init__(self, graph):
        self.g = graph
        self.reachable_vertices = []
        self.first_time = True  # we don't want to add the root
        self.nonzero_res = graph.new_edge_property("bool")
        self.eps = 0  # np.percentile(res.a[np.nonzero(res.a)], 90)


    def discover_vertex(self, u):
        #although this is python code all the time (instead of c++) it doesn't matter at all
        #as the actual flow computation takes much more time
        #==> no need to optimize this here
        if self.first_time:
            self.first_time = False
        else:
            self.reachable_vertices.append(self.g.vertex_index[u])

def own_min_cut(visitor, s, cap, res):
    visitor.first_time = True
    visitor.reachable_vertices = []
    visitor.nonzero_res.a = res.a > visitor.eps
    visitor.g.set_edge_filter(visitor.nonzero_res)

    gt.search.bfs_search(visitor.g, s, visitor)

    visitor.g.clear_filters()

    return visitor.reachable_vertices #list of vertex indices
