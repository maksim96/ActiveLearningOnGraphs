import graph_tool as gt
from graph_tool.search import BFSVisitor
import numpy as np

class VisitorExample(BFSVisitor):



    def __init__(self, graph):
        self.g = graph
        self.reachable_vertices = []
        self.first_time = True  # we don't want to add the root


    def discover_vertex(self, u):
        if self.first_time:
            self.first_time = False
        else:
            self.reachable_vertices.append(self.g.vertex_index[u])

def own_min_cut(graph, s, cap, res):
    nonzero_res = graph.new_edge_property("bool")
    eps = 0#np.percentile(res.a[np.nonzero(res.a)], 90)
    nonzero_res.a = res.a > eps
    graph.set_edge_filter(nonzero_res)

    visitor = VisitorExample(graph)

    gt.search.bfs_search(graph, s, visitor)

    graph.clear_filters()

    return visitor.reachable_vertices #list of vertex indices
