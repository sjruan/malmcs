import numpy as np
import networkx as nx


def maximum_matching(threshold_matrix):
    """

    :param threshold_matrix: |A| x |B|
    :return: the maximal matching number, assignment
    """
    set_a_len, set_b_len = threshold_matrix.shape
    B = nx.Graph()
    B.add_nodes_from([i for i in range(set_a_len)], bipartite=0)
    B.add_nodes_from([i for i in range(set_a_len, set_a_len + set_b_len)], bipartite=1)
    B.add_edges_from([(i, j + set_a_len) for (i, j), value in np.ndenumerate(threshold_matrix) if value == 1])
    top_nodes = {n for n, d in B.nodes(data=True) if d['bipartite'] == 0}
    match_dict = nx.bipartite.hopcroft_karp_matching(B, top_nodes)
    match_num = 0
    assignment = []
    for i in range(set_a_len):
        if i in match_dict:
            assignment.append(match_dict.get(i) - set_a_len)
            match_num += 1
        else:
            assignment.append(-1)
    return match_num, assignment
