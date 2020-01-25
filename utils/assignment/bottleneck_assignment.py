import numpy as np
from utils.assignment.hopcroft_karp_matching import maximum_matching
from utils.distance import distance


def solve_bap(agents, targets, other_costs, later_pos=None):
    """
    Solve the bottleneck assignment problem.
    Threshold algorithm
    :param agents: k agents, each agent is in a position
    :param targets: k target locations
    :param other_costs: other costs attached with agents
    :param later_pos: later positions after targets (Optional, index must be aligned with agents)
    :return: minmaxcost, a vector, index is agent index, value is target location index
    """
    k = len(agents)
    if later_pos is None:
        cost_matrix = cal_cost_matrix(agents, targets, other_costs)
    else:
        cost_matrix = cal_cost_matrix_with_later_pos(agents, targets, other_costs, later_pos)
    c_values = np.sort(cost_matrix.flatten())
    start_idx = 0
    end_idx = len(c_values) - 1
    opt_c = float('inf')
    opt_assignment = None
    while start_idx <= end_idx:
        mid_idx = (start_idx + end_idx) // 2
        cur_c = c_values[mid_idx]
        threshold_matrix = cal_threshold_matrix(cost_matrix, cur_c)
        max_matching_num, assignment = maximum_matching(threshold_matrix)
        # print("c idx: " + str(mid_idx))
        # print("matching num: " + str(max_matching_num))
        if max_matching_num == k:
            # perfect matching
            end_idx = mid_idx - 1
            opt_c = cur_c
            # print("current c: " + str(cur_c))
            opt_assignment = assignment
        else:
            start_idx = mid_idx + 1
    # print("opt c: " + str(opt_c) + "\n")
    return opt_c, opt_assignment


def cal_threshold_matrix(distance_matrix, c):
    shape = distance_matrix.shape
    threshold_matrix = np.zeros(shape, dtype=int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if distance_matrix[i, j] <= c:
                threshold_matrix[i, j] = 1
    return threshold_matrix


def cal_cost_matrix(agents, targets, other_costs):
    k = len(agents)
    distance_matrix = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            distance_matrix[i, j] = distance(agents[i][1], agents[i][0], targets[j][1], targets[j][0]) + \
                                    other_costs[i]
    return distance_matrix


def cal_cost_matrix_with_later_pos(agents, targets, other_costs, later_pos):
    k = len(agents)
    distance_matrix = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            distance_matrix[i, j] = distance(agents[i][1], agents[i][0], targets[j][1], targets[j][0]) + \
                                    distance(targets[j][1], targets[j][0], later_pos[i][1], later_pos[i][0]) + \
                                    other_costs[i]
    return distance_matrix
