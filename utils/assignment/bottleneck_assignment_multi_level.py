import numpy as np
from utils.assignment.bottleneck_assignment import solve_bap
from utils.distance import distance


def solve_mbap(agents, seq_targets, already_spent_costs):
    """
    Solve the multi-level bottleneck assignment problem.
    A multi-level bottleneck assignment approach to the bus drivers' rostering problem
    :param agents: k agents, each agent is in a position
    :param seq_targets: two-dim array, T x k
    :param already_spent_costs: already spent costs of k agents
    :return: minmaxcost, a matrix, T x k, index is agent index, each row is target location index
    """
    k = len(agents)
    T = len(seq_targets)
    opt_assignment = -1 * np.ones((T, k), dtype=int)
    opt_cost = 0
    # phase 1, init assign
    # print('phase 1...')
    pre_costs = already_spent_costs.copy()
    pre_pos = agents.copy()
    for i in range(T):
        c_i, assign_i = solve_bap(pre_pos, seq_targets[i], pre_costs)
        opt_cost = c_i
        for j in range(k):
            cur_pos = seq_targets[i][assign_i[j]]
            # update opt_assignment
            opt_assignment[i, j] = assign_i[j]
            # update pre_costs (the cost from seq_targets[i-1] to seq_targets[i])
            pre_costs[j] += distance(pre_pos[j][1], pre_pos[j][0], cur_pos[1], cur_pos[0])
            # update agent pre_pos
            pre_pos[j] = cur_pos
    # print('phase 1 opt cost: ' + str(opt_cost))
    # phase 2, iterate assign until coverage
    # print('phase 2...')
    unit_costs = cal_unit_costs(opt_assignment, agents, seq_targets)
    assignment_unstable = True
    while assignment_unstable:
        assignment_unstable = False
        for i in range(T):
            other_costs = cal_other_costs(i, unit_costs) + already_spent_costs
            if i == 0:
                pre_pos = agents.copy()
            else:
                for j in range(k):
                    pre_pos[j] = seq_targets[i - 1][opt_assignment[i - 1, j]]
            later_pos = None
            if i != (T - 1):
                later_pos = []
                for j in range(k):
                    later_pos.append(seq_targets[i + 1][opt_assignment[i + 1, j]])
            c_i, assign_i = solve_bap(pre_pos, seq_targets[i], other_costs, later_pos)
            if c_i < opt_cost:
                assignment_unstable = True
                opt_cost = c_i
                for j in range(k):
                    opt_assignment[i, j] = assign_i[j]
                unit_costs = update_unit_costs(unit_costs, agents, seq_targets, opt_assignment, i)
            # print('phase 2 opt cost:' + str(opt_cost) + ', current cost:' + str(c_i))
    return opt_cost, opt_assignment


def cal_unit_costs(assignment, agents, seq_targets):
    k = len(agents)
    T = len(seq_targets)
    unit_costs = np.zeros((k, T), dtype=float)
    for j in range(k):
        for i in range(T):
            if i == 0:
                pre_loc = agents[j]
            else:
                pre_loc = seq_targets[i - 1][assignment[i - 1, j]]
            cur_loc = seq_targets[i][assignment[i, j]]
            unit_costs[j, i] = distance(pre_loc[1], pre_loc[0], cur_loc[1], cur_loc[0])
    return unit_costs


def cal_other_costs(unfixed_idx, unit_costs):
    """
    Calculate the cost before unfix idx and after unfix idx
    :param unfixed_idx: the unfixed assignment idx
    :param unit_costs: the each step cost of each agent (k x T)
    :return: other cost of agents except the unfixed idx
    """
    # cost before seq_targets[i-1] + cost after seq_targets[i+1]
    other_costs = np.sum(unit_costs[:, 0:unfixed_idx], axis=1) + np.sum(unit_costs[:, (unfixed_idx + 2):], axis=1)
    return other_costs


def update_unit_costs(unit_costs, agents, seq_targets, assignment, changed_idx):
    k = len(agents)
    T = len(seq_targets)
    for j in range(k):
        if changed_idx == 0:
            pre_loc = agents[j]
        else:
            pre_loc = seq_targets[changed_idx - 1][assignment[changed_idx - 1, j]]
        cur_loc = seq_targets[changed_idx][assignment[changed_idx, j]]
        unit_costs[j, changed_idx] = distance(pre_loc[1], pre_loc[0], cur_loc[1], cur_loc[0])
        # if the changed arc is not the last arc, recompute the next unit costs
        if changed_idx != (T - 1):
            pre_loc = cur_loc
            cur_loc = seq_targets[changed_idx + 1][assignment[changed_idx + 1, j]]
            unit_costs[j, changed_idx + 1] = distance(pre_loc[1], pre_loc[0], cur_loc[1], cur_loc[0])
    return unit_costs
