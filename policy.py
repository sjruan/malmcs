from common.temporal_idx import TemporalIdx
from datetime import datetime, timedelta
import numpy as np
from utils.assignment.bottleneck_assignment_multi_level import solve_mbap
from utils.candidate_location import get_candidate_locations_general, get_candidate_locations, get_candidate_locations_ellipse
from utils.maximal_coverage import select_top_k_positions, select_best_location_with_score, cal_influence_count, \
    cal_remaining_heat_map, select_best_location, cal_covered_users
from utils.distance import distance, point_distance
import copy


class DepotPolicy:

    def __init__(self, depot):
        self.depot = depot

    def next_locations(self, cur_ts, cur_locations, already_spent_costs):
        pass


class EADSPolicy(DepotPolicy):
    def __init__(self, k, predict_model, start_day, end_day, start_hour, end_hour,
                 time_interval, radius, cost_limit, depot):
        super().__init__(depot)
        self.nb_ts_per_day = 1440 // time_interval
        self.t_idx = TemporalIdx(start_day, end_day, time_interval)
        self.predict_model = predict_model
        self.k = k
        self.radius = radius
        self.cost_limit = cost_limit
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.time_interval = time_interval

    def next_locations(self, cur_ts, cur_locations, already_spent_costs):
        cur_day = datetime.strptime(self.t_idx.ts_to_datetime(cur_ts).strftime('%Y-%m-%d'), '%Y-%m-%d')
        end_hour_ts = self.t_idx.datetime_to_ts(cur_day + timedelta(hours=self.end_hour))
        pred_num = end_hour_ts - cur_ts - 1
        pred_heat_maps = self.predict_model.predict(cur_ts, pred_num)
        allocation_strategy = self.planning_eads(pred_heat_maps, cur_locations, already_spent_costs)
        return allocation_strategy[0]

    def planning_eads(self, predicted_heat_maps, agents, already_spent_costs):
        """
        :return: T+1 size k agent each time step
        """
        k = len(agents)
        T = predicted_heat_maps.shape[0]

        # initializing targets
        future_positions = select_each_frame_best_locations(k, self.radius, predicted_heat_maps)
        future_positions.append([self.depot] * k)

        # check mbap
        opt_c, opt_assignment = solve_mbap(agents, future_positions, already_spent_costs)
        if opt_c < self.cost_limit:
            print("use min max cost strategy")
            planning_decision = []
            for i in range(T + 1):
                time_step_decision = []
                for j in range(k):
                    time_step_decision.append(future_positions[i][opt_assignment[i, j]])
                planning_decision.append(time_step_decision)
            return planning_decision

        # heuristics
        print("use heuristic strategy")
        assignment = distance_constrained_maximal_locations(np.sum(predicted_heat_maps, axis=0), agents,
                                                            already_spent_costs, self.radius, self.cost_limit, self.depot)
        clusters = init_clusters(agents, already_spent_costs, assignment, T, self.depot)
        tweak_locations_global(clusters, self.cost_limit, predicted_heat_maps, self.radius)
        planning_decision = []
        for i in range(T + 1):
            time_step_decision = []
            for j in range(k):
                time_step_decision.append(clusters[j].seq_targets[i])
            planning_decision.append(time_step_decision)
        return planning_decision


class Cluster:
    def __init__(self, agent, already_spent_cost, seq_targets):
        self.agent = agent
        self.already_spent_cost = already_spent_cost
        self.seq_targets = seq_targets
        cost = distance(self.agent[1], self.agent[0],
                        self.seq_targets[0][1], self.seq_targets[0][0])
        for i in range(len(self.seq_targets) - 1):
            cost += distance(self.seq_targets[i][1], self.seq_targets[i][0],
                             self.seq_targets[i + 1][1], self.seq_targets[i + 1][0])
        self.cost = cost + self.already_spent_cost

    def re_cal_cost(self):
        cost = point_distance(self.agent, self.seq_targets[0])
        for i in range(len(self.seq_targets) - 1):
            cost += point_distance(self.seq_targets[i], self.seq_targets[i + 1])
        self.cost = cost + self.already_spent_cost


def init_clusters(agents, already_spent_costs, static_locations, T, depot):
    # T + 1 locations
    k = len(agents)
    clusters = []
    for i in range(k):
        seq_targets = []
        for j in range(T):
            seq_targets.append(copy.deepcopy(static_locations[i]))
        seq_targets.append(depot)
        clusters.append(Cluster(agents[i], already_spent_costs[i], seq_targets))
    return clusters


def update_cluster(cluster, t, loc):
    cluster.seq_targets[t] = copy.copy(loc)
    cluster.re_cal_cost()


def tweak_locations_global(clusters, cost_limit, predicted_heat_maps, radius):
    # agent and T+1 is fixed
    T, row, col = predicted_heat_maps.shape
    k = len(clusters)
    clusters_with_idx = list(zip(clusters, range(k)))

    # if all locations are stable, the algorithm can stop
    unstable = True
    while unstable:
        # print("current cluster cost: " + str(tweak_cluster.cost))
        best_loc = None
        best_coverage_gain = -1
        best_t = -1
        best_cluster_idx = -1
        for cluster_with_idx in clusters_with_idx:
            tweak_cluster = cluster_with_idx[0]
            tweak_idx = cluster_with_idx[1]
            avail_energy = cost_limit - tweak_cluster.cost
            for t in range(T):
                loc, coverage_gain = choose_best_location_with_coverage_gain(t, avail_energy, radius, clusters,
                                                                             tweak_idx, predicted_heat_maps)
                # print("coverage gain: " + str(coverage_gain))
                if coverage_gain > best_coverage_gain:
                    best_coverage_gain = coverage_gain
                    best_loc = loc
                    best_t = t
                    best_cluster_idx = tweak_idx
        if best_coverage_gain > 0:
            # print("best coverage gain: " + str(best_coverage_gain))
            update_cluster(clusters[best_cluster_idx], best_t, best_loc)
            unstable = True
        else:
            unstable = False
    # print("current cluster cost: " + str(tweak_cluster.cost))
    return clusters


def choose_best_location_with_coverage_gain(t, avail_energy, radius, clusters, tweak_idx, predicted_heat_maps):
    """

    :param t: tweak time stamp
    :param avail_energy: the energy currently remaining
    :param radius:
    :param clusters: the cluster to tweak
    :param tweak_idx
    :param predicted_heat_maps
    :return: (loc, coverage_gain)
    """
    cluster = clusters[tweak_idx]
    T, row, col = predicted_heat_maps.shape
    old_locations = [cluster.seq_targets[t] for cluster in clusters]
    other_locations = copy.deepcopy(old_locations)
    del other_locations[tweak_idx]
    heat_map_remain = cal_remaining_heat_map(predicted_heat_maps[t, :, :], other_locations, radius)

    # the first location
    if t == 0:
        avail_energy = avail_energy + point_distance(cluster.agent, cluster.seq_targets[0]) + \
                       point_distance(cluster.seq_targets[0], cluster.seq_targets[1])
        if cluster.agent[0] == cluster.seq_targets[1][0] and cluster.agent[1] == cluster.seq_targets[1][1]:
            if avail_energy == 0:
                return cluster.seq_targets[0], 0
            else:
                candidates = get_candidate_locations(cluster.agent, avail_energy / 2.0, row, col)
        else:
            if point_distance(cluster.agent, cluster.seq_targets[1]) == avail_energy:
                return cluster.seq_targets[0], 0
            else:
                candidates = get_candidate_locations_ellipse(cluster.agent, cluster.seq_targets[1],
                                                             avail_energy, row, col)
    # internal case
    else:
        avail_energy = avail_energy + point_distance(cluster.seq_targets[t - 1], cluster.seq_targets[t]) + \
                       point_distance(cluster.seq_targets[t], cluster.seq_targets[t + 1])
        if cluster.seq_targets[t - 1][0] == cluster.seq_targets[t + 1][0] and \
                cluster.seq_targets[t - 1][1] == cluster.seq_targets[t + 1][1]:
            if avail_energy == 0:
                return cluster.seq_targets[t], 0
            else:
                candidates = get_candidate_locations(cluster.seq_targets[t - 1], avail_energy / 2.0, row, col)
        else:
            if point_distance(cluster.seq_targets[t - 1], cluster.seq_targets[t + 1]) == avail_energy:
                return cluster.seq_targets[t], 0
            else:
                candidates = get_candidate_locations_ellipse(cluster.seq_targets[t - 1], cluster.seq_targets[t + 1],
                                                             avail_energy, row, col)
    if len(candidates) != 0:
        tweaked_location = select_best_location(candidates, heat_map_remain, radius)
    else:
        raise Exception("no candidates")
    new_locations = copy.deepcopy(old_locations)
    new_locations[tweak_idx] = tweaked_location
    gain = cal_covered_users(new_locations, predicted_heat_maps[t, :, :], radius) - \
        cal_covered_users(old_locations, predicted_heat_maps[t, :, :], radius)
    return tweaked_location, gain


def select_each_frame_best_locations(k, radius, predicted_heat_maps):
    """
    init future positions to the maximal k coverage positions in each time stamp
    :param k: agent number
    :param radius: agent influence radius
    :param predicted_heat_maps: predicted heat maps (next timeslot to the end of the day)
    :return: T x k future positions
    """
    T = predicted_heat_maps.shape[0]
    future_positions = []
    for i in range(T):
        best_positions = select_top_k_positions(k, predicted_heat_maps[i, :, :], radius)
        future_positions.append(best_positions)
    return future_positions


def distance_constrained_maximal_locations(heat_map, cur_locations, already_spent_costs, radius, D, depot):
    H, W = heat_map.shape
    k = len(cur_locations)
    # initialize all assignment to None
    planning_decision = [None] * k
    var_heat_map = heat_map.copy()

    # until k locations are selected
    for i in range(k):
        b_score = float("-inf")
        b_agent = None
        b_cost = float("inf")
        b_loc = None
        # iterate all unassigned agents, assign maximal locations to agent with cheapest cost
        unassigned_it = (j for j in range(k) if planning_decision[j] is None)
        for j in unassigned_it:
            cur_loc = cur_locations[j]
            candidates = get_candidate_locations_general(cur_loc, depot, D - already_spent_costs[j], H, W)
            if len(candidates) == 0:
                loc = cur_loc
                score = cal_influence_count(var_heat_map, loc[1], loc[0], radius)
                cost = 0
            else:
                best_location_with_score = select_best_location_with_score(candidates, var_heat_map, radius)
                loc = best_location_with_score[0]
                score = best_location_with_score[1]
                cost = point_distance(cur_loc, loc) + point_distance(loc, depot)
            cheaper = False
            if b_loc is not None:
                cheaper = (b_loc[0] == loc[0] and b_loc[1] == loc[1] and cost < b_cost)
            if score > b_score or cheaper:
                b_agent = j
                b_loc = loc
                b_score = score
                b_cost = cost
        # choose location with with maximal coverage and lowest cost assignment
        planning_decision[b_agent] = b_loc
        # update heat map
        var_heat_map = cal_remaining_heat_map(var_heat_map, [b_loc], radius)
    return planning_decision
