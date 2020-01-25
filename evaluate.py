from datetime import datetime, timedelta
from policy import EADSPolicy
from model import PredictionModel
import numpy as np
from utils.data_process import generate_masked_heat_map
from common.temporal_idx import TemporalIdx
from utils.distance import point_distance
from utils.maximal_coverage import cal_covered_users


class Evaluator:
    def __init__(self, data_path, start_day, end_day, eval_start_day, time_interval,
                 start_hour, end_hour, radius, depot):
        self.data = generate_masked_heat_map(np.load(data_path))
        self.t_idx = TemporalIdx(start_day, end_day, time_interval)
        self.end_day = end_day
        self.eval_start_day = eval_start_day
        self.time_interval = time_interval
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.radius = radius
        self.depot = depot

    def evaluate(self, policy, k):
        acc_coverage = 0
        day = self.eval_start_day
        start_hour_offset = int(self.start_hour * (60 / self.time_interval))
        eval_ts_num = int((self.end_hour - self.start_hour) * (60 / self.time_interval))
        end_day = self.end_day
        while day < end_day:
            print(day)
            already_spent_costs = [0] * k
            day_ts = self.t_idx.datetime_to_ts(day)
            day_coverage = 0
            # before service time
            positions = [self.depot] * k
            whole_day_routine = [[init_loc] for init_loc in positions]
            # decide first location
            next_positions = policy.next_locations(day_ts + start_hour_offset - 1, positions, already_spent_costs)
            for j in range(k):
                dis = point_distance(positions[j], next_positions[j])
                already_spent_costs[j] += dis
                whole_day_routine[j].append(next_positions[j])
            positions = next_positions
            for i in range(eval_ts_num):
                cur_true_heat_map = self.data[day_ts + start_hour_offset + i, :, :]
                day_coverage += cal_covered_users(positions, cur_true_heat_map, self.radius)
                if i < eval_ts_num - 1:
                    next_positions = policy.next_locations(day_ts + start_hour_offset + i, positions,
                                                           already_spent_costs)
                    for j in range(k):
                        dis = point_distance(positions[j], next_positions[j])
                        already_spent_costs[j] += dis
                        whole_day_routine[j].append(next_positions[j])
                    positions = next_positions
            print("energy consumption: ")
            for j in range(k):
                # add return to depot cost
                already_spent_costs[j] += point_distance(positions[j], self.depot)
                print("agent {0}: {1}".format(j, already_spent_costs[j]))
            print('whole day routine:')
            for j in range(k):
                whole_day_routine[j].append(self.depot)
                print(whole_day_routine[j])
            acc_coverage += day_coverage
            day += timedelta(days=1)
        return acc_coverage


if __name__ == '__main__':
    depot = (32, 66)
    k = 20
    cost_limit = 50
    start_day = datetime(2018, 1, 1)
    end_day = datetime(2018, 11, 1)
    eval_start_day = datetime(2018, 10, 1)
    # Note: the radius is 0-indexed, i.e., radius=2 <-> r=2 in the paper
    radius = 1
    time_interval = 60
    start_hour = 10
    end_hour = 22
    nb_ts_per_day = 1440 // time_interval
    start_time_str = start_day.strftime('%Y%m%d')
    end_time_str = end_day.strftime('%Y%m%d')
    data_path = './data/MALMCS_data/frames_{}_{}_{}.npy'.format(start_time_str, end_time_str, nb_ts_per_day)
    model_path = './data/MALMCS_data/pred_all_stresnet_mf4_masked.pkl'
    prediction_model = PredictionModel(model_path, start_day, end_day, nb_ts_per_day, eval_start_day, start_hour)
    evaluator = Evaluator(data_path, start_day, end_day, eval_start_day, time_interval,
                          start_hour, end_hour, radius, depot)
    policy = EADSPolicy(k, prediction_model, start_day, end_day, start_hour, end_hour,
                        time_interval, radius, cost_limit, depot)
    tot_coverage = evaluator.evaluate(policy, k)
    print(tot_coverage / 31)
