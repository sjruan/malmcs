import numpy as np
from common.temporal_idx import TemporalIdx
import pickle
from datetime import datetime, timedelta


class PredictionModel:
    def __init__(self, data_path, start_day, end_day, nb_ts_per_day, eval_start_day, start_hour):
        self.eval_start_day = eval_start_day
        self.start_hour = start_hour
        self.t_idx = TemporalIdx(start_day, end_day, 1440 // nb_ts_per_day)
        with open(data_path, 'rb') as f:
            self.predictions = pickle.load(f)

    def predict(self, t, num):
        cur_time = self.t_idx.ts_to_datetime(t)
        cur_day = datetime(cur_time.year, cur_time.month, cur_time.day)
        predict_start_ts = self.t_idx.datetime_to_ts(cur_day + timedelta(hours=self.start_hour)) - 1
        tod_idx = t - predict_start_ts
        day_idx = (cur_day - self.eval_start_day).days
        # H X W X T to T X H X W
        return np.transpose(self.predictions[tod_idx][day_idx], (2, 0, 1))[:num]
