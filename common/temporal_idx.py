from datetime import timedelta


class TemporalIdx:
    """
    end time is exclusive
    """
    def __init__(self, start_time, end_time, time_interval_in_minutes):
        self.start_time = start_time
        self.end_time = end_time
        self.time_interval = time_interval_in_minutes
        self.ts_num = int((end_time - start_time).total_seconds() // (time_interval_in_minutes * 60))

    def ts_to_datetime(self, ts):
        return self.start_time + timedelta(minutes=self.time_interval * ts)

    def datetime_to_ts(self, cur_time):
        ts = int((cur_time - self.start_time).total_seconds() // (self.time_interval * 60))
        if ts < 0 or ts >= self.ts_num:
            raise Exception("cur_time is not in time range")
        return ts
