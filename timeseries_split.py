import pandas as pd
from datetime import timedelta


class TimeSeriesDivide:
    def __init__(self, time_series, resolution, split_percentage=None, train_days=None, test_days=None,
                 resample_size=None,
                 resample_col=None):
        self.time_series = time_series
        self.split_percentage = split_percentage
        self.size = len(time_series)
        self.resolution = resolution
        self.resample_size = resample_size
        self.resample_col = resample_col
        self.train_days = train_days
        self.test_days = test_days

    def split_train_test_by_percentage(self):
        split_index = int(self.size * self.split_percentage)
        split_date = pd.Timestamp(self.time_series.index[-split_index].date())
        train = self.time_series[:split_date - self.resolution]
        test = self.time_series[split_date:]
        return train, test

    def split_train_test_by_days(self):
        # we are splitting the time series from the end
        end_date = self.time_series.index[-1]

        test_index = end_date - timedelta(days=self.test_days)
        if self.train_days:
            train_index = test_index - timedelta(days=self.train_days) + self.resolution
        else:
            train_index = self.time_series.index[0]

        return self.time_series[train_index: test_index], self.time_series[
                                                          test_index + self.resolution:]

    def resample_time_series(self):
        return self.time_series[self.resample_col].resample(self.resample_size).mean().to_frame()
