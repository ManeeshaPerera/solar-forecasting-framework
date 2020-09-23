import pandas as pd


class SeasonalNaive:
    def __init__(self):
        self.data = None
        self.resolution = None
        self.period = None
        self.num_periods = None
        self.column = None
        self.options = None

    def _set_default_options(self):
        if self.options is None:
            self.options = {}

        self.resolution = pd.to_timedelta(self.options['resolution'])
        self.column = self.data.columns[0]
        if 'period' not in self.options:
            self.options['period'] = '1 day'
        if 'num_periods' not in self.options:
            self.options['num_periods'] = 1

    def fit(self, data, options=None):
        # super().fit(data, options)
        self.data = data
        self.options = options

        # Ensure defaults are set for any remaining unspecified options
        self._set_default_options()
        self.period = pd.to_timedelta(self.options['period'])
        self.num_periods = self.options['num_periods']

    def predict(self):
        fc_start = self.data.index[-1] + self.resolution
        fc_end = self.data.index[-1] + self.period
        timestamps = pd.Series(pd.date_range(fc_start, fc_end, freq=self.resolution))

        values = [0] * len(timestamps)

        for p in range(0, self.num_periods):
            # Find start and end of this particular period
            period_start = self.data.index[-1] - pd.Timedelta(self.period * (p + 1)) + self.resolution
            period_end = period_start + (len(timestamps) - 1) * self.resolution

            # Multiple all values in this period by the relevant weight for this period
            period_values = [x for x in self.data[period_start:period_end][self.column]]

            # Maintain running total
            values = [x + y for x, y in zip(values, period_values)]

        # Create and return pandas data frame
        return pd.DataFrame({'timestamp': timestamps, 'solarpower': values}).set_index('timestamp')
