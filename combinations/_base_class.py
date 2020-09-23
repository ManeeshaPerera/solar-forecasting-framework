class ForecastCombinations:
    def __init__(self, train_fc, in_sample, out_sample, horizon):
        self.weights = None
        self.train_fc = train_fc
        self.in_sample = in_sample
        self.out_sample = out_sample
        self.horizon = horizon
        self.MASE_denominator = None

    def get_forecast(self, data):
        fc = data.copy()
        for w in range(0, len(self.weights)):
            fc.loc[:, fc.columns[w]] *= self.weights[w]

        # Now calculate the forecast combination
        return fc.sum(axis=1)

    def calculate_MASE_denom(self, seasonality):
        # calculate MASE denominator for fast search
        training_series = self.in_sample['solarpower'].to_numpy()

        m = seasonality
        ne = 0
        n = training_series.shape[0]
        for i in range(m, len(training_series)):
            ne = ne + abs(training_series[i] - training_series[i - m])
        denominator = ne / (n - m)

        self.MASE_denominator = denominator
