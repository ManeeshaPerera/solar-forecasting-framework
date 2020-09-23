from combinations._base_class import ForecastCombinations


class EqualWeight(ForecastCombinations):
    def __init__(self, train_fc, in_sample=None, out_sample=None, horizon=None):
        super().__init__(train_fc, in_sample, out_sample, horizon)

    def find_weights(self):
        self.weights = [1 / len(self.train_fc.columns)] * len(self.train_fc.columns)

    def get_forecast(self, data):
        return super().get_forecast(data)
