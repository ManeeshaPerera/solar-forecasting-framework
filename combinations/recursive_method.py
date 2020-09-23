from combinations._base_class import ForecastCombinations
import numpy as np
import util


class RecursiveEnsemble(ForecastCombinations):
    def __init__(self, train_fc, in_sample, out_sample, horizon, initial_matrix, seasonality, threshold,
                 max_iter=10000):
        super().__init__(train_fc, in_sample, out_sample, horizon)
        self.matrix = initial_matrix
        self.seasonality = seasonality
        self.threshold = threshold
        self.max_iter = max_iter

    def get_forecast(self, data):
        return super().get_forecast(data)

    def find_weights(self):
        data = self.train_fc.copy()
        # calculate MASE denominator for fast search
        self.calculate_MASE_denom(self.seasonality)

        iter_count = 0
        while True:
            iter_count = iter_count + 1
            # calculate the error for all methods
            fc_error = self._get_error(data)

            max_error_model = max(fc_error.keys(), key=(lambda k: fc_error[k]))
            min_error_model = min(fc_error.keys(), key=(lambda k: fc_error[k]))

            # calculating the error difference between best and worst model
            error_diff = fc_error[max_error_model] - fc_error[min_error_model]
            if error_diff > self.threshold and iter_count <= self.max_iter:
                # replace the worse model with the average
                data[max_error_model] = data.mean(axis=1)

                # edit the ensemble matrix with the weights
                method_index = data.columns.tolist().index(max_error_model)
                for i in range(0, len(self.matrix[:, method_index])):
                    self.matrix[i][method_index] = self.matrix[i].sum() / len(self.matrix[i])

            else:
                break
        # return best weights
        self.weights = self.matrix[:, data.columns.tolist().index(min_error_model)].tolist()
        return self.weights

    def _get_error(self, data):
        fc_error = {}
        for method in data.columns:
            fc_error[method] = np.mean(
                util.test_MASE(self.in_sample['solarpower'].to_numpy(), self.out_sample['solarpower'].to_numpy(),
                               data[method].to_numpy(), self.horizon,
                               self.seasonality, self.MASE_denominator))
        return fc_error
