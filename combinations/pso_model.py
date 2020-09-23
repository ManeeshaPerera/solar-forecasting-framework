from combinations._base_class import ForecastCombinations
import pyswarms as ps
import pyswarms.utils.search.random_search as randomHyper
import util
import numpy as np
import scipy


class PSO(ForecastCombinations):
    # increase the iterations and hyper_parameter_iter for better optimization
    def __init__(self, train_fc, in_sample, out_sample, dimension,
                 num_particles, horizon, seasonality, options,
                 bounds=None, iterations=10, softmax=False, hyper_parameter_iter=10):
        super().__init__(train_fc, in_sample, out_sample, horizon)
        self.dimension = dimension
        self.num_particles = num_particles
        self.bounds = bounds
        self.seasonality = seasonality
        self.iterations = iterations
        self.softmax = softmax
        self.options = options
        self.hyper_parameter_iter = hyper_parameter_iter

    def get_forecast(self, data):
        return super().get_forecast(data)

    def hyper_parameter_search(self):
        # calculate MASE denominator for fast search
        self.calculate_MASE_denom(self.seasonality)

        # search hyper parameters
        g = randomHyper.RandomSearch(ps.single.GlobalBestPSO, n_particles=self.num_particles, dimensions=self.dimension,
                                     options=self.options, bounds=self.bounds, iters=self.iterations,
                                     objective_func=self._objective_func, n_selection_iters=self.hyper_parameter_iter)
        _, best_options = g.search()
        self.options = best_options

    def find_weights(self):
        # Call instance of PSO
        optimizer = ps.single.GlobalBestPSO(n_particles=self.num_particles, dimensions=self.dimension,
                                            options=self.options, bounds=self.bounds)

        # Perform optimization
        best_cost, best_pos = optimizer.optimize(self._objective_func, iters=self.iterations, n_processes=4)

        self.weights = best_pos

        if self.softmax:
            self.weights = scipy.special.softmax(self.weights)

        return self.weights

    def _objective_func(self, x):
        # this function should return values for each particle
        cost_values = []
        # calculating the metric for each particle
        for particle in x:
            fc = self.train_fc.copy()

            # multiply the predictions of each method with it's weights
            for i in range(0, len(particle)):
                fc.loc[:, fc.columns[i]] *= particle[i]

            # Now calculate the new weighted forecasts
            fc_combine = fc.sum(axis=1)
            error = np.mean(
                util.test_MASE(self.in_sample['solarpower'].to_numpy(), self.out_sample['solarpower'].to_numpy(),
                               fc_combine.to_numpy(), self.horizon,
                               self.seasonality, self.MASE_denominator))
            cost_values.append(error)
        # return the cost function
        return np.array(cost_values)
