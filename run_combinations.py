from combinations.equal_weight import EqualWeight
from combinations.pso_model import PSO
from combinations.recursive_method import RecursiveEnsemble
import constants as const
import pandas as pd
import numpy as np


def run_combinations(horizon, forecast, forecast_test, data_train, data_out_sample):
    weights = {'weight': [], 'method': [], 'comb_method': []}
    horizon_info = const.HORIZON_INFO[horizon]
    seasonality = horizon_info['arima_params'][
        'seasonal_freq']
    methods = forecast.columns.tolist()

    pso_initial_options = {'c1': [0, 10],
                           'c2': [0, 10],
                           'w': [0, 10],
                           'k': [1, 20],
                           'p': 2}
    num_pso_particles = 100

    # Run equal weight
    equal_weight = EqualWeight(forecast)
    equal_weight.find_weights()

    add_weights(weights, equal_weight.weights, methods, 'average')

    eq_fc = equal_weight.get_forecast(forecast)
    eq_fc_test = equal_weight.get_forecast(forecast_test)

    # Run PSO
    dimension = len(forecast.columns)
    pso = PSO(forecast, data_train, data_out_sample, dimension, num_pso_particles,
              horizon_info['horizon_as_int'],
              seasonality, options=pso_initial_options)
    pso.hyper_parameter_search()
    pso.find_weights()
    add_weights(weights, pso.weights, methods, 'pso')
    pso_fc = pso.get_forecast(forecast)
    pso_fc_test = pso.get_forecast(forecast_test)

    # PSO with bounds
    pso_b = PSO(forecast, data_train, data_out_sample, dimension, num_pso_particles,
                horizon_info['horizon_as_int'],
                seasonality, options=pso_initial_options, bounds=(np.array([0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1])))
    pso_b.hyper_parameter_search()
    pso_b.find_weights()
    add_weights(weights, pso_b.weights, methods, 'pso [0,1]')
    pso_b_fc = pso_b.get_forecast(forecast)
    pso_b_fc_test = pso_b.get_forecast(forecast_test)

    # Add to Unity
    pso_b.weights = pso_b.weights / pso_b.weights.sum()
    add_weights(weights, pso_b.weights, methods, 'pso- convex')
    pso_b_fc_scaled = pso_b.get_forecast(forecast)
    pso_b_fc_test_scaled = pso_b.get_forecast(forecast_test)

    # Run recursive ensemble
    print("start recursive ensemble")
    matrix = np.identity(len(forecast.columns))
    re = RecursiveEnsemble(forecast, data_train, data_out_sample, horizon_info['horizon_as_int'], matrix, seasonality,
                           0.001)
    re.find_weights()
    add_weights(weights, re.weights, methods, 're')
    re_fc = re.get_forecast(forecast)
    re_fc_test = re.get_forecast(forecast_test)

    train = pd.concat([pso_fc, pso_b_fc, pso_b_fc_scaled, eq_fc, re_fc], axis=1)
    train.columns = ['pso', 'pso [0,1]', 'pso- convex', 'average', 're']

    test = pd.concat([pso_fc_test, pso_b_fc_test, pso_b_fc_test_scaled, eq_fc_test, re_fc_test], axis=1)
    test.columns = ['pso', 'pso [0,1]', 'pso- convex', 'average', 're']

    return train, test, pd.DataFrame(weights)


def add_weights(dic, weights, methods, comb_name):
    for w in range(0, len(weights)):
        dic['weight'].append(weights[w])
        dic['method'].append(methods[w])
        dic['comb_method'].append(comb_name)
