from tsmodels.naive import NaiveModel
from tsmodels.autoregressive import AutoRegressive
from tsmodels.machinlearning import MachineLearningRegression
from machinelearning.linearregression import LinearRegression
from machinelearning.svr import SupportVectorRegression
import pandas as pd
import constants as const
from tsmodels.autoregessive_params import model_parameters as md


def run_base_models(horizon, train_data, test_data, train_weather, test_weather):
    horizon_info = const.HORIZON_INFO[horizon]
    # run the base models
    print('Training base learners started')
    horizon_int = horizon_info["horizon_as_int"]
    resolution = horizon_info["resolution"]
    resolution_str = horizon_info["resolution_as_str"]

    # Seasonal naive
    naive_model = NaiveModel(train_data, test_data, horizon_int, resolution,
                             horizon, resolution_str)
    naive_forecasts = naive_model.get_forecasts()

    # MLR
    mlr = MachineLearningRegression(LinearRegression(), train_data, test_data, horizon_int,
                                    resolution,
                                    horizon, resolution_str, train_weather, test_weather)
    mlr_forecasts = mlr.get_forecasts()

    # SVR
    svr = MachineLearningRegression(SupportVectorRegression(), train_data, test_data, horizon_int,
                                    resolution,
                                    horizon, resolution_str, train_weather, test_weather)
    svr_forecasts = svr.get_forecasts()

    seasonal_freq = md[resolution]['seasonal_freq']
    seasonality = md[resolution]['seasonality']
    fourier_terms = md[resolution]['fourier_terms']
    fourier = md[resolution]['fourier']
    maxiter = md[resolution]['maxiter']

    # SARIMA
    print("Training ARIMA")
    arima = AutoRegressive(train_data, test_data, horizon_int, resolution,
                           horizon, resolution_str, fourier=fourier, fourier_terms=fourier_terms,
                           seasonal_freq=seasonal_freq, seasonality=seasonality, maxiter=maxiter)
    arima.find_arima_params()
    arima_forecasts = arima.get_forecasts()

    # SARIMAX
    print("Training ARIMAX")
    arimax = AutoRegressive(train_data, test_data, horizon_int, resolution,
                            horizon, resolution_str, fourier=fourier, fourier_terms=fourier_terms,
                            seasonal_freq=seasonal_freq, seasonality=seasonality, maxiter=maxiter,
                            train_features=train_weather, test_features=test_weather)
    arimax.find_arima_params()
    arimax_forecasts = arimax.get_forecasts()

    results = pd.concat([naive_forecasts, arima_forecasts, arimax_forecasts, mlr_forecasts, svr_forecasts], axis=1)
    results.columns = ['sn', '(s)arima', '(s)arimax', 'mlr', 'svr']

    return results
