from tsmodels.naive import NaiveModel
from tsmodels.autoregressive import AutoRegressive
from tsmodels.machinlearning import MachineLearningRegression
from machinelearning.linearregression import LinearRegression
from machinelearning.svr import SupportVectorRegression
import pandas as pd
import constants as const


def run_base_models(horizon, train_data, test_data, train_weather, test_weather):
    horizon_info = const.HORIZON_INFO[horizon]

    # run the base models
    print('Training base learners started')
    horizon_int = horizon_info["horizon_as_int"]
    resolution = horizon_info["resolution"]
    resolution_str = horizon_info["resolution_as_str"]

    # Seasonal naive
    print("Training SN")
    naive_model = NaiveModel(train_data, test_data, horizon_int, resolution,
                             horizon, resolution_str)
    naive_forecasts = naive_model.get_forecasts()

    # MLR
    print("Training MLR")
    mlr = MachineLearningRegression(LinearRegression(), train_data, test_data, horizon_int,
                                    resolution,
                                    horizon, resolution_str, train_weather, test_weather)
    mlr_forecasts = mlr.get_forecasts()

    # SVR
    print("Training SVR")
    svr = MachineLearningRegression(SupportVectorRegression(), train_data, test_data, horizon_int,
                                    resolution,
                                    horizon, resolution_str, train_weather, test_weather)
    svr_forecasts = svr.get_forecasts()

    seasonal_freq = horizon_info['arima_params']['seasonal_freq']
    seasonality = horizon_info['arima_params']['seasonality']
    fourier_terms = horizon_info['arima_params']['fourier_terms']
    fourier = horizon_info['arima_params']['fourier']
    maxiter = horizon_info['arima_params']['maxiter']

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
