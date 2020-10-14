# Example code to run the global forecasting approach
import pandas as pd
import os
import constants
import util
from datetime import timedelta
from timeseries_split import TimeSeriesDivide
import run_global_models


def get_data(filename):
    # whether pre processing is required to remove any unrealistic values
    pre_process = True

    # Let's first read the original data
    data = pd.read_csv(os.path.join(constants.DATA, filename), index_col=['timestamp'])
    # covert index to datetime
    data.index = pd.to_datetime(data.index)

    if pre_process:
        # pre processing the data to remove unrealistic values
        data = util.remove_unrealistic_values(data)

    # split time series to train and test/ out-samples
    ts_divide = TimeSeriesDivide(data, test_days=2,
                                 resolution=timedelta(hours=1))
    train, test = ts_divide.split_train_test_by_days()

    # split the time series for training and validation
    ts_divide_ensemble = TimeSeriesDivide(train, test_days=4,
                                          resolution=timedelta(hours=1))
    train_in_sample, train_out_sample = ts_divide_ensemble.split_train_test_by_days()

    # read external features
    external_features = pd.read_csv(
        os.path.join(constants.DATA, 'external.csv'), index_col=['timestamp'])
    # covert index to datetime
    external_features.index = pd.to_datetime(external_features.index)

    return data, train_in_sample, train_out_sample, test, external_features[data.index[0]: data.index[-1]]


def run_global_forecast_approach(horizon):
    data_info = {}
    house_id = 1

    for file in ['house1.csv', 'house2.csv']:
        data_info[house_id] = {}
        data, train_in_sample_data, train_out_sample_data, test_data, external = get_data(file)
        data_info[house_id]['data'] = data
        data_info[house_id]['train'] = train_in_sample_data
        data_info[house_id]['val'] = train_out_sample_data
        data_info[house_id]['test'] = test_data
        data_info[house_id]['external'] = external
        house_id = house_id + 1

    forecasts = run_global_models.run_global_model(data_info, horizon)
    return forecasts
