# Example code to run the base learners and forecast combinations.
# Note you can define any forecasting model as the base learners for this framework.
import pandas as pd
import os
import constants
from datetime import timedelta
from timeseries_split import TimeSeriesDivide
import run_base_models as base_learners
import run_combinations as combinations


def run_forecast_combinations_approach(data, horizon):
    # split time series to train and test/ out-samples
    ts_divide = TimeSeriesDivide(data, test_days=2,
                                 resolution=timedelta(hours=1))
    train, test = ts_divide.split_train_test_by_days()

    # split the in-sample data as training subset 1 and training subset 2
    ts_divide_ensemble = TimeSeriesDivide(train, test_days=4,
                                          resolution=timedelta(hours=1))
    train_base_learner, train_ensemble = ts_divide_ensemble.split_train_test_by_days()

    # read external features
    external_features = pd.read_csv(
        os.path.join(constants.DATA, 'external.csv'), index_col=['timestamp'])
    # covert index to datetime
    external_features.index = pd.to_datetime(external_features.index)
    external_train = external_features[train.index[0]: train.index[-1]]
    external_test = external_features[test.index[0]: test.index[-1]]

    external_train_base_learner = external_features[train_base_learner.index[0]: train_base_learner.index[-1]]
    external_train_ensemble = external_features[train_ensemble.index[0]: train_ensemble.index[-1]]

    # Training the base models
    test_fc = base_learners.run_base_models(horizon, train, test, external_train, external_test)

    # Training the base learns for ensemble
    train_ensemble_fc = base_learners.run_base_models(horizon, train_base_learner, train_ensemble,
                                                      external_train_base_learner, external_train_ensemble)

    print("Test forecasts: ", test_fc)
    print("Train ensemble forecasts: ", train_ensemble_fc)

    # Now let's Run forecast combinations

    combinations_training, combinations_testing, weights = combinations.run_combinations(horizon,
                                                                                         train_ensemble_fc, test_fc,
                                                                                         train_base_learner,
                                                                                         train_ensemble)
    print("Weights from each combination method\n")
    print(weights)

    fc = pd.concat([test_fc, combinations_testing], axis=1)

    return fc
