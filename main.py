# Example code to run the base learners and forecast combinations.
# Note you define any forecasting model as the base learner for this framework.
import pandas as pd
import os
import constants
import util
from datetime import timedelta
import numpy as np
from timeseries_split import TimeSeriesDivide
import run_base_models as base_learners
import run_combinations as combinations
import plot

if __name__ == '__main__':
    # 1 day horizon
    horizon = '1D'
    horizon_info = constants.HORIZON_INFO[horizon]
    pre_process = True  # pre process the data to remove any unrealistic values

    # Let's first read the original data
    data = pd.read_csv(os.path.join(constants.DATA, 'house1.csv'), index_col=['timestamp'])
    # covert index to datetime
    data.index = pd.to_datetime(data.index)
    print(data)

    if pre_process:
        # pre processing the data to remove unrealistic values
        house_data = util.remove_unrealistic_values(data)

    # split time series to train and test/ out-samples
    ts_divide = TimeSeriesDivide(data, test_days=2,
                                 resolution=timedelta(hours=1))
    train, test = ts_divide.split_train_test_by_days()

    # split the time series to train-in-sample and train-out-samples to train ensembles
    ts_divide_ensemble = TimeSeriesDivide(data, test_days=4,
                                          resolution=timedelta(hours=1))
    train_base_learner, train_ensemble = ts_divide_ensemble.split_train_test_by_days()

    print(train, test, train_base_learner, train_ensemble)

    # read external features
    external_features = pd.read_csv(
        os.path.join(constants.DATA, 'external.csv'), index_col=['timestamp'])
    # covert index to datetime
    external_features.index = pd.to_datetime(data.index)
    external_train = external_features[train.index[0]: train.index[-1]]
    external_test = external_features[test.index[0]: test.index[-1]]

    external_train_base_learner = external_features[train_base_learner.index[0]: train_base_learner.index[-1]]
    external_train_ensemble = external_features[train_ensemble.index[0]: train_ensemble.index[-1]]

    print(external_train, external_test, external_train_base_learner, external_train_ensemble)

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
    print(combinations_training)
    print(combinations_testing)
    print(weights)

    print("Final forecasts\n")
    fc = pd.concat([test_fc, combinations_testing], axis=1)
    print(fc)

    print("mean MASE across samples\n")
    dic_mase = {}
    for method in fc:
        mase = util.test_MASE(train['solarpower'].to_numpy(), test['solarpower'].to_numpy(),
                              fc[method].to_numpy(), horizon_info['horizon_as_int'],
                              24)
        dic_mase[method] = np.mean(mase)
    print(dic_mase)

    plot.plot_fc(fc, test).show()
