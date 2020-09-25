# Example code of the forecast framework

from forecast_combinations_approach import run_forecast_combinations_approach as fc_approach
from global_forecast_approach import run_global_forecast_approach as global_model
from timeseries_split import TimeSeriesDivide
import plot
import numpy as np
import pandas as pd
import util
import os
import constants
from datetime import timedelta

if __name__ == '__main__':
    # forecast horizon
    horizon = '1D'
    horizon_info = constants.HORIZON_INFO[horizon]

    # Let's first read the original data
    data = pd.read_csv(os.path.join(constants.DATA, 'house1.csv'), index_col=['timestamp'])
    # covert index to datetime
    data.index = pd.to_datetime(data.index)
    print(data)
    # whether pre processing is required to remove any unrealistic values
    pre_process = True

    if pre_process:
        # pre processing the data to remove unrealistic values
        data = util.remove_unrealistic_values(data)

    # split time series to train and test/ out-samples
    ts_divide = TimeSeriesDivide(data, test_days=2,
                                 resolution=timedelta(hours=1))
    train, test = ts_divide.split_train_test_by_days()

    forecasts_combinations = fc_approach(data, horizon)
    # global forecast model will read data separately from multiple households and provide forecasts for the house1
    forecast_global = global_model(horizon)[0]

    print("Final forecasts\n")
    fc = pd.concat([forecasts_combinations, forecast_global], axis=1)
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
