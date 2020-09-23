from solartime import SolarTime
import pandas as pd
from datetime import timezone
from pytz import timezone
import constants
import os
import numpy as np
import sklearn.metrics as skmetrics
import math
from sklearn.preprocessing import StandardScaler


def read_data(resolution, path, pre_process=False):
    data_dic = {}
    for location in constants.PATH_DIC:
        location_info = constants.PATH_DIC[location]
        for house_id in location_info['id']:
            filename = str(house_id) + '_' + resolution
            house_data = pd.read_pickle(os.path.join('data/solar', location, 'pickle', filename))
            house_data = house_data.rename(columns={'solar': 'solarpower'})

            if pre_process:
                # pre processing the data to remove unrealistic values
                house_data = remove_non_solar_values(house_data, lat=location_info['lat'],
                                                     lon=location_info['lon'],
                                                     time_zone=location_info['timezone'])
                house_data = remove_unrealistic_values(house_data)

            # let's save the data in results
            house_data.to_pickle(os.path.join(path, str(house_id), 'min'))
            data_dic[str(house_id)] = {
                'location': location,
                'timeseries': house_data
            }
    return data_dic


def remove_unrealistic_values(data_frame):
    new_values = [power if power > 0 else 0 for power in data_frame['solarpower'].values]
    data_frame['solarpower'] = new_values
    return data_frame


# calculate the MASE for a given horizon
def MASE(training_series, testing_series, prediction_series, m, denom=None):
    h = testing_series.shape[0]
    numerator = np.sum(np.abs(testing_series - prediction_series))

    n = training_series.shape[0]
    if denom is None:
        ne = 0
        for i in range(m, len(training_series)):
            ne = ne + abs(training_series[i] - training_series[i - m])
        denominator = ne / (n - m)
    else:
        denominator = denom
    return (numerator / denominator) / h


# calculate the MASE across test samples
def test_MASE(train_sample, test_sample, forecasts, horizon, seasonality, denom=None):
    errors = []
    for h in range(0, len(test_sample), horizon):
        # let's add a condition to only calculate the error after sunrise and before sunset
        if np.count_nonzero(test_sample[h: h + horizon]):
            errors.append(
                MASE(train_sample, test_sample[h: h + horizon], forecasts[h: h + horizon], seasonality, denom))

    return errors
