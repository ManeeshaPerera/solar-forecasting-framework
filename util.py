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


# calculate the MASE across samples
def test_MASE(train_sample, test_sample, forecasts, horizon, seasonality, denom=None):
    errors = []
    for h in range(0, len(test_sample), horizon):
        # let's add a condition to only calculate the error after sunrise and before sunset
        if np.count_nonzero(test_sample[h: h + horizon]):
            errors.append(
                MASE(train_sample, test_sample[h: h + horizon], forecasts[h: h + horizon], seasonality, denom))

    return errors

def process_data(data):
    data.reset_index(drop=True, inplace=True)
    x = data.drop(['solarpower'], axis=1)
    y = data['solarpower']
    return x, y
