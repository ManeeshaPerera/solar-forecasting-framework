import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class MachineLearning:
    def __init__(self):
        # Initialise instance attributes
        self.model = None
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.options = None
        self.resolution = None
        self.column = None
        self.scaler_X = None
        self.scaler_y = None

    def _check_required_options(self):
        self.resolution = pd.to_timedelta(self.options['resolution'])

    def _set_default_options(self):
        """
        Ensure that a default value is set for any options that were not supplied
        :return: None
        """
        if self.options is None:
            self.options = {}
        if 'column' not in self.options:
            self.options['column'] = None
        if 'include_temporal' not in self.options:
            self.options['include_temporal'] = True
        if 'scale' not in self.options:
            self.options['scale'] = True
        if 'remove_negative' not in self.options:
            self.options['remove_negative'] = True

    def fit(self, X, y, options=None):
        self.X = X
        self.y = y
        self.options = options
        self._check_required_options()

        self._set_default_options()
        self.column = y.columns[0]

        # Ensure X and y have same length by removing any values at start or end that don't exist in both
        common_start = max(self.X.index[0], self.y.index[0])
        common_end = min(self.X.index[-1], self.y.index[-1])
        self.X = self.X[common_start:common_end]
        self.y = self.y[common_start:common_end]

        # Convert X and y to 2D and 1D numpy arrays, respectively.
        self.X = np.array(self.X)
        self.y = np.array(self.y)

        # If scaling required (default yes), then scale, keeping handles to X and y scalers.
        if self.options['scale']:
            self.scaler_X = MinMaxScaler().fit(self.X)
            self.X = self.scaler_X.transform(self.X)
            self.scaler_y = MinMaxScaler().fit(self.y.reshape(-1, 1))
            self.y = self.scaler_y.transform(self.y)

    def predict(self, X_test):
        self.X_test = X_test

        # Forecast timestamps match timestamps in X_test
        timestamps = self.X_test.index

        # Convert to numpy array
        self.X_test = np.array(self.X_test)

        if self.options['scale']:
            X_test_scaled = self.scaler_X.transform(self.X_test)
            y_test_scaled = np.array(self.model.predict(X_test_scaled))
            self.y_test = self.scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
        else:
            self.y_test = self.model.predict(self.X_test)

        # Convert to pandas dataframe
        self.y_test = pd.DataFrame({'timestamp': timestamps, 'solarpower': self.y_test}).set_index('timestamp')

        return self.y_test
