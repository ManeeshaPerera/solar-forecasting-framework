class ForecastModels:
    """ Base class for any forecasting model """

    def __init__(self, train_data, test_data, horizon, resolution, horizon_as_string=None, resolution_as_string=None,
                 train_features=None, test_features=None):
        self.train_data = train_data
        self.test_data = test_data
        self.horizon = horizon
        self.resolution = resolution
        self.horizon_as_string = horizon_as_string
        self.resolution_as_string = resolution_as_string
        self.train_features = train_features
        self.test_features = test_features
