from tsmodels._base_class import ForecastModels
import pandas as pd


class MachineLearningRegression(ForecastModels):
    def __init__(self, model, train_data, test_data, horizon, resolution, horizon_as_string, resolution_as_string,
                 train_features, test_features):
        super().__init__(train_data, test_data, horizon, resolution, horizon_as_string, resolution_as_string,
                         train_features, test_features)
        self.model = model

    def get_forecasts(self):
        forecast_results = []
        self.model.fit(self.train_features, self.train_data, options={'resolution': self.resolution_as_string})
        for i in range(0, len(self.test_data.index), self.horizon):
            forecast_results.append(self.model.predict(self.test_features[i:i + self.horizon]))
        return pd.concat(forecast_results)

    def get_forecast_high_resolution(self, window_length):
        forecast_results = []
        self.model.fit(self.train_features, self.train_data, options={'resolution': self.resolution_as_string})
        for i in range(0, len(self.test_data.index), window_length):
            forecast_results.append(self.model.predict(self.test_features[i:i + self.horizon]))
        return pd.concat(forecast_results)

