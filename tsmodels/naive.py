from tsmodels._base_class import ForecastModels
from naive_models.persistence import SeasonalNaive
import pandas as pd


class NaiveModel(ForecastModels):
    def __init__(self, train_data, test_data, horizon, resolution, horizon_as_string, resolution_as_string):
        super().__init__(train_data, test_data, horizon, resolution, horizon_as_string, resolution_as_string)

    def get_forecasts(self):
        recent_data = self.train_data[-self.horizon:]
        forecast_results = []
        for h in range(0, len(self.test_data.index), self.horizon):
            sn_model = SeasonalNaive()
            sn_model.fit(recent_data,
                         options={'resolution': self.resolution_as_string, 'period': self.horizon_as_string})

            sn_forecast = sn_model.predict()
            forecast_results.append(sn_forecast)
            recent_data = recent_data.append(self.test_data[h:h + self.horizon])
        return pd.concat(forecast_results)
