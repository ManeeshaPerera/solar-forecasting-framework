from tsmodels._base_class import ForecastModels
from pmdarima import auto_arima
from pmdarima.preprocessing import FourierFeaturizer
from pmdarima import pipeline
from pmdarima import arima
import pandas as pd


class AutoRegressive(ForecastModels):
    def __init__(self, train_data, test_data, horizon, resolution, horizon_as_string, resolution_as_string,
                 train_features=None, test_features=None, fourier=None, fourier_terms=None, seasonality=False,
                 seasonal_freq=1,
                 maxiter=None):
        super().__init__(train_data, test_data, horizon, resolution, horizon_as_string, resolution_as_string,
                         train_features, test_features)
        self.fourier_terms = fourier_terms
        self.fourier = fourier
        self.seasonality = seasonality
        self.seasonal_freq = seasonal_freq
        self.maxiter = maxiter
        self.model = None

    def find_arima_params(self):
        # we are fitting a model with fourier terms and the modelling the seasonality as a fourier series
        if self.fourier:
            # Let's create a pipeline with multiple stages.
            model = pipeline.Pipeline([
                ("fourier", FourierFeaturizer(m=self.seasonal_freq, k=self.fourier_terms)),
                ("arima", arima.AutoARIMA(stepwise=True, trace=True, error_action="ignore",
                                          seasonal=False,  # because we use Fourier
                                          suppress_warnings=True))
            ])
            model.fit(self.train_data.iloc[:, 0], exogenous=self.train_features)
        else:
            model = auto_arima(self.train_data.iloc[:, 0], self.train_features,
                               seasonal=self.seasonality,
                               m=self.seasonal_freq, trace=True,
                               error_action='ignore',
                               suppress_warnings=True)
        self.model = model
        return model

    def get_forecasts(self):
        forecast_results = []

        for i in range(0, len(self.test_data.index), self.horizon):
            if self.test_features is not None:
                forecast = self.model.predict(n_periods=self.horizon,
                                              exogenous=self.test_features[i: i + self.horizon])
                self.model.update(self.test_data[i:i + self.horizon].iloc[:, 0],
                                  exogenous=self.test_features[i:i + self.horizon], maxiter=self.maxiter)
            else:
                forecast = self.model.predict(n_periods=self.horizon)
                self.model.update(self.test_data[i:i + self.horizon].iloc[:, 0], maxiter=self.maxiter)
            forecast_results.extend(forecast)
        return pd.DataFrame(forecast_results, index=self.test_data.index, columns=['solarpower'])

    def get_forecast_high_resolution(self, window_length):
        forecast_results = []
        index = []

        for i in range(0, len(self.test_data.index), window_length):
            index.extend(self.test_data[i: i + self.horizon].index)
            if self.test_features is not None:
                forecast = self.model.predict(n_periods=self.horizon,
                                              exogenous=self.test_features[i: i + self.horizon])
                self.model.update(self.test_data[i:i + window_length].iloc[:, 0],
                                  exogenous=self.test_features[i:i + window_length], maxiter=self.maxiter)
            else:
                forecast = self.model.predict(n_periods=self.horizon)
                self.model.update(self.test_data[i:i + window_length].iloc[:, 0], maxiter=self.maxiter)
            forecast_results.extend(forecast)
        return pd.DataFrame(forecast_results, index=pd.Index(index, name='timestamp'), columns=['solarpower'])
