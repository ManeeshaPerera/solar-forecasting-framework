DATA = 'data/'
RESULTS = 'results/'

# a dictionary including all information relevant to the forecast horizon and resolution
HORIZON_INFO = {
    '1D': {
        'resolution': 'H',
        'horizon_as_int': 24,
        'resolution_as_str': '1H',
        'arima_params': {
            'seasonal_freq': 24,
            'seasonality': False,
            'fourier': True,
            'fourier_terms': None,
            'maxiter': None,
        }
    }
}
