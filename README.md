## A Forecasting Framework for Residential Solar Photovoltaic Power Forecasting


This repository includes the code of the forecasting framework proposed in paper titled as "A Forecasting Framework for Multi-resolution and Multi-horizon Residential Solar Photovoltaic Power Forecasting"

#### Directory Structure
```
|-main.py - example code to run all forecast approaches and visualise
|-data - data folder
|-constants.py - constants related to the project
|-plot.py - functions related to plotting
|-run_base_models.py - code to run all base learners
|-run_combinations.py - code to run all forecast combination methods
|-timeseries_split.py - class to split time series data
|-run_global_models.py - code to run global models
|-forecast_combinations_approach - code for the forecast combination approach (this runs the run_bse_models and run_combinations)
|-global_forecast_approach - code for the gloabl forecast approach (this runs the run_global_models)
|-util.py - additional functions
|-boosting - python class implementation of the LightGBM model
|-combinations - class implemenations for average, pso methods and recursive ensemble
|-machine_learning - class implementations for linear and support vector regression models
|-naive_models - class implementation of the seasonal naive model
|-tsmodels - class implenation of the (s)ARIMA and (s)ARIMAX models
```

#### Package Requirements
Package requirements are listed in `requirements.txt`

#### Example Code
A working example of the code is in `main.py` for small sample dataset provided in the data folder.

##### Running the code
`python main.py`

The following is an example output of the `main.py`. It will provide the mean MASE across all test samples and a visualisation of the forecasts produced by all approaches.

```Mean Absolute Scaled Error (MASE) for the test samples```
```json
{
    "sn": 0.4617352375226971, 
    "(s)arima": 0.8710616362438794,
    "(s)arimax": 0.7290526167551878, 
    "mlr": 0.9119233826493682, 
    "svr": 0.8491110380174636, 
    "pso": 1.0867953618937825, 
    "pso [0,1]": 0.4541319252235865, 
    "pso- convex": 0.6832882493648744, 
    "average": 0.6411990407241464, 
    "re": 0.7161379879466082,
    "lgb": 1.4269188941948645
}
 ```
```Visulatisation of the forecasts produced by all approaches``` 
![forecasts](data/example_output.png)