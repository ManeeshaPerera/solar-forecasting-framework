## A Forecasting Framework for Residential Solar Photovoltaic Power Forecasting


This repository includes the code of the forecasting framework proposed in paper titled as "A Forecasting Framework for Multi-resolution and Multi-horizon Residential Solar Photovoltaic Power Forecasting"

Directory Structure:
```
|-main.py - example code to run all forecast models (base learners, combinations) and visulaise forecasts
|-data - data folder
|-constants.py - constants related to the project
|-plot.py - functions related to plotting
|-run_base_models.py - code to run all base learners
|-run_combination.py - code to run all forecast combination methods
|-timeseries_split.py - class to split time series data
|-util.py - additional functions
|- 
```
Package requirements are listed in `requirements.txt`

A working example of the code is in `main.py` on running the base forecasters and the combinations on a small sample dataset provided in the data folder.
The following is an example output of the `main.py`

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
 
![forecasts](data/example_output.png)