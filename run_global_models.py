import pandas as pd
import constants as const
import util
from boosting.lightgbm import LIGHTGBM
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def run_global_model(data_info, horizon):
    feature_info = {
        'past_tim_lags': 48
    }
    horizon_int = const.HORIZON_INFO[horizon]['horizon_as_int']

    data = {}
    for house_id, info in data_info.items():
        house_data = info['data']
        train_end = info['train'].index[-1]
        val_start = info['val'].index[0]
        val_end = info['val'].index[-1]
        test_start = info['test'].index[0]
        test_end = info['test'].index[-1]
        external_variables = info['external']

        scaler = StandardScaler()
        scaler.fit(info['train'])
        house_data['solarpower'] = scaler.transform(house_data)

        # create lags
        for i in range(1, feature_info['past_tim_lags'] + 1):
            house_data["lag_{}".format(i)] = house_data['solarpower'].shift(i)

        # drop the lags with NAN
        house_data = house_data.dropna()

        # add external features relevant to particular house
        scaler2 = StandardScaler()
        scaler2.fit(external_variables[:train_end])
        external_variables[external_variables.columns] = scaler2.transform(external_variables)

        house_data = pd.concat([house_data, external_variables], axis=1)
        house_data = house_data.dropna()

        data[house_id] = {}
        data[house_id]['train'] = house_data[:train_end]
        data[house_id]['val'] = house_data[val_start:val_end]
        data[house_id]['test'] = house_data[test_start:test_end]

    # create the global feature matrix using all data from houses
    TRAIN_X, TRAIN_Y, VAL_X, VAL_Y, test = create_train_val_test(data)

    # Run the lightGBM Model
    print('LightGBM Hyper-parameter tuning and model training')
    lgb = LIGHTGBM(TRAIN_X, TRAIN_Y, VAL_X, VAL_Y, horizon_int, lags=feature_info['past_tim_lags'])
    lgb.hyper_parameter_search()
    lgb.train()

    # get forecast for each house
    forecast_array = []
    for house_id, info in data_info.items():
        scaler = StandardScaler()
        # read house data
        scaler.fit(info['train'])
        test_data = test[house_id]['x_test']

        # get forecasts
        fc_ = lgb.get_forecast(test_data)
        # trans-from the scaled forecasts to back to original scaled
        fc = scaler.inverse_transform(fc_)
        fc_df = pd.DataFrame(fc, index=info['test'].index)
        fc_df.columns = ['lgb']
        forecast_array.append(fc_df)
    return forecast_array


def create_train_val_test(data_dic):
    # create train, val, test samples
    train_matrix = []
    y_train_matrix = []

    x_val_matrix = []
    y_val_matrix = []

    test = {}

    for house_id, house_info in data_dic.items():
        X_train, y_train = util.process_data(house_info['train'])
        train_matrix.append(X_train)
        y_train_matrix.append(y_train)

        # validation
        X_val, y_val = util.process_data(house_info['val'])
        x_val_matrix.append(X_val)
        y_val_matrix.append(y_val)

        # test
        test[house_id] = {}
        X_test = house_info['test'].drop(['solarpower'], axis=1)
        y_test = house_info['test']['solarpower']
        test[house_id]['x_test'] = X_test
        test[house_id]['y_test'] = y_test

    TRAIN_X = pd.concat(train_matrix, ignore_index=True)
    VAL_X = pd.concat(x_val_matrix, ignore_index=True)

    TRAIN_Y = pd.concat(y_train_matrix, ignore_index=True)
    VAL_Y = pd.concat(y_val_matrix, ignore_index=True)

    return TRAIN_X, TRAIN_Y, VAL_X, VAL_Y, test
