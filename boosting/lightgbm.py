import lightgbm as lgb
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd


class LIGHTGBM:
    def __init__(self, X, y, val_X, val_y, horizon, learning_rate=0.05, early_stopping_round=50, num_iterations=200,
                 random_state=11, seed=101, objective='regression', num_leaves=None, feature_fraction=None,
                 bagging_fraction=None, max_depth=None, min_split_gain=None, min_child_weight=None, metric='rmse',
                 init_points=10, n_iter=10, lags=True):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.early_stopping_round = early_stopping_round
        self.num_iterations = num_iterations
        self.random_state = random_state
        self.seed = seed
        self.objective = objective
        self.num_leaves = num_leaves
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.max_depth = max_depth
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.metric = metric
        self.init_points = init_points
        self.n_iter = n_iter
        self.val_X = val_X
        self.val_y = val_y
        self.model = None
        self.horizon = horizon
        self.lags = lags

    def hyper_parameter_search(self):
        # Range of hyper parameters
        param_range = {
            'num_leaves': (80, 100),
            'feature_fraction': (0.1, 0.9),
            'bagging_fraction': (0.8, 1),
            'max_depth': (17, 25),
            'min_split_gain': (0.001, 0.1),
            'min_child_weight': (10, 25),
            'learning_rate': (0.000001, 1)
        }

        optimizer = BayesianOptimization(self._crossval, param_range, random_state=self.random_state)

        # Optimize
        optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)

        lightgbm_params = optimizer.max['params']

        self.num_leaves = int(lightgbm_params['num_leaves'])
        self.feature_fraction = lightgbm_params['feature_fraction']
        self.bagging_fraction = lightgbm_params['bagging_fraction']
        self.max_depth = int(lightgbm_params['max_depth'])
        self.min_split_gain = lightgbm_params['min_split_gain']
        self.min_child_weight = lightgbm_params['min_child_weight']
        self.learning_rate = lightgbm_params['learning_rate']

    def _crossval(self, num_leaves, feature_fraction, bagging_fraction, max_depth, min_split_gain,
                  min_child_weight, learning_rate):
        data = lgb.Dataset(self.X, self.y)
        # default params
        params = {'objective': self.objective, 'num_iterations': self.num_iterations,
                  'learning_rate': learning_rate, 'early_stopping_round': self.early_stopping_round,
                  'metric': self.metric, "num_leaves": int(round(num_leaves)),
                  'feature_fraction': max(min(feature_fraction, 1), 0),
                  'bagging_fraction': max(min(bagging_fraction, 1), 0), 'max_depth': int(round(max_depth)),
                  'min_split_gain': min_split_gain, 'min_child_weight': min_child_weight, 'verbose': -1,
                  'num_threads': 6}

        # cross validation
        cv_results = lgb.cv(params, data, nfold=5, seed=self.seed, categorical_feature=[], stratified=False,
                            verbose_eval=None, metrics=['rmse'])

        return -np.min(cv_results['rmse-mean'])

    def train(self):
        params = {'objective': self.objective, 'num_iterations': self.num_iterations,
                  'learning_rate': self.learning_rate, 'early_stopping_round': self.early_stopping_round,
                  'metric': self.metric, "num_leaves": self.num_leaves,
                  'feature_fraction': self.feature_fraction,
                  'bagging_fraction': self.bagging_fraction, 'max_depth': self.max_depth,
                  'min_split_gain': self.min_split_gain, 'min_child_weight': self.min_child_weight, 'num_threads': 6}

        lgb_train = lgb.Dataset(self.X, self.y)
        lgb_eval = lgb.Dataset(self.val_X, self.val_y, reference=lgb_train)
        val_dic = {}

        gbm = lgb.train(params, lgb_train, num_boost_round=200,
                        valid_sets=lgb_eval, evals_result=val_dic, verbose_eval=True)

        self.model = gbm

    def get_forecast(self, test):
        all_forecasts = []
        for sample in range(0, len(test), self.horizon):
            fc = self.recursive_forecast(test[sample: sample + self.horizon])
            all_forecasts.extend(fc)
        return all_forecasts

    def recursive_forecast(self, test_sample):
        forecasts = []
        for step in range(0, self.horizon):
            x_val = pd.DataFrame(test_sample.iloc[step]).transpose()
            if self.lags:
                for i in range(0, len(forecasts)):
                    x_val['lag_' + str(step - i)] = forecasts[i]

            forecasts.extend(self.model.predict(x_val))
        return forecasts
