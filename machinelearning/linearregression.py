from sklearn import linear_model
from machinelearning._base_class import MachineLearning


class LinearRegression(MachineLearning):
    def __init__(self):
        super().__init__()

    def _set_default_options(self):
        super()._set_default_options()

    def fit(self, X, y, options=None):
        super().fit(X, y, options=options)
        # Linear regression
        self.model = linear_model.LinearRegression(fit_intercept=False, copy_X=True).fit(self.X, self.y)

    def predict(self, X_test):
        return super().predict(X_test)
