from sklearn.svm import SVR
from machinelearning._base_class import MachineLearning


class SupportVectorRegression(MachineLearning):
    def __init__(self):
        super().__init__()

    def _set_default_options(self):
        super()._set_default_options()

    def fit(self, X, y, options=None, gamma='scale'):
        super().fit(X, y, options=options)
        self.model = SVR(gamma=gamma).fit(self.X, self.y.ravel())

    def predict(self, X_test):
        return super().predict(X_test)
