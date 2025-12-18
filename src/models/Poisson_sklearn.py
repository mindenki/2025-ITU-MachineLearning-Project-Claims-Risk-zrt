from sklearn.linear_model import PoissonRegressor


class poisson_model:

    def __init__(self, alpha = None, max_iter = None):

        self.model = PoissonRegressor(alpha = alpha, max_iter = max_iter)
        pass

    def fit(self, x, y):
        self.model_fit = self.model.fit(x, y)

    def predict(self, X_test):
        return self.model_fit.predict(X_test)


