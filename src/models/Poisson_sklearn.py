from sklearn.linear_model import PoissonRegressor


class poisson_model():

    def __init__(self, alpha = None, max_iter = None):
        self.model = PoissonRegressor(alpha, max_iter)
        pass

    def poisson_model(self, x, y):
        return self.model.fit(x, y)

    def predict(self, X_test):
        return self.model.predict(X_test)


