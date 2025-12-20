import statsmodels.api as sm


class poisson_model:

    def __init__(self, X, y):
        self.model= sm.GLM(y, X, family=sm.families.NegativeBinomial())
        self.model_fit = None

    def fit(self):
        self.model_fit = self.model.fit()

    def predict(self, X):
        return self.model_fit.predict(X)