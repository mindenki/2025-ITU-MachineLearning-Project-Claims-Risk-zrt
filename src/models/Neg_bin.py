import numpy as np
import statsmodels.api as sm

class neg_bin_model:
    def __init__(self, X, y):
        self.X = sm.add_constant(X)
        self.y = y

        self.model = sm.GLM(
            self.y,
            self.X,
            family=sm.families.NegativeBinomial()
        )
        self.model_fit = None

    def fit(self):
        self.model_fit = self.model.fit()
        return self.model_fit
    def predict(self, X):
        if self.model_fit is None:
            raise RuntimeError("Model must be fit before predict().")

        X = sm.add_constant(X)
        return self.model_fit.predict(X)
