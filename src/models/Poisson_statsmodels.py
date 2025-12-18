import statsmodels.api as sm


class poisson_model:

    def __init__(self, x, y):
        self.model= sm.GLM(y, x, family=sm.families.NegativeBinomial())
        pass

    def fit(self):
        self.model_fit = self.model.fit()

    def predict(self, x):
        return self.model_fit.predict(x)
    
