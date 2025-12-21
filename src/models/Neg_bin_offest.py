import numpy as np
import statsmodels.api as sm

class neg_bin_model_offset:
    def __init__(self, X, y, exposure, eps=1e-12):
        X = sm.add_constant(X)
        self.X = X
        self.y = y

        exposure = np.clip(exposure, eps, None)
        self.offset = np.log(exposure)

        self.model = sm.GLM(
            y,
            self.X,
            family=sm.families.NegativeBinomial(),
            offset=self.offset
        )
        self.model_fit = None

    def fit(self):
        self.model_fit = self.model.fit()
        return self.model_fit

    def predict(self, X, exposure, eps=1e-12):
        if self.model_fit is None:
            raise RuntimeError("Model must be fit before calling predict().")

        X = sm.add_constant(X)
        exposure = np.clip(exposure, eps, None)
        offset = np.log(exposure)

        return self.model_fit.predict(X, offset=offset)
