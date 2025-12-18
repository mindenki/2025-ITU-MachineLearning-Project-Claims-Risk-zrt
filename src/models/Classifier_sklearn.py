from sklearn.ensemble import RandomForestClassifier

class RFC:

    def __init__(self, n_estimators=None, max_depth=None, min_samples_split=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, )
        pass

    def fit(self, x, y):
        self.model_fit = self.model.fit(x, y)

    def predict(self, X_test):
        return self.model_fit.predict(X_test)