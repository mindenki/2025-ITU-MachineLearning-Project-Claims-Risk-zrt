from sklearn.ensemble import RandomForestClassifier

class RFC:

    def __init__(self,
                n_estimators=None,
                max_depth=None,
                min_samples_split=None,
                class_weight=None
            ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,
            class_weight=class_weight
        )
        pass

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]