from sklearn.tree import DecisionTreeRegressor

class DT:
    def __init__(self, criterion=None, max_depth=None, min_samples_split=None, min_samples_leaf=None, random_state=42):
        self.model = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=random_state)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)