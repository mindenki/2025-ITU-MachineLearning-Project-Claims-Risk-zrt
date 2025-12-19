from sklearn.model_selection import RandomizedSearchCV, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)

def run_random_search(pipeline, param_dist, X, y, n_iter=25, scoring="neg_root_mean_squared_error", cv=cv, n_jobs= -1):
    """ Run Randomized Search CV to find the best hyperparameters for the given pipeline."""
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=3,
        n_jobs=n_jobs,
        random_state=42,
        return_train_score=True,
        refit=True
    )
    search.fit(X, y)
    
    return search