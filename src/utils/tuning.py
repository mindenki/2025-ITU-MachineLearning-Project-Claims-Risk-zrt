import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold

from sklearn.metrics import log_loss

def weighted_log_loss(y_true, y_pred_proba, **kwargs):
    w = np.where(y_true == 1, 5, 1)
    return log_loss(y_true, y_pred_proba, sample_weight=w)

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