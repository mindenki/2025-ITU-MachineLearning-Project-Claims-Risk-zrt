import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_poisson_deviance

def logclaim_to_claimrate(y_log):
    """ Convert log-claims back to claim rates."""
    return np.expm1(y_log)  # exp(y) - 1

def eval_regressor_model(model, X, y, y_pred=None, log_given=True):
    """ Evaluation of the model on the given set."""
    if log_given:
        y_pred = model.predict(X)
    metrics = {
        "MAE": mean_absolute_error(y, y_pred),
        "RMSE": root_mean_squared_error(y, y_pred),
    }
    
        
    return metrics, y_pred

