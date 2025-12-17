from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_poisson_deviance


def eval_model(model, X, y):
    """ Last evaluation of the model on test set."""
    y_pred = model.predict(X)
    metrics = {
        "MAE": mean_absolute_error(y, y_pred),
        "RMSE": root_mean_squared_error(y, y_pred, squared=False),
        "PoissonDeviance": mean_poisson_deviance(y, y_pred),
    }
    return metrics, y_pred