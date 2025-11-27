import numpy as np

def MSE(preds, targets):
    diff = preds - targets
    loss = (diff ** 2).mean()
    grad = (2.0 * diff) / preds.shape[0]
    return loss, grad

def MAE(preds, targets, eps=1e-8):
    diff = preds - targets
    loss = np.abs(diff).mean()
    grad = np.sign(diff) / preds.shape[0]
    return loss, grad

def Huber(preds, targets, delta=1.0):
    diff = preds - targets
    abs_diff = np.abs(diff)

    quad = abs_diff < delta
    linear = ~quad

    loss = (
        0.5 * (diff[quad] ** 2).mean()
        + delta * (abs_diff[linear] - 0.5 * delta).mean()
    )

    grad = np.where(
        abs_diff < delta,
        diff,
        delta * np.sign(diff)
    ) / preds.shape[0]

    return loss, grad

def LogCosh(preds, targets):
    diff = preds - targets
    loss = np.mean(np.log(np.cosh(diff)))
    grad = np.tanh(diff) / preds.shape[0]
    return loss, grad
