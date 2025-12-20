from math import sqrt
import numpy as np
from ..utils.batch_iterator import batch_iterator
from ..utils.optimiziers import SGD, SGDMomentum, Adam, Adagrad
from ..utils.losses import MSE, MAE, Huber, LogCosh


# Mapping string identifiers to optimizer classes
OPTIMIZERS = {
    "sgd": SGD,
    "sgd_momentum": SGDMomentum,
    "adam": Adam,
    "adagrad": Adagrad,
}

# Mapping string identifiers to loss functions
LOSSES = {
    "mse": MSE,
    "mae": MAE,
    "huber": Huber,
    "logcosh": LogCosh
}


# ReLU activation function
def ReLU(x):
    return np.maximum(0, x)

# Derivative of ReLU
def dReLU(x):
    return (x > 0.0).astype(x.dtype)

# Identity activation
def identity(x):
    return x

# Derivative of identity activation
def didentity(x):
    return np.ones_like(x)


class Layer:
    """
    Fully connected neural network layer with manual forward and backward passes.
    """
    def __init__(self, in_feats, out_feats, activation, dactivation):
        # He initialization for stable gradients with ReLU activations
        rng = np.random.default_rng(42)
        std = sqrt(2.0 / in_feats)
        self.W = rng.normal(0.0, std, size=(out_feats, in_feats))
        self.b = np.zeros(out_feats, dtype=np.float32)

        # Cached values for backpropagation
        self.x = None
        self.y = None

        # Gradient buffers
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Activation function and its derivative
        self.act = activation
        self.dact = dactivation

    def forward(self, x):
        # Store input for gradient computation
        self.x = x
        self.y = np.matmul(x, self.W.T) + self.b
        return self.act(self.y)

    def backward(self, grad_out):
        # Chain rule: gradient through activation
        D = grad_out * self.dact(self.y)

        # Accumulate gradients for weights and bias
        self.dW += np.matmul(D.T, self.x)
        self.db += D.sum(axis=0)

        # Propagate gradient to previous layer
        grad_in = np.matmul(D, self.W)
        return grad_in

    def step(self, lr):
        # Parameter update using gradient descent
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def zero_grad(self):
        # Reset gradients after each batch
        self.dW.fill(0.0)
        self.db.fill(0.0)


class MLP:
    """
    Simple feed-forward multilayer perceptron implemented from scratch.
    """
    def __init__(self, input_dim, hidden_sizes):
        output_dim = 1
        sizes = [input_dim] + list(hidden_sizes) + [output_dim]

        # ReLU for hidden layers, identity for output
        activations = [(ReLU, dReLU)] * len(hidden_sizes) + [(identity, didentity)]

        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(
                Layer(
                    sizes[i],
                    sizes[i + 1],
                    activations[i][0],
                    activations[i][1],
                )
            )

    def forward(self, x):
        # Sequential forward pass through all layers
        pred = x
        for layer in self.layers:
            pred = layer.forward(pred)
        return pred

    def backward(self, grad):
        # Backpropagate gradients in reverse order
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def zero_grad(self):
        # Clear gradients for all layers
        for layer in self.layers:
            layer.zero_grad()


class Trainer:
    """
    Training wrapper providing batching, optimization, and optional validation.
    Designed to be compatible with sklearn-style hyperparameter search.
    """
    def __init__(
        self,
        input_dim,
        hidden_sizes,
        optimizer,
        loss_fn,
        model=MLP,
        batch_size=32,
        epochs=100,
        shuffle=True,
    ):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.model = model
        self.model_ = self.model(input_dim=self.input_dim, hidden_sizes=self.hidden_sizes)

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle

    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        # Ensure target has shape (n_samples, 1)
        y = y.reshape(-1, 1)
        history = {"train_loss": [], "val_loss": []}

        # Instantiate optimizer
        if isinstance(self.optimizer, str):
            optimizer = OPTIMIZERS[self.optimizer]()
        else:
            optimizer = self.optimizer

        # Resolve loss function
        if isinstance(self.loss_fn, str):
            loss_fn = LOSSES[self.loss_fn]
        else:
            loss_fn = self.loss_fn

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            # Mini-batch training loop
            for Xb, yb in batch_iterator(
                X, y, batch_size=self.batch_size, shuffle=self.shuffle
            ):
                preds = self.model_.forward(Xb)
                loss, grad = loss_fn(preds, yb)

                self.model_.zero_grad()
                self.model_.backward(grad)
                optimizer.step(self.model_)

                epoch_loss += loss
                n_batches += 1

            epoch_loss /= max(n_batches, 1)
            history["train_loss"].append(epoch_loss)

            # Optional validation loss
            if X_val is not None and y_val is not None:
                yv = y_val.reshape(-1, 1)
                val_preds = self.model_.forward(X_val)
                val_loss, _ = loss_fn(val_preds, yv)
                history["val_loss"].append(val_loss)
            else:
                history["val_loss"].append(None)

            if verbose:
                if X_val is not None and y_val is not None:
                    print(
                        f"Epoch {epoch:3d} | "
                        f"train_loss={epoch_loss:.6f} | "
                        f"val_loss={val_loss:.6f}"
                    )
                else:
                    print(f"Epoch {epoch:3d} | train_loss={epoch_loss:.6f}")

        return history

    def predict(self, X):
        # Forward pass only, no gradients
        preds = self.model_.forward(X)
        return preds.ravel()

    def get_params(self, deep=True):
        # Required for sklearn compatibility
        params = {
            "input_dim": self.input_dim,
            "hidden_sizes": self.hidden_sizes,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "shuffle": self.shuffle,
            "loss_fn": self.loss_fn,
            "optimizer": self.optimizer,
        }

        if deep and hasattr(self.optimizer, "__dict__"):
            for k, v in self.optimizer.__dict__.items():
                params[f"optimizer__{k}"] = v

        return params

    def set_params(self, **params):
        # Allow sklearn-style parameter updates
        rebuild_model = False

        for key, value in params.items():
            if "__" in key:
                obj_name, attr_name = key.split("__", 1)
                obj = getattr(self, obj_name)
                setattr(obj, attr_name, value)
            else:
                setattr(self, key, value)
                if key in ("input_dim", "hidden_sizes"):
                    rebuild_model = True

        if rebuild_model:
            self.model_ = self.model(
                input_dim=self.input_dim,
                hidden_sizes=self.hidden_sizes
            )

        return self
