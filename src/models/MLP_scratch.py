from math import sqrt
import numpy as np
from ..utils.batch_iterator import batch_iterator
from ..utils.optimiziers import SGD, SGDMomentum, Adam, Adagrad
from ..utils.losses import MSE, MAE, Huber, LogCosh


OPTIMIZERS = {
    "sgd": lambda: SGD(lr=1e-3),
    "sgd_momentum": lambda: SGDMomentum(lr=1e-3, momentum=0.9),
    "adam": lambda: Adam(lr=1e-3),
    "adagrad": lambda: Adagrad(lr=1e-2)
}

LOSSES = {
    "mse": MSE,
    "mae": MAE,
    "huber": Huber,
    "logcosh": LogCosh
}


def ReLU(x):
    return np.maximum(0, x)

def dReLU(x):
    return (x>0.0).astype(x.dtype)
    
class Layer:
    def __init__(self, in_feats, out_feats, activation, dactivation):
        # He parameter initialization
        rng = np.random.default_rng(42)
        std = sqrt(2.0/in_feats)
        self.W = rng.normal(0.0, std, size=(out_feats, in_feats))
        self.b = np.zeros(out_feats, dtype=np.float32)
        
        self.x = None
        self.y = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.act = activation
        self.dact = dactivation
        
    def forward(self, x):
        self.x = x
        self.y = np.matmul(x, self.W.T) + self.b
        return self.act(self.y)
    
    def backward(self, grad_out):
        
        # derivative of loss wrt the activated output (post activation)
        D = grad_out * self.dact(self.y)
        
        # derivative of loss wrt the weights
        # summing over the batch
        self.dW += np.matmul(D.T, self.x)
        
        # derivative of loss wrt the bias
        # summing over the batch
        self.db += D.sum(axis=0)
        
        # derivative of loss wrt the inputs
        grad_in = np.matmul(D, self.W)
        
        return grad_in
        
    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
        
    def zero_grad(self):
        # reseting the gradients for weigths and biases after batch
        self.dW.fill(0.0)
        self.db.fill(0.0)
    
class MLP:
    def __init__(self, input_dim, hidden_sizes):
        output_dim = 1
        sizes = [input_dim] + list(hidden_sizes) + [output_dim]
        activations = [(ReLU, dReLU)] * len(hidden_sizes) + [(lambda x:x, lambda x:np.ones_like(x))]
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Layer(sizes[i], sizes[i+1], activations[i][0], activations[i][1]))
            
    def forward(self, x):
        pred = x
        for layer in self.layers:
            pred = layer.forward(pred)
        return pred
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

class Trainer:
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
        y = y.reshape(-1, 1)
        history = {"train_loss": [], "val_loss": []}

        if isinstance(self.optimizer, str):
            self.optimizer = OPTIMIZERS[self.optimizer]()
        if isinstance(self.loss_fn, str):
            self.loss_fn = LOSSES[self.loss_fn]
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            for Xb, yb in batch_iterator(X, y, batch_size=self.batch_size, shuffle=self.shuffle):
                preds = self.model_.forward(Xb)
                
                loss, grad = self.loss_fn(preds, yb)

                self.model_.zero_grad()
                self.model_.backward(grad)
                self.optimizer.step(self.model_)

                epoch_loss += loss
                n_batches += 1

            epoch_loss /= max(n_batches, 1)
            history["train_loss"].append(epoch_loss)

            if X_val is not None and y_val is not None:
                yv = y_val.reshape(-1, 1)
                val_preds = self.model_.forward(X_val)
                val_loss, _ = self.loss_fn(val_preds, yv)
                history["val_loss"].append(val_loss)
            else:
                history["val_loss"].append(None)

            if verbose:
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch:3d} | train_loss={epoch_loss:.6f} | val_loss={val_loss:.6f}")
                else:
                    print(f"Epoch {epoch:3d} | train_loss={epoch_loss:.6f}")

        return history

    def predict(self, X):
        preds = self.model_.forward(X)
        return preds.ravel()
    
    def get_params(self, deep=True):
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
        rebuild_model = False

        for key, value in params.items():
            if "__" in key:
                obj_name, attr_name = key.split("__", 1)
                if not hasattr(self, obj_name):
                    raise ValueError(f"Unknown nested object '{obj_name}' in param '{key}'")
                obj = getattr(self, obj_name)
                if not hasattr(obj, attr_name):
                    raise ValueError(f"Object '{obj_name}' has no attribute '{attr_name}'")
                setattr(obj, attr_name, value)

            else:
                if not hasattr(self, key):
                    raise ValueError(f"Unknown parameter '{key}' for MLPWrapper")

                setattr(self, key, value)

                if key in ("input_dim", "hidden_sizes"):
                    rebuild_model = True

        if rebuild_model:
            self.model_ = self.model(input_dim=self.input_dim, hidden_sizes=self.hidden_sizes)

        return self

