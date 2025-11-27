from batch_iterator import batch_iterator

class Trainer:
    def __init__(
        self,
        model,
        input_dim,
        hidden_sizes,
        optimizer,
        loss_fn,
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

