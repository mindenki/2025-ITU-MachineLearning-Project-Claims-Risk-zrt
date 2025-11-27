import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from torch.utils.tensorboard import SummaryWriter

from src.models.MLP_scratch import MLP
from src.utils.losses import MSE, MAE, Huber, LogCosh
from src.utils.optimiziers import SGD, SGDMomentum, Adagrad, Adam
from src.utils.trainer import Trainer
from src.data.load_data import load_data
from src.preprocess.preprocess_MLP import prepare_datasets_with_clusters

seed = 42
np.random.seed(seed)

train_df, test_df = load_data(raw=False)

X_train, X_test, y_train_std, y_test_std, artefacts = prepare_datasets_with_clusters(
    train_df,
    test_df,
    n_components=40,
    k_clusters=5,
    seed=seed,
)


y_scaler = artefacts["y_scaler"]

input_dim = X_train.shape[1]
hidden_sizes = [256, 128]

epochs = 20
batch_size = 2048

optimizer = Adam(lr=3e-3)
loss_fn  = MSE

net = Trainer(
    model=MLP,
    input_dim=input_dim,
    hidden_sizes=hidden_sizes,
    optimizer=optimizer,
    loss_fn=loss_fn,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
)


log_dir = "../runs"
os.mkdir(log_dir, exists=True)
writer = SummaryWriter(Path(log_dir))
writer.add_text("config", str(net.get_params()))

history = net.fit(
    X_train,
    y_train_std,
    X_val=X_test,
    y_val=y_test_std,
    verbose=True,
)

for epoch, (train_loss, val_loss) in enumerate(
    zip(history["train_loss"], history["val_loss"]), start=1
):
    writer.add_scalar("Loss/train", train_loss, epoch)
    if val_loss is not None:
        writer.add_scalar("Loss/val", val_loss, epoch)

writer.close()

print("Training completed.")

def invert_target(std_z, scaler):
    """
    std_z: standardized z (mean-0, std-1), where z = log1p(rate)
    scaler: the StandardScaler fitted on z
    returns: z (unstandardized)
    """
    std_z = np.asarray(std_z).reshape(-1, 1)
    z = scaler.inverse_transform(std_z).ravel()
    return z

def predict_rate(model_wrapper, X, scaler):
    """
    model_wrapper: MLPWrapper
    X: features
    scaler: StandardScaler on z = log1p(rate)
    returns: predicted rate per exposure unit (>=0)
    """
    z_std_hat = model_wrapper.predict(X)
    z_hat = invert_target(z_std_hat, scaler)
    rate_hat = np.expm1(z_hat)
    rate_hat = np.clip(rate_hat, 0.0, None)
    return rate_hat

X_train_final = X_train
X_test_final  = X_test

rate_hat_train = predict_rate(net, X_train_final, y_scaler)
rate_hat_test  = predict_rate(net, X_test_final,  y_scaler)

z_train_true = invert_target(y_train_std, y_scaler)
z_test_true  = invert_target(y_test_std,  y_scaler)

rate_true_train = np.expm1(z_train_true)
rate_true_test  = np.expm1(z_test_true)

rmse_train = root_mean_squared_error(rate_true_train, rate_hat_train, squared=False)
mae_train  = mean_absolute_error(rate_true_train, rate_hat_train)
r2_train   = r2_score(rate_true_train, rate_hat_train)

rmse_test = root_mean_squared_error(rate_true_test, rate_hat_test, squared=False)
mae_test  = mean_absolute_error(rate_true_test, rate_hat_test)
r2_test   = r2_score(rate_true_test, rate_hat_test)

print("\n=== Evaluation (rates per exposure unit) ===")
print(f"Train: RMSE={rmse_train:.6f}, MAE={mae_train:.6f}, R2={r2_train:.4f}")
print(f"Test : RMSE={rmse_test:.6f}, MAE={mae_test:.6f}, R2={r2_test:.4f}")
