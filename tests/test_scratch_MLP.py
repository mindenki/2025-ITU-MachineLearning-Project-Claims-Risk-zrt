import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.MLP_scratch import MLP, MSE, batch_iterator
from src.data.load_data import load_data
from src.data.preprocess import prepare_datasets_with_clusters
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

seed = 42
np.random.seed(seed)

train_df, test_df = load_data(raw=False)

X_train, X_test, y_train_std, y_test_std, artefacts = (
    prepare_datasets_with_clusters(
        train_df,
        test_df,
        n_components=40,
        k_clusters=5,
        seed=seed,
    )
)

epochs = 20
batch_size = 2048
lr = 3e-3

model = MLP(40, [256, 128], lr)
loss_func = MSE

log_dir = "../runs"

writer = SummaryWriter(Path(log_dir))
global_step = 0

for epoch in range(epochs):
    running = 0.0
    num_batches = 0

    for Xb, yb in batch_iterator(X_train, y_train_std, batch_size, shuffle=True, seed=epoch):
        preds = model.forward(Xb)
        loss, dyp = loss_func(preds, yb)

        model.zero_grad()
        model.backward(dyp)
        model.step()

        running += float(loss)
        num_batches += 1

    epoch_loss = running / max(1, num_batches)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss (MSE on z_std): {epoch_loss:.4f}")

print("Training completed.")

def invert_target(std_z, scaler):
    z = scaler.inverse_transform(np.asarray(std_z).reshape(-1, 1)).ravel()
    return z

def predict_rate(model, X, scaler):
    z_std_hat = model.forward(X).ravel()
    z_hat = invert_target(z_std_hat, scaler)
    rate_hat = np.expm1(z_hat)
    rate_hat = np.clip(rate_hat, 0.0, None)
    return rate_hat

def invert_target(std_z, scaler):
    z = scaler.inverse_transform(np.asarray(std_z).reshape(-1, 1)).ravel()
    return z

def predict_rate(model, X, scaler):
    z_std_hat = model.forward(X).ravel()
    z_hat = invert_target(z_std_hat, scaler)
    rate_hat = np.expm1(z_hat)
    rate_hat = np.clip(rate_hat, 0.0, None)
    return rate_hat

rate_hat_train = predict_rate(model, X_train_final, y_scaler)
rate_hat_test  = predict_rate(model, X_test_final,  y_scaler)

z_train_true = invert_target(y_train_std, y_scaler)
z_test_true  = invert_target(y_test_std,  y_scaler)

rate_true_train = np.expm1(z_train_true)
rate_true_test  = np.expm1(z_test_true)

rmse_train = mean_squared_error(rate_true_train, rate_hat_train)
mae_train  = mean_absolute_error(rate_true_train, rate_hat_train)
r2_train   = r2_score(rate_true_train, rate_hat_train)

rmse_test = mean_squared_error(rate_true_test, rate_hat_test)
mae_test  = mean_absolute_error(rate_true_test, rate_hat_test)
r2_test   = r2_score(rate_true_test, rate_hat_test)

print("\n=== Evaluation (rates per exposure unit) ===")
print(f"Train: RMSE={rmse_train:.6f}, MAE={mae_train:.6f}, R2={r2_train:.4f}")
print(f"Test : RMSE={rmse_test:.6f}, MAE={mae_test:.6f}, R2={r2_test:.4f}")
