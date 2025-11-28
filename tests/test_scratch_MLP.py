import sys
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from torch.utils.tensorboard import SummaryWriter

from scipy.sparse import load_npz

from src.models.MLP_scratch import MLP, Trainer
from src.utils.losses import MSE, MAE, Huber, LogCosh
from src.utils.optimiziers import SGD, SGDMomentum, Adagrad, Adam
from src.data.load_data import load_data
from src.preprocess.preprocess_MLP import prepare_datasets_with_clusters

seed = 42
np.random.seed(seed)

train_ds, test_ds = load_data(raw=False, target="log_ClaimRate")

X_train, y_train = train_ds
X_test, y_test = test_ds

input_dim = X_train.shape[1]
hidden_sizes = [256, 128]

epochs = 20
batch_size = 2048

optimizer = Adam(lr=3e-3)
loss_fn  = MSE

model = Trainer(
    model=MLP,
    input_dim=input_dim,
    hidden_sizes=hidden_sizes,
    optimizer=optimizer,
    loss_fn=loss_fn,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
)


log_dir = Path("runs")
log_dir.mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(Path(log_dir))
writer.add_text("config", str(model.get_params()))

history = model.fit(
    X_train,
    y_train,
    X_val=X_test,
    y_val=y_test,
    verbose=True,
)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

for epoch, (train_loss, val_loss) in enumerate(
    zip(history["train_loss"], history["val_loss"]), start=1
):
    writer.add_scalar("Loss/train", train_loss, epoch)
    if val_loss is not None:
        writer.add_scalar("Loss/val", val_loss, epoch)

writer.close()

print("Training completed.")



rmse_train = root_mean_squared_error(y_pred_train, y_train)
mae_train  = mean_absolute_error(y_pred_train, y_train)
r2_train   = r2_score(y_pred_train, y_train)

rmse_test = root_mean_squared_error(y_pred_train, y_train)
mae_test  = mean_absolute_error(y_pred_train, y_train)
r2_test   = r2_score(y_pred_train, y_train)

print("\n=== Evaluation (rates per exposure unit) ===")
print(f"Train: RMSE={rmse_train:.6f}, MAE={mae_train:.6f}, R2={r2_train:.4f}")
print(f"Test : RMSE={rmse_test:.6f}, MAE={mae_test:.6f}, R2={r2_test:.4f}")
