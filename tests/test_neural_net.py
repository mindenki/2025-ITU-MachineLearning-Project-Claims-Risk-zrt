import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from src.data.load_data import load_data
from src.data.preprocess import preprocess, PCA
from src.models.MLP_torch import MLP, Dataset

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

train_df, test_df = load_data(raw=False)
y_train = train_df["ClaimRate"]
X_train = train_df.drop(columns=["ClaimRate"])
X_test  = test_df.drop(columns=["ClaimRate"])
y_test  = test_df["ClaimRate"]

y_scaler   = StandardScaler().fit(y_train.to_numpy().reshape(-1, 1))
y_train_std = y_scaler.transform(y_train.to_numpy().reshape(-1, 1)).ravel()
y_test_std  = y_scaler.transform(y_test.to_numpy().reshape(-1, 1)).ravel()

pre = preprocess(X_train)
X_train_pre = pre.fit_transform(X_train)
X_test_pre  = pre.transform(X_test)

pca = PCA(X_train_pre.shape[1]-1, seed=seed)

X_train_svd = pca.fit_transform(X_train_pre).astype("float32")
X_test_svd  = pca.transform(X_test_pre).astype("float32")

x_scaler = StandardScaler().fit(X_train_svd)
X_train_reduced = x_scaler.transform(X_train_svd).astype("float32")
X_test_reduced  = x_scaler.transform(X_test_svd).astype("float32")

train_ds = Dataset(X_train_reduced, y_train_std)
test_ds  = Dataset(X_test_reduced,  y_test_std)

model = MLP(input_size=X_train_reduced.shape[1])

epochs = 100
batch_size = 32
lr = 1e-3

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

for epoch in range(epochs):
    model.train()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    running = 0.0

    for inputs, targets in train_loader:
        inputs  = inputs.float()
        targets = targets.float().view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()

        running += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running/len(train_loader):.4f}")

print("Training complete.")

@torch.no_grad()
def evaluate(model, test_ds, batch_size=256, device="cpu"):
    model.eval().to(device)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_y_std = []
    all_p_std = []

    for X, y_std in loader:
        X = X.float().to(device)
        y_std = y_std.float().view(-1, 1).to(device)
        p_std = model(X)

        all_y_std.append(y_std.cpu())
        all_p_std.append(p_std.cpu())

    y_std = torch.cat(all_y_std, dim=0).numpy()
    p_std = torch.cat(all_p_std, dim=0).numpy()

    y = y_scaler.inverse_transform(y_std)
    p = y_scaler.inverse_transform(p_std)

    diff = p - y
    mse  = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(diff)))
    y_mean = float(np.mean(y))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else float("nan"))

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

metrics = evaluate(model, test_ds, batch_size=256, device="cpu")
print(metrics)