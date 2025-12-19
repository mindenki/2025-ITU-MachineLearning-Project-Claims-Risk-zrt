import torch
import torch.nn as nn
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            X = getattr(X, "to_numpy", lambda: X)()
            self.X = torch.as_tensor(X, dtype=torch.float32)
            y = getattr(y, "to_numpy", lambda: y)()
            self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
  
class MLP(nn.Module):
    def __init__(self, input_size=9, hidden_sizes=[64, 32]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
            nn.Softplus()
        )
    def forward(self, x):
        return self.layers(x)
    

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_model(model, X_train, y_train, X_val, y_val, lr=1e-3, batch_size=256, epochs=50, return_epoch_history=False):
    # Dataset and DataLoader
    train_dataset = Dataset(X_train, y_train)
    if X_val is not None and y_val is not None:
        val_dataset = Dataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if X_val is not None and y_val is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_rmse_list = []
    val_rmse_list = []
    
    # Training
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_targets = []
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1)) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            all_train_preds.append(y_pred.detach().numpy())
            all_train_targets.append(y_batch.numpy())
        
        train_preds_epoch = np.concatenate(all_train_preds).flatten()
        train_targets_epoch = np.concatenate(all_train_targets).flatten()
        train_rmse_epoch = np.sqrt(np.mean((train_targets_epoch - train_preds_epoch) ** 2))
        train_rmse_list.append(train_rmse_epoch)
        
        train_loss /= len(train_loader.dataset)
        if X_val is not None and y_val is not None:
            # Validation
            model.eval()
            all_val_preds = []
            all_val_targets = []
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch.unsqueeze(1))
                    val_loss += loss.item() * X_batch.size(0)
                    all_val_preds.append(y_pred.numpy())
                    all_val_targets.append(y_batch.numpy())
                    
            val_loss /= len(val_loader.dataset)
            val_preds_epoch = np.concatenate(all_val_preds).flatten()
            val_targets_epoch = np.concatenate(all_val_targets).flatten()
            val_rmse_epoch = np.sqrt(np.mean((val_targets_epoch - val_preds_epoch) ** 2))
            val_rmse_list.append(val_rmse_epoch)
        else:
            val_loss = None
            val_rmse_epoch = None
        if return_epoch_history:
            return model, train_rmse_list, val_rmse_list
        else:   
            return model, train_rmse_list[-1], val_rmse_list[-1] if val_rmse_list else None
    

