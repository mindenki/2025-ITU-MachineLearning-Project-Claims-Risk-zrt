import torch
import torch.nn as nn


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
    def __init__(self, input_size=9):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
        )


    def forward(self, x):
        return self.layers(x)