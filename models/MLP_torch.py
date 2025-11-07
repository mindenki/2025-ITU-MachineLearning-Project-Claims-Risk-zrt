import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, scale_data=True):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

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
    
    
if __name__ == '__main__':

    torch.manual_seed(42)

    # Load dataset
    # TO BE IMPLEMENTED
    X, y = load_dataset(...)

    dataset = Dataset(X, y)
    trainloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)

    mlp = MLP(input_size=X.shape[1])

    loss_function = nn.L1Loss() # Need to choose a right loss function, just an example here
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001) # Try different optimizers and tune lr
    epochs = 100 # tune it later

    for epoch in range(epochs):
        current_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            
            optimizer.zero_grad()
            
            outputs = mlp(inputs)
            
            loss = loss_function(outputs, targets)
            
            loss.backward()
            
            optimizer.step()
            
            current_loss += loss.item()
            if i % 32 == 0:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 32))
                current_loss = 0.0
    print("Training finished.")