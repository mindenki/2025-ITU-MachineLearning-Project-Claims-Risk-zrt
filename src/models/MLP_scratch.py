from math import sqrt
import numpy as np

def ReLU(x):
    return max(0, x)

def dReLU(x):
    return (x>0.0).astype(x.dtype)
    
class Layer:
    def __init__(self, in_feats, out_feats, activation, dactivation):
        # He parameter initialization
        rng = np.random.default_rng(42)
        std = sqrt(2.0/len(in_feats))
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
        self.y = np.matmul(x, self.W) + self.b
        return self.act(self.y)
    
    def backward(self, grad_out):
        
        # derivative of loss wrt the activated output (post activation)
        D = grad_out * self.dact(self.z)
        
        # derivative of loss wrt the weights
        # summing over the batch
        self.dW += np.matmul(D.T, self.x)
        
        # derivative of loss wrt the bias
        # summing over the batch
        self.b += D.sum(axis=0)
        
        # derivative of loss wrt the inputs
        grad_in = np.matmul(D, self.W)
        
        return grad_in
        
    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
        
    def zero_grad(self):
        # reseting the gradients for weigths and biases after batch
        self.dW.fill(0.0)
        self.dn.fill(0.0)
    
class MLP:
    def __init__(self, input_dim, hidden_sizes, lr):
        self.lr = lr
        output_dim = 1
        sizes = [input_dim] + len(hidden_sizes) + [output_dim]
        activations = [(ReLU(), dReLU())] * len(hidden_sizes) + [(lambda x:x, lambda x:np.ones_like(x))]
        self.layers = []
        for i in range(len(hidden_sizes)):
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
            
    def step(self):
        for layer in self.layers:
            layer.step(self.lr)
        
        
def MSE(preds, targets):
    n = preds.shape[0]
    diff = preds - targets
    loss = (diff **2)/n
    grad = (2.0 * diff)/n
    return loss, grad

# Example of one pass
# model = MLP(...)
# preds = model.forward(X)
# loss, dyp = MSE(preds, y)
# model.zero_grad()
# model.backward(dyp)
# model.step()