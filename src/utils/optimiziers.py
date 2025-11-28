import numpy as np

class SGD:
    def __init__(self, lr=1e-3):
        self.lr = lr
        
    def step(self, model):
        for layer in model.layers:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db
            
class SGDMomentum:
    def __init__(self, lr=1e-3, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.vel = {}

    def step(self, model):
        for layer in model.layers:
            key = id(layer)

            if key not in self.vel:
                self.vel[key] = {
                    "vW": np.zeros_like(layer.W),
                    "vB": np.zeros_like(layer.b),
                }

            vW = self.vel[key]["vW"]
            vB = self.vel[key]["vB"]

            vW_new = self.momentum * vW - self.lr * layer.dW
            vB_new = self.momentum * vB - self.lr * layer.db
            
            layer.W += vW_new
            layer.b += vB_new

            self.vel[key]["vW"] = vW_new
            self.vel[key]["vB"] = vB_new
            
class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.state = {}
        
    def step(self, model):
        self.t += 1
        for layer in model.layers:
            key = id(layer)
            if key not in self.state:
                self.state[key] = {
                    "mW": np.zeros_like(layer.W),
                    "vW": np.zeros_like(layer.W),
                    "mB": np.zeros_like(layer.b),
                    "vB": np.zeros_like(layer.b),
                }
            st = self.state[key]
            
            gW = layer.dW
            gB = layer.db
            
            st["mW"] = self.beta1 * st["mW"] + (1 - self.beta1) * gW
            st["vW"] = self.beta2 * st["vW"] + (1 - self.beta2) * (gW ** 2)
            st["mB"] = self.beta1 * st["mB"] + (1 - self.beta1) * gB
            st["vB"] = self.beta2 * st["vB"] + (1 - self.beta2) * (gB ** 2)
            
            mW_hat = st["mW"] / (1 - self.beta1 ** self.t)
            vW_hat = st["vW"] / (1 - self.beta2 ** self.t)
            mB_hat = st["mB"] / (1 - self.beta1 ** self.t)
            vB_hat = st["vB"] / (1 - self.beta2 ** self.t)
            
            layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * mB_hat / (np.sqrt(vB_hat) + self.eps)
            
class Adagrad:
    def __init__(self, lr=1e-2, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.state = {}

    def step(self, model):
        for layer in model.layers:
            key = id(layer)

            if key not in self.state:
                self.state[key] = {
                    "gW": np.zeros_like(layer.W),
                    "gB": np.zeros_like(layer.b),
                }

            gW = self.state[key]["gW"]
            gB = self.state[key]["gB"]
            
            gW += layer.dW ** 2
            gB += layer.db ** 2

            layer.W -= self.lr * layer.dW / (np.sqrt(gW) + self.eps)
            layer.b -= self.lr * layer.db / (np.sqrt(gB) + self.eps)

            self.state[key]["gW"] = gW
            self.state[key]["gB"] = gB