import numpy as np

def batch_iterator(X, y, batch_size, shuffle=True, seed=42):
    N = X.shape[0]
    idx = np.arange(N)
    rng = np.random.default_rng(seed) if shuffle else None
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, N, batch_size):
        end = start + batch_size
        b = idx[start:end]
        Xb = X[b]
        yb = y[b].reshape(-1, 1)
        yield Xb, yb