import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    print(X.shape)
    W = np.zeros(X.shape[1], dtype=float)
    print(W.shape)
    b = 0.0
    for step in range(steps):
        z = X @ W + b
        p = _sigmoid(z)
        W = W - lr * (X.T @ (np.squeeze(p) - y)/y.shape[0])
        b = b - lr * np.sum((p-y))/y.shape[0]

    return W, b
    