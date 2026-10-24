import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    dimensions = np.arange(d_model//2 if d_model%2==0 else (d_model+1)//2, dtype=float).repeat(2)[:d_model].reshape(1, -1)
    positions = np.arange(seq_len).reshape(-1, 1)
    positional_enc = np.zeros((seq_len, d_model), dtype=float)
    positional_enc[:, ::2] = np.sin(positions/np.power(base, (2*dimensions[:, ::2]/d_model)))
    positional_enc[:, 1::2] = np.cos(positions/np.power(base, (2*dimensions[:, 1::2]/d_model)))

    return positional_enc