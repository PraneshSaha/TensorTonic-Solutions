import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    x = np.array(x)
    dropout_mask =  rng.random(x.shape) if rng else np.random.random(x.shape)
    dropout_mask = (dropout_mask > p).astype(int)/(1-p)
    output = x*dropout_mask
    
    return output, dropout_mask
    