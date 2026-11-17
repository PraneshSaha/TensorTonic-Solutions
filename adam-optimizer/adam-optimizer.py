import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    grad = np.array(grad)
    param = np.array(param)
    m_new = beta1 * np.array(m) + (1-beta1) * grad
    v_new = beta2 * np.array(v) + (1-beta2) * grad**2
    m_new_scaled = m_new/(1 - beta1**t)
    v_new_scaled = v_new/(1 - beta2**t)
    
    param_new = param - lr * m_new_scaled/(np.sqrt(v_new_scaled) + eps)

    return param_new, m_new, v_new