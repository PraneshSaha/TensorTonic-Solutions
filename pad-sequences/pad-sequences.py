import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    max_len = max_len if max_len else (max(len(seq) for seq in seqs) if seqs else 0)
    padded_seq = np.zeros((len(seqs), max_len), dtype=int)
    padded_seq[:, :] = pad_value

    for row_num in range(len(seqs)):
        padded_seq[row_num, :min(max_len, len(seqs[row_num]))] = seqs[row_num][:max_len]

    return padded_seq