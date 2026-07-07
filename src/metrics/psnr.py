import numpy as np


def psnr(x: np.ndarray, y: np.ndarray) -> np.float64:
    assert x.shape == y.shape, "Input array must have the same shape"

    x = x / 255.0 if x.max() > 1.0 else x
    y = y / 255.0 if y.max() > 1.0 else y

    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return float('inf')  # No noise, perfect match
    return 20 * np.log10(mse ** -0.5)
