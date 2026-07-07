import numpy as np
from scipy.ndimage import gaussian_filter


def ssim(x: np.ndarray, y: np.ndarray, sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03) -> np.float64:
    assert x.shape == y.shape, "Input array must have the same shape"
    x = x / 255.0 if x.max() > 1.0 else x
    y = y / 255.0 if y.max() > 1.0 else y

    C1 = k1 ** 2
    C2 = k2 ** 2

    mu1 = np.stack([gaussian_filter(x[..., i], sigma) for i in range(3)], axis=-1)
    mu2 = np.stack([gaussian_filter(y[..., i], sigma) for i in range(3)], axis=-1)

    mu1_squared = mu1 ** 2
    mu2_squared = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_squared = np.stack([gaussian_filter(x[..., i] ** 2, sigma) for i in range(3)], axis=-1) - mu1_squared
    sigma2_squared = np.stack([gaussian_filter(y[..., i] ** 2, sigma) for i in range(3)], axis=-1) - mu2_squared
    sigma12   = np.stack([gaussian_filter(x[..., i] * y[..., i], sigma) for i in range(3)], axis=-1) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_squared + mu2_squared + C1) * (sigma1_squared + sigma2_squared + C2)
    return (numerator / denominator).mean()
