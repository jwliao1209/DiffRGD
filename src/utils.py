import random
from typing import List

import imageio.v2 as imageio
import numpy as np
import torch


def set_random_seed(random_seed: int = 0, deterministic: bool = True) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def save_video(x: List[np.ndarray], path: str) -> None:
    imageio.mimsave(path, x, fps=30, codec='libx264')
