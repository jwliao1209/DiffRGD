from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
from PIL import Image
from diffusers.utils import BaseOutput


@dataclass
class ImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    images: Union[List[Image.Image], np.ndarray]
    xt_traj: Optional[List[np.ndarray]] = None
    x0t_traj: Optional[List[np.ndarray]] = None
