import glob
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, resize

from src.constants import GT, MASK, KERNEL, MEASUREMENT, GAUSSIAN_MEASUREMENT, POISSON_MEASUREMENT
from src.constants import BOX_INPAINTING, RANDOM_INPAINTING, MOTION_BLUR


class ImageProcessor:
    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        return (x - 0.5) / 0.5

    @staticmethod
    def denormalize(x: torch.Tensor) -> torch.Tensor:
        return (x + 1) / 2


class ImageDataset(Dataset):
    def __init__(
        self,
        root: str = "data/imagenet/*",
        size: tuple = (256, 256),
        num_samples: int = -1,
    ) -> None:
        self.num_samples = num_samples
        self.size = size
        self.data_list = sorted(glob.glob(root))[:num_samples]
        self.processor = ImageProcessor()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        image_path = self.data_list[idx]

        # Image tensor with shape (C, H, W), values in [0, 1], resized, then
        # normalized to [-1, 1]
        image = to_tensor(Image.open(image_path).convert("RGB"))
        resized_image = resize(image, self.size)
        normalized_image = self.processor.normalize(resized_image)
        return {
            "filename": Path(image_path).stem,
            "resized_image": resized_image,
            "normalized_image": normalized_image,
        }


class InverseProblemDataset(Dataset):
    """Loads (ground truth, measurement) pairs produced by generate_dataset.py."""

    def __init__(
        self,
        task: str = "random_inpainting",
        root: str = "dataset/ffhq/random_inpainting/*",
        num_samples: int = -1,
    ):
        self.task = task
        self.num_samples = num_samples
        self.data_list = sorted(glob.glob(root))[:num_samples]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        folder = self.data_list[idx]
        gt = to_tensor(Image.open(Path(folder, f"{GT}.png")))
        measurement = torch.Tensor(np.load(Path(folder, f"{MEASUREMENT}.npy")))
        gaussian_measurement = torch.Tensor(np.load(Path(folder, f"{GAUSSIAN_MEASUREMENT}.npy")))
        poisson_measurement = torch.Tensor(np.load(Path(folder, f"{POISSON_MEASUREMENT}.npy")))

        if self.task in [BOX_INPAINTING, RANDOM_INPAINTING]:
            return {
                "filename": Path(folder).stem,
                GT: gt,
                MASK: to_tensor(Image.open(Path(folder, f"{MASK}.png"))),
                MEASUREMENT: measurement,
                GAUSSIAN_MEASUREMENT: gaussian_measurement,
                POISSON_MEASUREMENT: poisson_measurement,
            }

        elif self.task == MOTION_BLUR:
            return {
                "filename": Path(folder).stem,
                GT: gt,
                KERNEL: torch.Tensor(np.load(Path(folder, f"{KERNEL}.npy"))),
                MEASUREMENT: measurement,
                GAUSSIAN_MEASUREMENT: gaussian_measurement,
                POISSON_MEASUREMENT: poisson_measurement,
            }

        return {
            "filename": Path(folder).stem,
            GT: gt,
            MEASUREMENT: measurement,
            GAUSSIAN_MEASUREMENT: gaussian_measurement,
            POISSON_MEASUREMENT: poisson_measurement,
        }
