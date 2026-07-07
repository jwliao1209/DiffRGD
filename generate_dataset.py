"""Generate measurement datasets for the inverse problem experiments.

For each image under data/<dataset>/, applies the task's degradation operator and
saves the ground truth, the clean measurement, and Gaussian / Poisson noisy
measurements under dataset/<dataset>/<task>/<filename>/.
"""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from src.constants import (
    DENOISING,
    BOX_INPAINTING,
    RANDOM_INPAINTING,
    SUPER_RESOLUTION,
    GAUSSIAN_BLUR,
    MOTION_BLUR,
    GT,
    MEASUREMENT,
    MASK,
    KERNEL,
    GAUSSIAN_MEASUREMENT,
    POISSON_MEASUREMENT,
)
from src.dataset import ImageDataset, ImageProcessor
from src.noise import GaussianNoise, PoissonNoise
from src.operators import (
    DenoisingOperator,
    BoxInpaintingOperator,
    RandomInpaintingOperator,
    SuperResolutionOperator,
    GaussianBlurOperator,
    MotionBlurOperator,
)
from src.utils import set_random_seed


OPERATORS = {
    DENOISING: DenoisingOperator,
    BOX_INPAINTING: BoxInpaintingOperator,
    RANDOM_INPAINTING: RandomInpaintingOperator,
    SUPER_RESOLUTION: SuperResolutionOperator,
    GAUSSIAN_BLUR: GaussianBlurOperator,
    MOTION_BLUR: MotionBlurOperator,
}


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-t", "--task", type=str, default="random_inpainting", choices=list(OPERATORS))
    parser.add_argument("-d", "--dataset", type=str, default="ffhq")
    parser.add_argument("-n", "--num_samples", type=int, default=100)
    parser.add_argument("--gaussian_noise_level", type=float, default=0.05)
    return parser.parse_args()


def save_image(image, path):
    image = (image.permute(0, 2, 3, 1).squeeze() * 255).clamp(0, 255).cpu().numpy().astype("uint8")
    if image.ndim == 3:
        Image.fromarray(image, "RGB").save(path)
    else:
        Image.fromarray(image, "L").save(path)


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()
    config = OmegaConf.load(f"configs/operators/{args.task}.yaml")
    operator = OPERATORS[args.task](**config.operator)

    dataset = ImageDataset(
        root=f"data/{args.dataset}/*",
        num_samples=args.num_samples,
        size=config.operator.get("image_size", (256, 256)),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    gaussian_noise = GaussianNoise(sigma=args.gaussian_noise_level)
    poisson_noise = PoissonNoise()
    processor = ImageProcessor

    for data in tqdm(dataloader):
        measurement = operator(data["normalized_image"])
        gaussian_measurement = gaussian_noise(measurement)
        poisson_measurement = poisson_noise(measurement)

        folder = Path("dataset", args.dataset, args.task, data["filename"][0])
        os.makedirs(folder, exist_ok=True)

        save_image(data["resized_image"], Path(folder, f"{GT}.png"))
        save_image(processor.denormalize(measurement), Path(folder, f"{MEASUREMENT}.png"))
        save_image(processor.denormalize(gaussian_measurement), Path(folder, f"{GAUSSIAN_MEASUREMENT}.png"))
        save_image(processor.denormalize(poisson_measurement), Path(folder, f"{POISSON_MEASUREMENT}.png"))

        np.save(Path(folder, f"{MEASUREMENT}.npy"), measurement[0])
        np.save(Path(folder, f"{GAUSSIAN_MEASUREMENT}.npy"), gaussian_measurement[0])
        np.save(Path(folder, f"{POISSON_MEASUREMENT}.npy"), poisson_measurement[0])

        if args.task in [BOX_INPAINTING, RANDOM_INPAINTING]:
            save_image(operator.mask, Path(folder, f"{MASK}.png"))
        if args.task == MOTION_BLUR:
            np.save(Path(folder, f"{KERNEL}.npy"), operator.kernel[0])

        # Resample the mask / blur kernel so each image gets its own degradation
        operator.reset_operator()
