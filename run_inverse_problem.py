"""Image restoration experiments (Section 4.1 and 4.2 of the paper).

Solves noisy linear inverse problems (inpainting, super-resolution, deblurring,
denoising) with inference-time diffusion guidance. Supports pixel-space diffusion
models (--model pdm) and Stable Diffusion (--model ldm).
"""

import os
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import (
    DENOISING,
    BOX_INPAINTING,
    RANDOM_INPAINTING,
    SUPER_RESOLUTION,
    GAUSSIAN_BLUR,
    MOTION_BLUR,
    GT,
    RECONSTRUCTION,
    MEASUREMENT,
    GAUSSIAN_MEASUREMENT,
    POISSON_MEASUREMENT,
    MASK,
    KERNEL,
    OUTPUT_DIR,
)
from src.dataset import InverseProblemDataset, ImageProcessor
from src.losses.linear_inverse_problem import LinearInverseProblemLoss
from src.operators import (
    DenoisingOperator,
    BoxInpaintingOperator,
    RandomInpaintingOperator,
    SuperResolutionOperator,
    GaussianBlurOperator,
    MotionBlurOperator,
)
from src.pipelines.pipeline_image_sampler import ImageSamplerPipeline
from src.pipelines.pipeline_sd_image_sampler import StableDiffusionImageSamplerPipeline
from src.utils import set_random_seed, save_video
from generate_dataset import save_image


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-t", "--task",
        type=str,
        default="random_inpainting",
        help="Restoration task. One of box_inpainting, random_inpainting, super_resolution, "
             "gaussian_blur, motion_blur, or denoising_<sigma> (e.g. denoising_0.1).",
    )
    parser.add_argument(
        "-m", "--method",
        type=str,
        default="diffrgd",
        choices=["dps", "mpgd", "dsg", "admmdiff", "diffrgd"],
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="ffhq",
        choices=["ffhq", "imagenet", "celeba_hq"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pdm",
        choices=["pdm", "ldm"],
        help="pdm: pixel-space diffusion model; ldm: Stable Diffusion v1.5.",
    )
    parser.add_argument("-n", "--num_samples", type=int, default=-1, help="-1 uses all samples.")
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--step_size", type=float, default=1.0, help="Guidance strength eta_t.")
    parser.add_argument("--eta", type=float, default=1.0, help="DDIM eta (stochasticity).")
    parser.add_argument(
        "--noise",
        type=str,
        default="gaussian",
        choices=["none", "gaussian", "poisson"],
        help="Measurement noise type of the inverse problem.",
    )
    parser.add_argument("-i", "--interval", type=int, default=1, help="Apply guidance every N steps.")
    parser.add_argument("-iter", "--inner_max_iter", type=int, default=1, help="Inner iterations K.")
    parser.add_argument("--rho", type=float, default=1.0, help="ADMM penalty parameter (admmdiff only).")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt (ldm only).")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Output resolution for Stable Diffusion, e.g. 512 for 8x or 768 for 12x "
             "super-resolution from 64x64 measurements (ldm only).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def build_operator(task: str, data: dict, device: str) -> torch.nn.Module:
    """Instantiate the degradation operator matching the measurement of this sample."""
    if task.startswith(DENOISING):
        return DenoisingOperator().to(device).to(torch.float16)
    if task == BOX_INPAINTING:
        return BoxInpaintingOperator(mask=data[MASK], device=device).to(torch.float16)
    if task == RANDOM_INPAINTING:
        return RandomInpaintingOperator(mask=data[MASK], device=device).to(torch.float16)
    if task == SUPER_RESOLUTION:
        return SuperResolutionOperator(size=(64, 64)).to(torch.float16)
    if task == GAUSSIAN_BLUR:
        return GaussianBlurOperator().to(device).to(torch.float16)
    if task == MOTION_BLUR:
        return MotionBlurOperator(kernel=data[KERNEL]).to(device).to(torch.float16)
    raise ValueError(f"Unknown inverse problem: {task}")


def load_pixel_diffusion_pipeline(dataset: str, device: str) -> ImageSamplerPipeline:
    if dataset == "celeba_hq":
        model_name = "google/ddpm-ema-celebahq-256"
    else:
        model_name = f"checkpoints/openai_{dataset}"
    pipe = ImageSamplerPipeline.from_pretrained(model_name)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe.to(device).to(torch.float16)


def load_stable_diffusion_pipeline(device: str) -> StableDiffusionImageSamplerPipeline:
    pipe = StableDiffusionImageSamplerPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device).to(torch.float16)
    for module in [pipe.unet, pipe.vae, pipe.text_encoder]:
        for param in module.parameters():
            param.requires_grad = False
    return pipe


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()

    device = "cuda"
    generator = torch.Generator(device).manual_seed(1024)

    processor = ImageProcessor()
    dataset = InverseProblemDataset(
        task=args.task,
        root=f"dataset/{args.dataset}/{args.task}/*",
        num_samples=args.num_samples,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    if args.model == "pdm":
        pipe = load_pixel_diffusion_pipeline(args.dataset, device)
    else:
        pipe = load_stable_diffusion_pipeline(device)

    measurement_key = {
        "none": MEASUREMENT,
        "gaussian": GAUSSIAN_MEASUREMENT,
        "poisson": POISSON_MEASUREMENT,
    }[args.noise]

    for data in tqdm(dataloader):
        operator = build_operator(args.task, data, device)
        loss_fn = LinearInverseProblemLoss(operator)
        measurement = data[measurement_key].to(device)

        start_time = time.time()
        if args.model == "pdm":
            outputs = pipe(
                method=args.method,
                reference_image=measurement,
                step_size=args.step_size,
                num_inference_steps=args.num_inference_steps,
                guidance_interval=args.interval,
                inner_max_iter=args.inner_max_iter,
                rho=args.rho,
                loss_fn=loss_fn,
                eta=args.eta,
                generator=generator,
                verbose=args.verbose,
            )
        else:
            outputs = pipe(
                method=args.method,
                reference_image=measurement,
                prompt=args.prompt,
                height=args.resolution,
                width=args.resolution,
                step_size=args.step_size,
                num_inference_steps=args.num_inference_steps,
                guidance_interval=args.interval,
                inner_max_iter=args.inner_max_iter,
                loss_fn=loss_fn,
                eta=args.eta,
                generator=generator,
                verbose=args.verbose,
            )
        elapsed_time = time.time() - start_time

        experiment = "image_restoration" if args.model == "pdm" else "sd_restoration"
        folder = Path(OUTPUT_DIR, experiment, args.dataset, args.task, args.method, data["filename"][0])
        os.makedirs(folder, exist_ok=True)
        save_image(data[GT], Path(folder, f"{GT}.png"))
        save_image(processor.denormalize(data[MEASUREMENT]), Path(folder, f"{MEASUREMENT}.png"))
        outputs.images[0].save(Path(folder, f"{RECONSTRUCTION}.png"))

        if args.task in [BOX_INPAINTING, RANDOM_INPAINTING]:
            save_image(operator.mask, Path(folder, f"{MASK}.png"))

        if getattr(outputs, "xt_traj", None):
            save_video(outputs.xt_traj, Path(folder, "xt_traj.mp4"))
        if getattr(outputs, "x0t_traj", None):
            save_video(outputs.x0t_traj, Path(folder, "x0t_traj.mp4"))

        print(f"Task: {args.task}, Method: {args.method}, Time: {elapsed_time:.2f} seconds")
