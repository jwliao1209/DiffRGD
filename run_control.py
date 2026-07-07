"""Conditional generation experiments (Section 4.3 of the paper).

Generates CelebA-HQ face images guided by a segmentation map, a sketch, or a
FaceID embedding extracted from a reference image. For the "dps" method with
these energy-based losses, the update rule is equivalent to FreeDoM.
"""

import gc
import glob
import os
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from diffusers.utils import load_image
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.constants import (
    OUTPUT_DIR,
    GENERATION, REFERENCE,
    REF_COND, GEN_COND,
    REF_COLOR_COND, GEN_COLOR_COND,
    REF_OVERLAY, GEN_OVERLAY,
)
from src.losses.face_id_loss import FaceIDLoss
from src.losses.face_seg_loss import FaceSegmentationLoss, colorize_mask, visualize_segmentation
from src.losses.face_sketch_loss import FaceSketchLoss
from src.pipelines.pipeline_image_sampler import ImageSamplerPipeline
from src.pipelines.pipeline_sd_image_sampler import StableDiffusionImageSamplerPipeline
from src.utils import set_random_seed


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-t", "--task",
        type=str,
        default="segmentation",
        choices=["segmentation", "sketch", "face_id"],
    )
    parser.add_argument(
        "-m", "--method",
        type=str,
        default="diffrgd",
        choices=["dps", "mpgd", "dsg", "admmdiff", "diffrgd"],
    )
    parser.add_argument("-d", "--dataset", type=str, default="celeba_hq")
    parser.add_argument(
        "--model",
        type=str,
        default="pdm",
        choices=["pdm", "ldm"],
        help="pdm: CelebA-HQ DDPM; ldm: Stable Diffusion v1.5.",
    )
    parser.add_argument("-n", "--num_samples", type=int, default=100)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--step_size", type=float, default=50, help="Guidance strength eta_t.")
    parser.add_argument("--eta", type=float, default=1.0, help="DDIM eta (stochasticity).")
    parser.add_argument("-i", "--interval", type=int, default=1, help="Apply guidance every N steps.")
    parser.add_argument("-iter", "--inner_max_iter", type=int, default=1, help="Inner iterations K.")
    parser.add_argument("--rho", type=float, default=1.0, help="ADMM penalty parameter (admmdiff only).")
    parser.add_argument("--prompt", type=str, default="A natural looking human face", help="ldm only.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def save_array_as_image(image: np.ndarray, path: Path) -> None:
    Image.fromarray(image).save(path)


def save_conditions(task: str, loss_fn, reference_image, generated_image, folder: Path, device: str) -> None:
    """Extract and save the task condition (parsing map / sketch / ID embedding)
    of both the reference and the generated image for later evaluation."""
    ref_tensor = to_tensor(reference_image).unsqueeze(0).to(device)
    gen_tensor = to_tensor(generated_image).unsqueeze(0).to(device)

    if task == "segmentation":
        ref_cond = loss_fn.segmentor(loss_fn.image_processor(ref_tensor))[0].argmax(dim=1).squeeze().cpu().numpy()
        gen_cond = loss_fn.segmentor(loss_fn.image_processor(gen_tensor))[0].argmax(dim=1).squeeze().cpu().numpy()
        save_array_as_image(ref_cond.astype(np.uint8), Path(folder, f"{REF_COND}.png"))
        save_array_as_image(gen_cond.astype(np.uint8), Path(folder, f"{GEN_COND}.png"))
        save_array_as_image(colorize_mask(ref_cond.astype(np.uint8)), Path(folder, f"{REF_COLOR_COND}.png"))
        save_array_as_image(colorize_mask(gen_cond.astype(np.uint8)), Path(folder, f"{GEN_COLOR_COND}.png"))
        visualize_segmentation(ref_cond, reference_image).save(Path(folder, f"{REF_OVERLAY}.png"))
        visualize_segmentation(gen_cond, generated_image).save(Path(folder, f"{GEN_OVERLAY}.png"))

    elif task == "sketch":
        ref_cond = loss_fn.get_condition(ref_tensor).squeeze().cpu().numpy() * 255
        gen_cond = loss_fn.get_condition(gen_tensor).squeeze().cpu().numpy() * 255
        save_array_as_image(ref_cond.astype(np.uint8), Path(folder, f"{REF_COND}.png"))
        save_array_as_image(gen_cond.astype(np.uint8), Path(folder, f"{GEN_COND}.png"))

    elif task == "face_id":
        ref_cond = loss_fn.get_condition(ref_tensor).squeeze().cpu().numpy()
        gen_cond = loss_fn.get_condition(gen_tensor).squeeze().cpu().numpy()
        np.save(Path(folder, f"{REF_COND}.npy"), ref_cond)
        np.save(Path(folder, f"{GEN_COND}.npy"), gen_cond)


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()

    device = "cuda"
    generator = torch.Generator(device).manual_seed(1024)

    if args.model == "pdm":
        pipe = ImageSamplerPipeline.from_pretrained("google/ddpm-ema-celebahq-256")
        pipe = pipe.to(device)
        for param in pipe.unet.parameters():
            param.requires_grad = False
    else:
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

    loss_fn = {
        "segmentation": FaceSegmentationLoss,
        "sketch": FaceSketchLoss,
        "face_id": FaceIDLoss,
    }[args.task](device=device)

    for image_path in sorted(glob.glob(f"data/{args.dataset}/*.jpg"))[:args.num_samples]:
        reference_image = load_image(image_path)
        folder = Path(OUTPUT_DIR, "condition_gen", args.dataset, args.task, args.method, Path(image_path).stem)
        os.makedirs(folder, exist_ok=True)

        start_time = time.time()
        if args.model == "pdm":
            image = pipe(
                method=args.method,
                reference_image=reference_image,
                step_size=args.step_size,
                num_inference_steps=args.num_inference_steps,
                guidance_interval=args.interval,
                inner_max_iter=args.inner_max_iter,
                rho=args.rho,
                loss_fn=loss_fn,
                eta=args.eta,
                verbose=args.verbose,
            ).images[0]
        else:
            image = pipe(
                method=args.method,
                reference_image=reference_image,
                prompt=args.prompt,
                height=512,
                width=512,
                step_size=args.step_size,
                num_inference_steps=args.num_inference_steps,
                guidance_interval=args.interval,
                inner_max_iter=args.inner_max_iter,
                loss_fn=loss_fn,
                eta=args.eta,
                generator=generator,
                verbose=args.verbose,
            ).images[0]
        elapsed_time = time.time() - start_time

        image.save(Path(folder, f"{GENERATION}.png"))
        reference_image.save(Path(folder, f"{REFERENCE}.png"))

        with torch.inference_mode():
            save_conditions(args.task, loss_fn, reference_image, image, folder, device)

        del reference_image, image
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Task: {args.task}, Method: {args.method}, Time: {elapsed_time:.2f} seconds")
