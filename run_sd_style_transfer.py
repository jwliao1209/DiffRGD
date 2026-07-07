"""Style-guided generation with Stable Diffusion (Appendix E.2 of the paper).

For each (style image, prompt) pair, generates an image whose CLIP Gram-matrix
statistics match the style image while following the text prompt.
"""

import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from diffusers import DDIMScheduler
from diffusers.utils import load_image

from src.constants import OUTPUT_DIR
from src.losses.style_loss import StyleLoss
from src.pipelines.pipeline_sd_image_sampler import StableDiffusionImageSamplerPipeline
from src.utils import set_random_seed


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m", "--method",
        type=str,
        default="diffrgd",
        choices=["dps", "mpgd", "dsg", "diffrgd"],
    )
    parser.add_argument("--step_size", type=float, default=2.5, help="Guidance strength eta_t.")
    parser.add_argument("--num_inference_steps", type=int, default=150)
    parser.add_argument("-i", "--interval", type=int, default=1, help="Apply guidance every N steps.")
    parser.add_argument("-iter", "--inner_max_iter", type=int, default=3, help="Inner iterations K.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--prompt_file", type=str, default="configs/prompts.json")
    parser.add_argument(
        "--style_images",
        type=str,
        nargs="+",
        default=[f"style_images/{i}.png" for i in range(1, 6)],
    )
    return parser.parse_args()


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()

    device = "cuda"
    generator = torch.Generator(device).manual_seed(1024)

    folder = Path(OUTPUT_DIR, "sd_style_transfer", args.method)
    os.makedirs(folder, exist_ok=True)

    prompts = read_json(args.prompt_file)["prompts"]

    pipe = StableDiffusionImageSamplerPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    for module in [pipe.unet, pipe.vae, pipe.text_encoder]:
        for param in module.parameters():
            param.requires_grad = False

    loss_fn = StyleLoss(device=device)

    for style_image_path in args.style_images:
        reference_image = load_image(style_image_path)
        for prompt in prompts:
            image = pipe(
                method=args.method,
                prompt=prompt["prompt"],
                height=512,
                width=512,
                eta=1,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                step_size=args.step_size,
                inner_max_iter=args.inner_max_iter,
                reference_image=reference_image,
                guidance_interval=args.interval,
                loss_fn=loss_fn,
                generator=generator,
            ).images[0]
            image.save(Path(folder, f"style_{Path(style_image_path).stem}_prompt_{prompt['id']}.png"))
