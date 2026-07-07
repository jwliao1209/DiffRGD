"""Convert the OpenAI guided-diffusion checkpoints (FFHQ from DPS, ImageNet from
Dhariwal & Nichol) into Diffusers pipelines under checkpoints/openai_<dataset>.

Download the raw checkpoints with scripts/download_model_checkpoint.sh first.
"""

import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import DDIMPipeline, DDIMScheduler, DDPMPipeline, DDPMScheduler

# Allow running this script directly from anywhere (src/ lives at the repo root)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import get_diffusion


@dataclass
class Config:
    model_name_or_path: str
    sample_size: int = 256
    device: str = "cuda"
    train_steps: int = 1000
    inference_steps: int = 50


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset", type=str, default="ffhq", choices=["ffhq", "imagenet"])
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the raw .pt checkpoint; defaults to checkpoints/temp/<dataset default>.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    default_checkpoints = {
        "ffhq": "checkpoints/temp/ffhq_10m.pt",
        "imagenet": "checkpoints/temp/imagenet256.pt",
    }
    config = Config(model_name_or_path=args.checkpoint or default_checkpoints[args.dataset])

    model, ts, alpha_prod_ts, alpha_prod_t_prevs = get_diffusion(config)
    model = model.to(torch.float16)

    # Sanity-check a forward pass
    x = torch.randn(1, 3, config.sample_size, config.sample_size).to(config.device).to(torch.float16)
    timesteps = torch.tensor([1]).to(config.device).to(torch.float16)
    output = model(x, timesteps)
    print(f"forward pass output: shape={tuple(output.sample.shape)}, "
          f"range=[{output.sample.min():.3f}, {output.sample.max():.3f}]")

    # Reuse the standard CelebA-HQ DDPM schedule configuration
    scheduler_config = dict(DDIMPipeline.from_pretrained("google/ddpm-ema-celebahq-256").scheduler.config)

    if args.dataset == "imagenet":
        # The ImageNet model predicts the variance as well
        scheduler_config["variance_type"] = "learned_range"
        pipe = DDPMPipeline(unet=model, scheduler=DDPMScheduler.from_config(scheduler_config))
    else:
        pipe = DDIMPipeline(unet=model, scheduler=DDIMScheduler.from_config(scheduler_config))

    output_dir = f"checkpoints/openai_{args.dataset}"
    pipe.save_pretrained(output_dir)
    print(f"saved converted pipeline to {output_dir}")

    # Verify the saved pipeline generates an image
    pipe = type(pipe).from_pretrained(output_dir)
    pipe.to("cuda")
    pipe().images[0].save("sanity_check.png")
    print("generated sanity_check.png from the converted pipeline")
