"""Evaluate style-guided generation with CLIP-score / Style-score (Table 10)."""

import glob
import json
import sys
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running this script directly from anywhere (src/ lives at the repo root)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.metrics import cal_clip_aes, cal_clip_score, cal_style_score

warnings.filterwarnings("ignore", message="Using a slow image processor.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics.functional.image.lpips")


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--method", type=str, default="diffrgd")
    parser.add_argument("--prompt_file", type=str, default="configs/prompts.json")
    parser.add_argument("--display", type=str, default="plain", choices=["plain", "latex"])
    return parser.parse_args()


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    args = parse_arguments()
    data_list = sorted(glob.glob(f"outputs/sd_style_transfer/{args.method}/*.png"))

    if not data_list:
        print("No data found in the specified directory.")
        exit()

    prompts = {str(p["id"]): p["prompt"] for p in read_json(args.prompt_file)["prompts"]}

    gen_list = []
    text_list = []
    style_list = []
    for path in data_list:
        # File names follow style_<style_id>_prompt_<prompt_id>.png
        parts = Path(path).stem.split("_")
        style_id, prompt_id = parts[1], parts[3]
        gen_list.append(np.array(Image.open(path)))
        text_list.append(prompts[prompt_id])
        style_list.append(np.array(Image.open(f"style_images/{style_id}.png").resize((512, 512)).convert("RGB")))

    clip_aes = cal_clip_aes(gen_list, device="cuda")
    clip_score = cal_clip_score(gen_list, text_list, device="cuda")
    style_score = cal_style_score(gen_list, style_list, device="cuda")

    print("=" * 80)
    print(f"Method: {args.method} (N={len(gen_list)})")
    print(f"{'CLIP-aes':<20}{'CLIP-score':<20}{'Style-score':<20}")
    print(f"{f'{clip_aes:.3f}':<20}{f'{clip_score:.3f}':<20}{f'{style_score:.3f}':<20}")
    print("=" * 80)
