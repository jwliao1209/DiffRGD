"""Evaluate conditional generation results (Table 5).

Metrics: FID, KID, CLIP aesthetic score, and a task-specific condition metric
(mIoU for segmentation; L2 distance of extracted conditions for sketch / FaceID).
"""

import glob
import sys
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running this script directly from anywhere (src/ lives at the repo root)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.constants import GENERATION, REFERENCE, REF_COND, GEN_COND
from src.metrics import cal_iou, cal_fid, cal_kid, cal_clip_aes, cal_l2_norm

warnings.filterwarnings("ignore", message="Using a slow image processor.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics.functional.image.lpips")


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-t", "--task",
        type=str,
        default="segmentation",
        choices=["segmentation", "sketch", "face_id"],
    )
    parser.add_argument("-m", "--method", type=str, default="diffrgd")
    parser.add_argument("-d", "--dataset", type=str, default="celeba_hq")
    parser.add_argument("-n", "--num", type=int, default=100)
    parser.add_argument("--display", type=str, default="plain", choices=["plain", "latex"])
    return parser.parse_args()


def load_results(data_list, task):
    ref_list, gen_list, ref_cond_list, gen_cond_list = [], [], [], []
    for folder in data_list:
        folder = Path(folder)
        ref_list.append(np.array(Image.open(folder / f"{REFERENCE}.png")))
        gen_list.append(np.array(Image.open(folder / f"{GENERATION}.png")))
        if task == "face_id":
            ref_cond_list.append(np.load(folder / f"{REF_COND}.npy"))
            gen_cond_list.append(np.load(folder / f"{GEN_COND}.npy"))
        else:
            ref_cond_list.append(np.array(Image.open(folder / f"{REF_COND}.png")))
            gen_cond_list.append(np.array(Image.open(folder / f"{GEN_COND}.png")))
    return ref_list, gen_list, ref_cond_list, gen_cond_list


if __name__ == "__main__":
    args = parse_arguments()
    data_list = sorted(glob.glob(f"outputs/condition_gen/{args.dataset}/{args.task}/{args.method}/*"))

    if not data_list:
        print("No data found in the specified directory.")
        exit()

    ref_list, gen_list, ref_cond_list, gen_cond_list = load_results(data_list[:args.num], args.task)

    fid = cal_fid(gen_list, ref_list, device="cuda")
    kid = cal_kid(gen_list, ref_list, device="cuda")
    clip_aes = cal_clip_aes(gen_list, device="cuda")

    if args.task == "segmentation":
        cond_name = "IoU"
        cond_value = cal_iou(ref_cond_list, gen_cond_list)
    else:
        cond_name = "L2 Loss"
        cond_value = cal_l2_norm(gen_cond_list, ref_cond_list).mean

    print("=" * 80)
    print(f"Task: {args.task}, Method: {args.method} (N={len(ref_list)})")
    if args.display == "latex":
        print(f"{cond_name:<10}{'FID':<10}{'KID':<10}{'CLIP-aes':<10}")
        print(f"{cond_value:.3f} & {fid:.2f} & {kid:.3f} & {clip_aes:.3f}")
    else:
        print(f"{cond_name:<20}{'FID':<20}{'KID':<20}{'CLIP-aes':<20}")
        print(f"{f'{cond_value:.3f}':<20}{f'{fid:.2f}':<20}{f'{kid:.3f}':<20}{f'{clip_aes:.3f}':<20}")
    print("=" * 80)
