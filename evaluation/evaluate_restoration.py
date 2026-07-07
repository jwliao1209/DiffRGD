"""Evaluate image restoration results with PSNR / SSIM / LPIPS / FID (Tables 1-3)."""

import glob
import sys
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running this script directly from anywhere (src/ lives at the repo root)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.constants import GT, RECONSTRUCTION
from src.metrics import cal_psnr, cal_ssim, cal_fid, cal_lpips
from src.utils import set_random_seed


warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics.functional.image.lpips")


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-t", "--task", type=str, default="random_inpainting")
    parser.add_argument("-m", "--method", type=str, default="diffrgd")
    parser.add_argument("-d", "--dataset", type=str, default="ffhq")
    parser.add_argument("-n", "--num", type=int, default=100)
    parser.add_argument("--display", type=str, default="plain", choices=["plain", "latex"])
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()
    data_list = sorted(glob.glob(f"outputs/image_restoration/{args.dataset}/{args.task}/{args.method}/*"))

    if not data_list:
        print("No data found in the specified directory.")
        exit()

    gt_list = []
    reconstruction_list = []
    for folder in data_list[:args.num]:
        gt_list.append(np.array(Image.open(Path(folder, f"{GT}.png"))))
        reconstruction_list.append(np.array(Image.open(Path(folder, f"{RECONSTRUCTION}.png"))))

    psnr = cal_psnr(reconstruction_list, gt_list)
    ssim = cal_ssim(reconstruction_list, gt_list)
    lpips = cal_lpips(reconstruction_list, gt_list, device="cuda")
    fid = cal_fid(reconstruction_list, gt_list, device="cuda")

    print("=" * 80)
    print(f"Task: {args.task}, Method: {args.method} (N={len(gt_list)})")
    if args.display == "latex":
        print(f"{'PSNR':<20}{'SSIM':<20}{'LPIPS':<20}{'FID'}")
        print(f"{psnr.mean:.2f} $\\pm$ {psnr.std:.2f} & "
              f"{ssim.mean:.3f} $\\pm$ {ssim.std:.3f} & "
              f"{lpips.mean:.3f} $\\pm$ {lpips.std:.3f} & "
              f"{fid:.2f}")
    else:
        print(f"{'PSNR':<20}{'SSIM':<20}{'LPIPS':<20}{'FID':<20}")
        print(f"{f'{psnr.mean:.2f} ± {psnr.std:.2f}':<20}"
              f"{f'{ssim.mean:.3f} ± {ssim.std:.3f}':<20}"
              f"{f'{lpips.mean:.3f} ± {lpips.std:.3f}':<20}"
              f"{f'{fid:.2f}':<20}")
    print("=" * 80)
