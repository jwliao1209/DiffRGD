from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore

from .clip_aes import AestheticScorer
from .iou import iou
from .psnr import psnr
from .ssim import ssim
from src.losses.style_loss import StyleLoss


@dataclass
class MetricOutputs:
    mean: Optional[np.float64] = None
    std: Optional[np.float64] = None


def cal_iou(x_list: List[np.ndarray], y_list: List[np.ndarray]) -> np.float64:
    iou_list = []
    for x, y in zip(x_list, y_list):
        assert x.shape == y.shape, "Input array must have the same shape"
        iou_list.append(iou(x, y))
    return np.mean(iou_list)


def cal_psnr(x_list: List[np.ndarray], y_list: List[np.ndarray]) -> MetricOutputs:
    psnr_list = []
    for x, y in zip(x_list, y_list):
        assert x.shape == y.shape, "Input array must have the same shape"
        psnr_list.append(psnr(x, y))
    return MetricOutputs(mean=np.mean(psnr_list), std=np.std(psnr_list))


def cal_ssim(x_list: List[np.ndarray], y_list: List[np.ndarray]) -> MetricOutputs:
    ssim_list = []
    for x, y in zip(x_list, y_list):
        assert x.shape == y.shape, "Input array must have the same shape"
        ssim_list.append(ssim(x, y))
    return MetricOutputs(mean=np.mean(ssim_list), std=np.std(ssim_list))


def cal_lpips(x_list: List[np.ndarray], y_list: List[np.ndarray], net_type: str = "vgg", device: str = "cpu") -> MetricOutputs:
    lpips_list = []
    lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type).to(device)
    for x, y in zip(x_list, y_list):
        assert x.shape == y.shape, "Input array must have the same shape"

        x = x / 255.0 if x.max() > 1.0 else x
        y = y / 255.0 if y.max() > 1.0 else y

        x = x * 2 - 1
        y = y * 2 - 1

        lpips_list.append(
            lpips(
                torch.tensor(x).permute(2, 0, 1)[None].to(torch.float32).to(device),
                torch.tensor(y).permute(2, 0, 1)[None].to(torch.float32).to(device),
            ).cpu().numpy()
        )

    return MetricOutputs(mean=np.mean(lpips_list), std=np.std(lpips_list))


def cal_fid(x_list: List[np.ndarray], y_list: List[np.ndarray], device: str = "cpu") -> np.float64:
    fid_func = FrechetInceptionDistance(feature=2048).to(device)

    for x, y in zip(x_list, y_list):
        assert x.shape == y.shape, "Input array must have the same shape"
        fid_func.update(torch.tensor(x).permute(2, 0, 1)[None].to(device), real=False)
        fid_func.update(torch.tensor(y).permute(2, 0, 1)[None].to(device), real=True)
    return np.float64(fid_func.compute().cpu())


def cal_kid(x_list: List[np.ndarray], y_list: List[np.ndarray], device: str = "cpu") -> np.float64:
    kid_func = KernelInceptionDistance(feature=2048, subset_size=len(x_list)).to(device)

    for x, y in zip(x_list, y_list):
        assert x.shape == y.shape, "Input array must have the same shape"
        kid_func.update(torch.tensor(x).permute(2, 0, 1)[None].to(device), real=False)
        kid_func.update(torch.tensor(y).permute(2, 0, 1)[None].to(device), real=True)
    return np.float64(kid_func.compute()[0].cpu())


def cal_clip_aes(x_list: List[np.ndarray], device: str = "cpu") -> np.float64:
    aes_scorer = AestheticScorer().to(device)
    scores = []
    for x in x_list:
        x = torch.tensor(x).permute(2, 0, 1).to(device)
        scores.append(aes_scorer(x).cpu().numpy())

    return np.mean(scores)


def cal_clip_score(x_list: List[np.ndarray], text_list: List[str], device: str = "cpu") -> np.float64:
    clip_scorer = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    scores = []
    for x, text in zip(x_list, text_list):
        x = torch.tensor(x).permute(0, 1, 2).unsqueeze(0).to(device)
        scores.append(clip_scorer(x, text).detach().cpu() / 100)
    return np.mean(np.float64(scores))


def cal_style_score(x_list: List[np.ndarray], style_list: List[np.ndarray], device: str = "cpu") -> np.float64:
    style_loss_fn = StyleLoss(device=device)
    scores = []
    for x, style in zip(x_list, style_list):
        x = torch.Tensor(x).permute(2, 0, 1).unsqueeze(0).to(device) / 255
        x = x * 2 - 1
        style = torch.Tensor(style).permute(2, 0, 1).unsqueeze(0).to(device) / 255
        score = style_loss_fn(x, style).detach().cpu().numpy()
        scores.append(score)
    return np.mean(scores)


def cal_l2_norm(x_list: List[np.ndarray], y_list: List[np.ndarray]) -> MetricOutputs:
    l2_norm_list = []
    for x, y in zip(x_list, y_list):
        assert x.shape == y.shape, "Input array must have the same shape"
        x = x / 255.0 if x.max() > 128 / 255 else x
        y = y / 255.0 if y.max() > 128 / 255 else y
        l2_norm_list.append(np.linalg.norm(x - y))
    return MetricOutputs(mean=np.mean(l2_norm_list), std=np.std(l2_norm_list))
