import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .models.bisenet import BiSeNet


BISENET_CHECKPOINT = "src/losses/checkpoints/bisnet.pth"

# Color palette for visualizing the 19-class CelebAMask-HQ face parsing maps
SEGMENTATION_COLORS = [
    [255, 255, 255],
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 0, 85],
    [255, 0, 170],
    [0, 255, 0],
    [85, 255, 0],
    [170, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [255, 255, 0],
    [255, 255, 85],
    [255, 255, 170],
    [255, 0, 255],
    [255, 85, 255],
    [255, 170, 255],
    [0, 255, 255],
    [85, 255, 255],
]


class SegmentationProcessor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def forward(self, x):
        assert x.min() >= 0 and x.max() <= 1, "Input tensor must be in [0, 1]"
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        return (x - self.mean) / self.std


class FaceSegmentationLoss(nn.Module):
    """Cross-entropy between the BiSeNet parsing of the generated image (in [-1, 1])
    and the parsing map of the reference image (in [0, 1])."""

    def __init__(
        self,
        device='cuda:0',
        requires_grad=False,
    ):
        super().__init__()
        self.device = device
        self.image_processor = SegmentationProcessor(device=device)
        self.segmentor = BiSeNet(n_classes=19, weight=BISENET_CHECKPOINT).to(self.device)

        if requires_grad is False:
            for param in self.segmentor.parameters():
                param.requires_grad = False

        self.segmentor.eval()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.image_processor((x + 1) / 2)
        y = self.image_processor(y)
        x_pred = self.segmentor(x)[0]
        y_mask = self.segmentor(y)[0].argmax(dim=1).long()
        return F.cross_entropy(x_pred, y_mask).mean()


def colorize_mask(seg_map: np.ndarray) -> np.ndarray:
    """Convert a (H, W) parsing map of class indices to an RGB image."""
    seg_map_img = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(SEGMENTATION_COLORS):
        seg_map_img[seg_map == idx] = color
    return seg_map_img


def visualize_segmentation(seg_map: np.ndarray, image: Image.Image) -> Image.Image:
    """Blend a colorized parsing map onto the input image."""
    seg_img = Image.fromarray(colorize_mask(seg_map), mode="RGB")
    return Image.blend(image.convert("RGB").resize(seg_img.size), seg_img, alpha=0.6)
