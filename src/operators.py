"""Degradation operators A(.) for the linear inverse problems y = A(x) + n."""

from typing import Optional

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torch import nn


class DenoisingOperator(nn.Module):
    def __init__(self, image_size: tuple = (256, 256)):
        super().__init__()

    def reset_operator(self):
        pass

    def forward(self, x):
        return x


class BoxInpaintingOperator(nn.Module):
    def __init__(
        self,
        image_size: tuple = (256, 256),
        mask_size: tuple = (16, 16),
        mask: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.image_size = image_size
        self.mask_size = mask_size
        self.device = device
        self.mask = mask.to(device) if mask is not None else self._generate_mask()

    def _generate_mask(self):
        image_height, image_width = self.image_size
        mask_height, mask_width = self.mask_size
        top = torch.randint(0, image_height - mask_height, (1,))
        left = torch.randint(0, image_width - mask_width, (1,))
        mask = torch.ones((image_height, image_width), dtype=torch.float32)
        mask[top : top + mask_height, left : left + mask_width] = 0.
        return mask.view(1, 1, image_height, image_width).to(self.device)

    def reset_operator(self):
        self.mask = self._generate_mask()

    def forward(self, x):
        return self.mask.to(x.dtype) * x
    

class RandomInpaintingOperator(nn.Module):
    def __init__(
        self,
        image_size: tuple = (256, 256),
        prob_range: tuple = (0.3, 0.7),
        mask: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.image_size = image_size
        self.prob_range = prob_range
        self.device = device
        self.mask = mask.to(device) if mask is not None else self._generate_mask()

    def _generate_mask(self):
        image_height, image_width = self.image_size
        total_pixels = image_height * image_width

        # Generate mask probability
        prob = torch.empty(1).uniform_(*self.prob_range).item()
        num_masked = int(total_pixels * prob)

        # Create a flat mask
        mask = torch.ones(total_pixels)
        indices = torch.randperm(total_pixels)[:num_masked]
        mask[indices] = 0

        return mask.view(1, 1, image_height, image_width).to(self.device)

    def reset_operator(self):
        self.mask = self._generate_mask()

    def forward(self, x):
        return self.mask.to(x.dtype) * x


class SuperResolutionOperator(nn.Module):
    def __init__(
        self,
        size: tuple = (64, 64),
        device: str = "cuda",
    ):
        super().__init__()
        self.size: tuple[int, int] = tuple(size)

    def reset_operator(self):
        pass

    def forward(self, x):
        return F.interpolate(
            x,
            size=self.size,
            mode='bicubic',
            align_corners=True,
            antialias=True,
        )


class GaussianBlurOperator(nn.Module):
    def __init__(
        self,
        kernel_size: int = 31,
        std: float = 3.0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.std = std

        self.pad = nn.ReflectionPad2d(self.kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            bias=False,
            groups=3,
        )
        self._setup_filter_weights()

    def _setup_filter_weights(self) -> None:
        init_filter = np.zeros((self.kernel_size, self.kernel_size))
        init_filter[self.kernel_size // 2,self.kernel_size // 2] = 1
        gaussian_filter = scipy.ndimage.gaussian_filter(init_filter, sigma=self.std)
        self.conv.weight = nn.Parameter(
            torch.from_numpy(gaussian_filter). \
                view(1, 1, *gaussian_filter.shape). \
                expand(3, 1, *gaussian_filter.shape). \
                to(torch.float32),
            requires_grad=False,
        )
    
    def reset_operator(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class MotionBlurOperator(nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int] = (61, 61),
        intensity: float = 0.5,
        kernel: Optional[torch.Tensor] = None,
        eps: float = 0.1,
    ):
        super().__init__()
        if len(kernel_size) != 2 or not all(isinstance(s, int) for s in kernel_size):
            raise ValueError("Size must be tuple of 2 positive INTEGERS")
        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError("Size must be tuple of 2 POSITIVE integers")

        if not isinstance(intensity, (int, float, np.float32, np.float64)):
            raise ValueError("Intensity must be a number between 0 and 1")
        if intensity < 0 or intensity > 1:
            raise ValueError("Intensity must be a number between 0 and 1")

        self.kernel_size = kernel_size
        self.motion_intensity = intensity
        self.canvas_size = tuple(2 * i for i in kernel_size)
        self.canvas_width, self.canvas_height = self.canvas_size
        self.canvas_diagonal = (self.canvas_width ** 2 + self.canvas_height ** 2) ** 0.5
        self.eps = eps

        self.pad = nn.ReflectionPad2d(self.kernel_size[0] // 2)
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            bias=False,
            groups=3,
        )
        self._setup_filter_weights(kernel)

    def _generate_path(self):
        max_path_length = 0.75 * self.canvas_diagonal * (
            np.random.uniform() + np.random.uniform(0, self.motion_intensity ** 2)
        )
        step_lengths = []
        while sum(step_lengths) < max_path_length:
            step = np.random.beta(1, 30) * (1 - self.motion_intensity + self.eps) * self.canvas_diagonal
            if step < max_path_length:
                step_lengths.append(step)
        num_steps = len(step_lengths)
        self.step_lengths = np.asarray(step_lengths)

        max_turn_angle = np.random.uniform(0, self.motion_intensity * np.pi)
        sign_flip_prob = np.random.beta(2, 20)
        turn_angles = [np.random.uniform(low=-max_turn_angle, high=max_turn_angle)]
        while len(turn_angles) < num_steps:
            angle = np.random.triangular(0, self.motion_intensity * max_turn_angle, max_turn_angle + self.eps)
            if np.random.uniform() < sign_flip_prob:
                angle *= -np.sign(turn_angles[-1])
            else:
                angle *= np.sign(turn_angles[-1])
            turn_angles.append(angle)
        self.turn_angles = np.asarray(turn_angles)

        increments = self.step_lengths * np.exp(1j * self.turn_angles)
        complex_path = np.cumsum(increments)
        path_center = sum(complex_path) / num_steps
        canvas_center = (self.canvas_width + 1j * self.canvas_height) / 2
        complex_path -= path_center
        complex_path *= np.exp(1j * np.random.uniform(0, np.pi))
        complex_path += canvas_center
        self.path_coords = [(pt.real, pt.imag) for pt in complex_path]

    def _setup_filter_weights(self, kernel=None) -> None:
        if kernel is None:
            self._generate_path()
            kernel_image = Image.new("RGB", self.canvas_size)
            painter = ImageDraw.Draw(kernel_image)
            painter.line(xy=self.path_coords, width=int(self.canvas_diagonal / 150))
            blur_radius = int(self.canvas_diagonal * 0.01)

            kernel_image = kernel_image. \
                filter(ImageFilter.GaussianBlur(radius=blur_radius)). \
                resize(self.kernel_size, resample=Image.LANCZOS). \
                convert("L")
            kernel = np.asarray(kernel_image, dtype=np.float32)
            kernel = kernel / np.sum(kernel)
        self.kernel = torch.from_numpy(kernel).view(1, 1, *kernel.shape) if isinstance(kernel, np.ndarray) else kernel
        self.conv.weight = nn.Parameter(
            self.kernel.expand(3, 1, -1, -1).to(torch.float32),
            requires_grad=False,
        )

    def reset_operator(self):
        self._setup_filter_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


