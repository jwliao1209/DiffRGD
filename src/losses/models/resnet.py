import torch
import torch.nn as nn
import torch.utils.model_zoo as modelzoo


RESNET18_URL = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + shortcut)


def create_layer_basic(in_channels: int, out_channels: int, n_blocks: int, stride: int = 1):
    layers = [ConvBlock(in_channels, out_channels, stride)]
    layers += [ConvBlock(out_channels, out_channels) for _ in range(n_blocks - 1)]
    return nn.Sequential(*layers)


class ResNet18(nn.Module):
    def __init__(self, use_pretrained: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = create_layer_basic(64, 64, n_blocks=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, n_blocks=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, n_blocks=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, n_blocks=2, stride=2)

        if use_pretrained:
            self.load_pretrained_weight()

    def load_pretrained_weight(self) -> None:
        pretrained = modelzoo.load_url(RESNET18_URL)
        pretrained = {k: v for k, v in pretrained.items() if not k.startswith('fc.')}
        self.load_state_dict(pretrained, strict=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)

        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32
