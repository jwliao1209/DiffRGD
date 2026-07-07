import torch
import torch.nn.functional as F
from torch import nn

from .resnet import ResNet18


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self) -> None:
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if not layer.bias is None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class BiSeNetBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels, n_classes: int):
        super(BiSeNetBlock, self).__init__()
        self.conv = ConvBlock(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_channels, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def init_weight(self) -> None:
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.conv(x))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_channels, out_channels, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_channels)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def init_weight(self) -> None:
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        return torch.mul(feat, atten)


class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        self.resnet = ResNet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_head16 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_avg = ConvBlock(512, 128, kernel_size=1, stride=1, padding=0)

        self.init_weight()

    def init_weight(self) -> None:
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1 = nn.Conv2d(
            out_channels,
            out_channels // 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_channels // 4,
            out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()
    
    def init_weight(self) -> None:
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, fsp: torch.Tensor, fcp: torch.Tensor) -> torch.Tensor:
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        return feat_atten + feat


class BiSeNet(nn.Module):
    def __init__(self, n_classes: int, weight: bool = None):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetBlock(256, 256, n_classes)
        self.conv_out16 = BiSeNetBlock(128, 64, n_classes)
        self.conv_out32 = BiSeNetBlock(128, 64, n_classes)
        self.init_weight()

        if weight is not None:
            self.load_state_dict(torch.load(weight))
    
    def init_weight(self) -> None:
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = feat_res8
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32
