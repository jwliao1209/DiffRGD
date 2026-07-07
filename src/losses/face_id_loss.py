from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def l2_norm(x, axis=1):
    norm = torch.norm(x, 2, axis, True)
    return torch.div(x, norm)


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """ A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + \
        [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    else:
        raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
    return blocks


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        res = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return res * x


class BottleNeckIR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(BottleNeckIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False), nn.BatchNorm2d(depth)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class BottleNeckIR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(BottleNeckIR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Backbone(nn.Module):
    def __init__(self, input_size, num_layers, mode='ir', drop_ratio=0.4, affine=True):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"

        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = BottleNeckIR
        elif mode == 'ir_se':
            unit_module = BottleNeckIR_SE

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        if input_size == 112:
            self.output_layer = nn.Sequential(
                nn.BatchNorm2d(512),
                nn.Dropout(drop_ratio),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 512),
                nn.BatchNorm1d(512, affine=affine)
            )
        else:
            self.output_layer = nn.Sequential(
                nn.BatchNorm2d(512),
                nn.Dropout(drop_ratio),
                nn.Flatten(),
                nn.Linear(512 * 14 * 14, 512),
                nn.BatchNorm1d(512, affine=affine)
            )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel,
                        bottleneck.depth,
                        bottleneck.stride
                    )
                )
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


def get_model(model_name, input_size):
    if model_name == 'ir_se50':
        return Backbone(input_size, num_layers=50, mode='ir_se', drop_ratio=0.4, affine=False)
    elif model_name == 'ir_se101':
        return Backbone(input_size, num_layers=100, mode='ir_se', drop_ratio=0.4, affine=False)
    elif model_name == 'ir_se152':
        return Backbone(input_size, num_layers=152, mode='ir_se', drop_ratio=0.4, affine=False)
    elif model_name == 'ir50':
        return Backbone(input_size, num_layers=50, mode='ir', drop_ratio=0.4, affine=False)
    elif model_name == 'ir101':
        return Backbone(input_size, num_layers=100, mode='ir', drop_ratio=0.4, affine=False)
    elif model_name == 'ir152':
        return Backbone(input_size, num_layers=152, mode='ir', drop_ratio=0.4, affine=False)
    else:
        raise ValueError("Invalid model name: {}. Must be one of [ir_se50, ir_se101, ir_se152, ir50, ir101, ir152]".format(model_name))


class FaceIDLoss(nn.Module):
    def __init__(self, device, pretrained_path="src/losses/checkpoints/model_ir_se50.pth", requires_grad=False):
        super(FaceIDLoss, self).__init__()
        self.facenet = Backbone(
            input_size=112,
            num_layers=50,
            drop_ratio=0.6,
            mode='ir_se',
        ).to(device)
        self._load_pretrained_weights(pretrained_path)
        self.face_pool = nn.AdaptiveAvgPool2d((112, 112))

        self.facenet.eval()
        if requires_grad is False:
            for param in self.facenet.parameters():
                param.requires_grad = False

    def _load_pretrained_weights(self, pretrained_path):
        state_dict = torch.load(pretrained_path, weights_only=False)
        self.facenet.load_state_dict(state_dict)

    def _extract_feats(self, x):
        if x.shape[1] != 256 or x.shape[2] != 256:
            x = F.interpolate(x, size=(256, 256), mode='bicubic', align_corners=False)
        x = x[:, :, 35:223, 32:220]
        x = self.face_pool(x)
        return self.facenet(x)
    
    def get_condition(self, x: torch.Tensor) -> torch.Tensor:
        if x.min() > -1 / 255:  # x is in [0, 1]
            x = x.clamp(0, 1)
            x = x * 2 - 1  # normalize to [-1, 1]
        assert -1 <= x.min() and x.max() <= 1, f"Input images must be normalized to [-1, 1] not {x.min()} to {x.max()}"
        return self._extract_feats(x)

    def forward(self, x, y):
        assert x.shape == y.shape, "Input images must have the same shape"
        x_feat = self.get_condition(x)
        y_feat = self.get_condition(y)
        return torch.linalg.norm(x_feat - y_feat)


if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256)).cuda()
    y = torch.rand((1, 3, 256, 256)).cuda()
    loss_fn = FaceIDLoss().cuda()
    loss = loss_fn(x, y)
    print(loss.item())
