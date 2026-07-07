import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.sketchnet import SketchNet


class SketchProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.resize(x, (256, 256), mode='bilinear', align_corners=False)


class FaceSketchLoss(nn.Module):
    def __init__(
        self,
        device='cuda:0',
        requires_grad=False,
    ):
        super().__init__()
        self.device = device
        self.image_processor = SketchProcessor()
        self.sketchnet = SketchNet(in_channels=3, out_channels=1, num_downsampling=8, base_filters=64, use_dropout=False)
        ckpt = torch.load("src/losses/checkpoints/sketchnet.pth", weights_only=True)
        self.sketchnet.load_state_dict(ckpt)
        self.sketchnet.to(self.device)

        if requires_grad is False:
            for param in self.sketchnet.parameters():
                param.requires_grad = False

        self.sketchnet.eval()
    
    def get_condition(self, x: torch.Tensor) -> torch.Tensor:
        if x.min() > -1 / 255:  # x is in [0, 1]
            x = x.clamp(0, 1)
            x = x * 2 - 1  # normalize to [-1, 1]
        assert -1 <= x.min() and x.max() <= 1, f"Input images must be normalized to [-1, 1] not {x.min()} to {x.max()}"
        return self.sketchnet(x)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape, "Input images must have the same shape"
        x_pred = self.get_condition(x)
        y_pred = self.get_condition(y)

        # from PIL import Image
        # Image.fromarray((x_pred.squeeze(0) * 255).clamp(0, 255).cpu().numpy().astype('uint8'), mode="L").save('mask.png')
        return torch.linalg.norm(x_pred - y_pred)


if __name__ == "__main__":
    from diffusers.utils import load_image
    from torchvision.transforms.functional import to_tensor

    image1 = to_tensor(load_image("data/celeba_hq/00000.jpg")).unsqueeze(0).cuda()
    loss_fn = FaceSketchLoss()
    loss = loss_fn(image1, image1)
    print(loss.item())
