import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class CLIPImageProcessor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def forward(self, x, size=(224, 224)):
        assert x.min() >= 0 and x.max() <= 1, "Input tensor must be in [0, 1]"
        x = F.interpolate(x, size=size, mode='bicubic', align_corners=False)
        return (x - self.mean) / self.std


class StyleLoss(nn.Module):
    def __init__(
        self,
        model_name='openai/clip-vit-base-patch16',
        device='cuda:0',
        requires_grad=False
    ):
        super().__init__()
        self.device = device
        self.image_processor = CLIPImageProcessor(device=device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)

        if requires_grad is False:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def get_clip_features(self, x):
        return self.model.vision_model(
            x,
            output_hidden_states=True,
        ).hidden_states[2][:, 1:] # remove CLS token

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.image_processor((x.clamp(-1, 1) + 1) / 2, size=(224, 224))
        y = self.image_processor(y, size=(224, 224))
        x_feat = self.get_clip_features(x)
        y_feat = self.get_clip_features(y)
        Gx = torch.bmm(x_feat.transpose(1, 2), x_feat).view(x.size(0), -1)
        Gy = torch.bmm(y_feat.transpose(1, 2), y_feat).view(y.size(0), -1)
        # return (torch.linalg.norm(Gx - Gy, dim=(1, 2))).mean()
        return ((Gx - Gy) ** 2).sum(dim=1).sqrt() / 100


if __name__ == "__main__":
    image1 = torch.rand((1, 3, 512, 512)).cuda()
    image2 = torch.rand((1, 3, 512, 512)).cuda()
    loss_fn = StyleLoss()
    loss = loss_fn(image1, image2)
    print(loss.item())
