import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel


MLP_PATH = "src/metrics/clip_aes_ckpt.pth"


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        self.mlp.load_state_dict(torch.load(MLP_PATH, weights_only=True))
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt")
        embed = self.clip.get_image_features(**{k: v.to(x.device) for k, v in inputs.items()})
        return self.mlp(embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)).squeeze(1)
