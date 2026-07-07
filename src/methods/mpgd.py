"""
Manifold Preserving Guided Diffusion (MPGD).
Paper: https://arxiv.org/abs/2311.16424
GitHub: https://github.com/KellyYutongHe/mpgd_pytorch
"""

import torch
from torch import nn


class MPGD(nn.Module):
    def __init__(self, loss_fn, scheduler, model, step_size=None, inner_max_iter=1, generator=None, guidance_interval=1, verbose=False):
        super().__init__()
        self.loss_fn = loss_fn
        self.step_size = step_size
        self.scheduler = scheduler
        self.model = model
        self.generator = generator
        self.inner_max_iter = inner_max_iter
        self.guidance_interval = guidance_interval
        self.verbose = verbose

    def step(self, sample, model_output, reference_image, t, eta, index, use_clipped_model_output=True, *args, **kwargs):
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        t_prev = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else self.scheduler.final_alpha_cumprod

        outputs = self.scheduler.step(
            model_output,
            t,
            sample,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            generator=self.generator,
        )

        prev_sample = outputs.prev_sample
        original_sample = outputs.pred_original_sample
        original_sample.requires_grad_(True)

        if index % self.guidance_interval == 0 and t_prev >= 0:
            for _ in range(self.inner_max_iter):
                loss = self.loss_fn(original_sample, reference_image)
                grad = torch.autograd.grad(outputs=loss, inputs=original_sample, create_graph=True)[0]
                step_size = self.step_size / alpha_prod_t ** 0.5
                original_sample = original_sample - step_size * grad

        epsilon = (sample - (alpha_prod_t ** 0.5) * original_sample) / (1 - alpha_prod_t) ** 0.5
        variance = self.scheduler._get_variance(t, t_prev)
        std_dev_t = eta * variance ** (0.5)
        sample_mean = alpha_prod_t_prev ** (0.5) * original_sample + \
                (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * epsilon
        prev_sample = sample_mean + std_dev_t * torch.randn_like(sample)

        if self.verbose:
            print(f"step size: {self.step_size}")
            print(f"loss: {loss.item()}")
            print(f"gradient norm: {torch.linalg.norm(grad).item()}")

        timesteps_prev = t_prev
        if timesteps_prev > 0:
            model_output = self.model(prev_sample, timesteps_prev).sample
            original_sample = self.scheduler.step(
                model_output,
                timesteps_prev,
                prev_sample,
                eta=eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=self.generator,
            ).pred_original_sample
        else:
            original_sample = prev_sample

        return prev_sample.detach(), original_sample.detach()
