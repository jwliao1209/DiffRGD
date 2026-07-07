"""
Diffusion with Spherical Gaussian constraint (DSG).
Paper: https://arxiv.org/abs/2402.03201
GitHub: https://github.com/LingxiaoYang2023/DSG2024
"""

import torch
from torch import nn


class DSG(nn.Module):
    def __init__(self, loss_fn, scheduler, model, step_size=None, generator=None, guidance_interval=1, verbose=False):
        super().__init__()
        self.loss_fn = loss_fn
        self.step_size = step_size
        self.scheduler = scheduler
        self.model = model
        self.generator = generator
        self.guidance_interval = guidance_interval
        self.verbose = verbose

    def step(self, sample, model_output, reference_image, t, eta, index, use_clipped_model_output=True, eps=1e-8, *args, **kwargs):
        sample = sample.clone().detach().requires_grad_(True)
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

        t_prev = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

        if index % self.guidance_interval == 0 and t_prev >= 0:
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else self.scheduler.final_alpha_cumprod
            variance = self.scheduler._get_variance(t, t_prev)
            std_dev_t = eta * variance ** (0.5)
            prev_sample_mean = alpha_prod_t_prev ** (0.5) * original_sample + \
                (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * model_output

            # Compute loss
            loss = self.loss_fn(original_sample, reference_image)
            grad = torch.autograd.grad(outputs=loss, inputs=sample)[0]
            _, c, h, w = sample.shape
            r = torch.sqrt(torch.tensor(c * h * w)) * std_dev_t
            d_star = -r * grad / (torch.linalg.norm(grad) + eps)
            d_sample = prev_sample - prev_sample_mean
            mix_direction = d_sample + self.step_size * (d_star - d_sample)
            mix_direction_norm = torch.linalg.norm(mix_direction)
            prev_sample = prev_sample_mean + r * mix_direction / (mix_direction_norm + eps)

            if self.verbose:
                print(f"timestep: {t}")
                print(f"prev_timestep: {t_prev}")
                print(f"step size: {self.step_size}")
                print(f"loss: {loss.item()}")
                print(f"gradient norm: {torch.linalg.norm(grad).item()}")

        timesteps_prev = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
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
