"""
Diffusion Posterior Sampling (DPS).
Paper: https://arxiv.org/abs/2209.14687
GitHub: https://github.com/DPS2022/diffusion-posterior-sampling

For conditional generation with an energy-based guidance loss, this update rule
is equivalent to FreeDoM (https://arxiv.org/abs/2303.09833).
"""

import torch
from torch import nn


class DPS(nn.Module):
    def __init__(self, loss_fn, scheduler, model, step_size=None, generator=None, guidance_interval=1, verbose=False):
        super().__init__()
        self.loss_fn = loss_fn
        self.step_size = step_size
        self.scheduler = scheduler
        self.model = model
        self.generator = generator
        self.guidance_interval = guidance_interval
        self.verbose = verbose

    def step(self, sample, model_output, reference_image, t, eta, index, use_clipped_model_output=True, *args, **kwargs):
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

        if index % self.guidance_interval == 0:
            # Compute loss
            loss = self.loss_fn(original_sample, reference_image)
            grad = torch.autograd.grad(outputs=loss, inputs=sample)[0]

            prev_sample = prev_sample - self.step_size * grad

            if self.verbose:
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
